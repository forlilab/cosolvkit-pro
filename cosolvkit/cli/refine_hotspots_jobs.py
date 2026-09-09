#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Turn annotated hotspots into runnable AutoPath refinement jobs.
#

import logging
import os
import subprocess
import sys

import numpy as np

from cosolvkit.analysis.sites.mmgbsa import (
    amber_exclude_mask,
    check_single_topology,
    select_frames,
    write_frame_trajectory,
)
from cosolvkit.analysis.sites.poses import write_pose

logger = logging.getLogger(__name__)

# AutoPath's own default (``ap_PLIP.prepare_mmgbsa_batch``). Amber names the monatomic
# ions NA and CL, not "Na+"/"Cl-": the ``+``/``-`` forms match nothing in a real prmtop
# and leave every counter-ion sitting in the MMGBSA receptor.
SOLVENT_STRIP = ":POP:HOH:WAT:NA:CL:K:MG"

# Minimal MMPBSA input. ``prepare_mmgbsa_batch`` rewrites the commented-out frame lines
# in place (it matches on the leading '#startframe'/'#endframe'/'#interval'), so those
# must stay exactly as written here; ``strip_mask`` is likewise matched by prefix when
# persistent waters are requested.
# Residues sander cannot atom-type for the ICOSA surface area that per-residue
# decomposition forces (MMPBSA sets gbsa=2 whenever idecomp is on). Verified on FosAKP:
# with MN present sander aborts with "bad atom type: Mn" and no usable error in the
# mdout. The other transition metals are the same class and are listed pre-emptively.
# Only relevant to decomposition runs; a plain MMGBSA run keeps them.
DECOMP_UNSUPPORTED_RESNAMES = ("MN", "ZN", "FE", "CO", "NI", "CU", "CD", "HG")

# Below this ligand-to-metal distance, dropping the metal changes the interaction
# energy enough that doing it silently would be dishonest, so the run stops instead.
DECOMP_METAL_MIN_DIST_ANG = 6.0

MMPBSA_IN_TEMPLATE = """\
&general
use_sander=1,

#startframe = 1,
#endframe = 1,
#interval = 1,

# Keep temporary files off; flip to 1 to debug a failed run.
keep_files=0,
debug_printlevel=2,
netcdf=1,

strip_mask= "{strip_mask}"
/
&gb
igb={igb},
saltcon={saltcon},
/
{decomp}"""

# The GB model dictates the radius set; a mismatched pair silently changes the
# energies rather than erroring, so the pairing is encoded here instead of being
# two independent defaults that can drift apart.
IGB_RADII = {
    1: "mbondi",
    2: "mbondi2",
    5: "mbondi2",
    7: "bondi",
    8: "mbondi3",
}
DEFAULT_IGB = 5
DEFAULT_SALTCON = 0.15


def radii_for_igb(igb, override=None):
    """Radius set matching *igb*, or *override* with a warning when it does not match."""
    canonical = IGB_RADII.get(int(igb))
    if override is None:
        if canonical is None:
            raise ValueError(
                f"No canonical radius set known for igb={igb}; pass --radii explicitly."
            )
        return canonical
    if canonical is not None and override != canonical:
        logger.warning(
            "radii=%s is not the canonical set for igb=%s (which is %s); "
            "the energies will not be comparable to runs using the pair.",
            override, igb, canonical,
        )
    return override

# The qfile already passes -do FINAL_DECOMP_mmpbsa.dat, but MMPBSA writes nothing
# without this section. csv_format=1 is what
# ProteinLigandAnalyzer.parse_mmpbsa_deltas_all_components expects.
DECOMP_BLOCK = """\
&decomp
idecomp=1,
dec_verbose=3,
csv_format=1,
/
"""


def target_tag(target, is_binding_site):
    """Directory name for a refinement target.

    Hotspot ``site_id`` is unique only within a cosolvent, so a hotspot tag carries the
    probe name; binding-site ids are global.
    """
    if is_binding_site:
        return f"bs_{target.site_id}"
    return f"hs_{target.cosolvent}_{target.site_id}"


def select_mmgbsa_jobs(hotspot, args, n_frames=None):
    """Pick (representative, selections) pairs for MMGBSA, best-occupied molecule first.

    A job is keyed by molecule because MMPBSA takes a single ligand mask: at a frame
    where a different copy occupied the blob, this molecule is elsewhere in the box.
    Records for the same molecule across replicas pool, since replicas share a topology
    — but frame indices are TRAJECTORY-LOCAL, so a pooled frame must stay attached to
    the record (and therefore the trajectory file) it came from. ``selections`` is
    ``[(ProbeOccupancy, [frame, ...]), ...]``, one entry per source record that
    contributed a chosen frame, ready to hand straight to :func:`write_frame_trajectory`.
    """
    records = list(hotspot.probe_occupancy)
    if args.mmgbsa_source is not None:
        records = [o for o in records if o.source_label == args.mmgbsa_source]
    if not records:
        return []

    by_molecule = {}
    for occ in records:
        by_molecule.setdefault((occ.probe_resname, occ.probe_resid), []).append(occ)

    ranked = sorted(
        by_molecule.items(),
        key=lambda kv: (-sum(o.n_frames_bound for o in kv[1]), kv[0][1]),
    )

    strategy = getattr(args, "mmgbsa_frame_strategy", "random")

    jobs = []
    for (_resname, _resid), occs in ranked[:args.mmgbsa_n_molecules]:
        check_single_topology(occs)
        representative = max(occs, key=lambda o: (o.n_frames_bound, o.source_label))
        selections = _select_selections(occs, args, strategy, n_frames)
        if selections:
            jobs.append((representative, selections))
    return jobs


def _select_selections(occs, args, strategy, n_frames=None):
    """Split the frame budget across records, then let ``select_frames`` do the choosing.

    Frame indices are trajectory-local, so the budget is allocated per record and the
    frames themselves come from :func:`cosolvkit.analysis.sites.mmgbsa.select_frames`,
    which owns the selection strategy (including the reserved ``"cluster"`` hook) and
    the shortfall logging. Allocation draws slots from the pooled candidate list without
    replacement, so a record contributes in proportion to how long it was bound — the
    same distribution the pooled draw gave — but nothing is ever merged across records.
    """
    counts = [len(occ.frames) for occ in occs]
    total = sum(counts)
    # n_frames=0 means every occupied frame — LIE is cheap enough to want them all.
    budget = args.mmgbsa_n_frames if n_frames is None else n_frames
    n = total if not budget else min(budget, total)
    if n <= 0:
        return []

    owners = np.repeat(np.arange(len(occs)), counts)
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(owners, size=n, replace=False)
    per_record = np.bincount(picked, minlength=len(occs))

    selections = []
    for i, occ in enumerate(occs):
        k = int(per_record[i])
        if k == 0:
            continue
        selections.append((occ, select_frames(occ, k, seed=args.seed,
                                              strategy=strategy)))
    return selections


def render_autopath_script(manifest, mmgbsa, mode):
    """Render the per-target driver that calls AutoPath directly.

    ``AutoPath`` is imported from ``autopath.autopath_core``: it is not registered in
    ``autopath/__init__.py``'s lazy-import table, so ``from autopath import AutoPath``
    raises AttributeError.
    """
    lines = [
        "#!/usr/bin/env python",
        '"""Generated by refine_hotspots. Runs on a compute node, not a login node."""',
        "",
        "import MDAnalysis as mda",
        "",
        f"POSE_PDB = {manifest['pose_pdb']!r}",
        f"LIGAND_SELECTION = {manifest['ligand_selection']!r}",
        f"POCKET_SELECTION = {list(manifest['pocket_selection'])!r}",
        f"EXPECTED_POCKET_RESNAMES = {list(manifest['pocket_resnames'])!r}",
        "",
        "",
        "def check_pocket(pose_pdb):",
        '    """Verify POCKET_SELECTION still indexes the residues named in',
        "    EXPECTED_POCKET_RESNAMES in the pose file about to be handed to AutoPath.",
        '    Catches a stale or mismatched manifest before a simulation restrains the',
        '    wrong residues; it does not check AutoPath\'s post-preparation renumbering,',
        '    since the prepared system path is not known here."""',
        "    residues = mda.Universe(pose_pdb).residues",
        "    got = [residues[i - 1].resname for i in POCKET_SELECTION]",
        "    if got != EXPECTED_POCKET_RESNAMES:",
        "        raise SystemExit(",
        '            f"pocket_selection shifted during preparation: expected "',
        '            f"{EXPECTED_POCKET_RESNAMES}, got {got}"',
        "        )",
        "",
        "",
    ]

    if mode in ("smd", "both"):
        lines += [
            "from autopath.autopath_core import AutoPath",
            "",
            "check_pocket(POSE_PDB)",
            "",
            "ap = AutoPath(",
            "    pdb_path=POSE_PDB,",
            "    pocket_selection=POCKET_SELECTION,",
            "    run_preparation=True,",
            "    run_equilibration=True,",
            "    run_sMDpulling=True,",
            "    run_metadynamics=False,",
            ")",
            "ap.run(ligand_selection=LIGAND_SELECTION)",
            "",
        ]

    specs = [] if mmgbsa is None else (
        [mmgbsa] if isinstance(mmgbsa, dict) else list(mmgbsa))

    if mode in ("mmgbsa", "both") and specs:
        lines += [
            "from autopath.ap_PLIP import ProteinLigandAnalyzer",
            "",
            "# NOTE: prepare_mmgbsa_batch only PREPARES the calculation. It writes one",
            "# qfiles_mmgbsa/<sysname>_mmgbsa.q per ligand plus a master",
            "# run_mmgbsa_batch.sh, both relative to this script's working directory,",
            "# and returns without computing anything. Nothing runs until someone",
            "# submits that second stage by hand:",
            "#     cd <this directory> && ./run_mmgbsa_batch.sh",
            "# The SLURM job you are reading now finishes as soon as the inputs exist.",
            "",
        ]
        for spec in specs:
            lines += [
                "ProteinLigandAnalyzer.prepare_mmgbsa_batch(",
                f"    sysname={spec['sysname']!r},",
                f"    prmtop={spec['prmtop']!r},",
                f"    traj_fname={spec['trajectory']!r},",
                f"    ligand_amber_selection={spec['ligand_amber_selection']!r},",
                f"    strip_amber_selection={spec['strip_amber_selection']!r},",
                "    traj_slice=None,",
                f"    mmpbsa_in={spec['mmpbsa_in']!r},",
                f"    output_folder={spec['output_folder']!r},",
                f"    radii={spec['radii']!r},",
                ")",
                "",
            ]

    return "\n".join(lines)


def _slurm_script(tag, script_path, template_path=None, workdir=None, mode="both",
                  python_exe=None):
    """Render the qfile that runs one target's driver.

    The ``cd`` is not cosmetic. AutoPath derives its system name from the PDB basename
    and creates ``<basename>/`` RELATIVE TO THE WORKING DIRECTORY, and
    ``prepare_mmgbsa_batch`` likewise writes ``qfiles_mmgbsa/`` and
    ``run_mmgbsa_batch.sh`` relative to it. Every target's pose is called ``pose.pdb``,
    so without the ``cd`` all N jobs would resolve to the same ``./pose_fixed/`` under
    the submit directory and clobber each other's ``system.pdb``, ``equilibration/``
    and ``sMD/``.

    The interpreter is pinned to an absolute path, defaulting to the one running this
    CLI. A bare ``python`` resolves against the compute node's PATH, which is typically
    a base install without MDAnalysis or autopath, so the job would die on import.
    """
    script_path = os.path.abspath(script_path)
    workdir = os.path.abspath(workdir) if workdir else os.path.dirname(script_path)
    python_exe = python_exe or sys.executable
    if template_path:
        with open(template_path) as fh:
            return (fh.read()
                    .replace("{{SCRIPT}}", script_path)
                    .replace("{{NAME}}", tag)
                    .replace("{{WORKDIR}}", workdir)
                    .replace("{{PYTHON}}", python_exe))

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={tag}",
        f"#SBATCH --output={tag}_%j.out",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --time=24:00:00",
    ]
    if mode in ("smd", "both"):
        lines += [
            "#SBATCH --gres=gpu:1",
            "",
            "# The smd leg runs GPU MD (equilibration + steered pulling), so a GPU is",
            "# requested here. A --mode mmgbsa job only writes MMPBSA inputs and needs",
            "# no GPU at all.",
        ]
    else:
        lines += [
            "",
            "# No --gres: the mmgbsa leg only PREPARES MMPBSA inputs on the CPU. The",
            "# GPU is needed by the smd leg, which this job does not run.",
        ]
    lines += [
        "",
        "# AutoPath and prepare_mmgbsa_batch both write relative to the working",
        "# directory; without this cd every target would collide in the submit dir.",
        f"cd {workdir} || exit 1",
        "",
        "# Absolute interpreter: a bare `python` on a compute node is usually a base",
        "# install with no MDAnalysis and no autopath, and the job dies on import.",
        f"{python_exe} {script_path}",
        "",
    ]
    return "\n".join(lines)


def generate_jobs(config, results, out_dir, args):
    """Write pose, MMGBSA inputs and driver scripts for the top targets.

    :return: paths of the generated SLURM qfiles, in the order they were written.
    """
    # Defensive: every path written into a generated script must be absolute, because
    # the qfile cds into the target directory first. A relative out_dir would otherwise
    # resolve against that new cwd and double (<out>/bs_1/<out>/bs_1/run_autopath.py).
    out_dir = os.path.abspath(out_dir)
    targets = _rank_targets(config, results, args)
    qfiles = []

    for target, is_bs in targets:
        tag = target_tag(target, is_bs)
        tag_dir = os.path.join(out_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        pose_ref = target.best_pose(gap_tolerance=args.gap_tolerance)
        if pose_ref is None:
            logger.warning(
                "Target %s has no occupancy record — no probe was ever inside it; "
                "skipping.", tag)
            continue

        manifest = write_pose(pose_ref, os.path.join(tag_dir, "pose.pdb"))

        mmgbsa_specs = None
        if args.mode in ("mmgbsa", "both"):
            mmgbsa_specs = _build_mmgbsa_inputs(config, target, tag, tag_dir, args)

        script_path = os.path.join(tag_dir, "run_autopath.py")
        with open(script_path, "w") as fh:
            fh.write(render_autopath_script(manifest, mmgbsa_specs, args.mode))

        qfile = os.path.join(tag_dir, "job.slurm")
        with open(qfile, "w") as fh:
            fh.write(_slurm_script(tag, script_path, args.slurm_template,
                                   workdir=tag_dir, mode=args.mode,
                                   python_exe=getattr(args, "python_exe", None)))
        qfiles.append(qfile)

    master = os.path.join(out_dir, "submit_all.sh")
    with open(master, "w") as fh:
        fh.write("#!/bin/bash\n")
        for q in qfiles:
            fh.write(f"sbatch {q}\n")
    os.chmod(master, 0o755)
    logger.info("Wrote %d job(s); master script %s.", len(qfiles), master)

    if args.submit:
        for q in qfiles:
            subprocess.run(["sbatch", q], check=True)
        logger.info("Submitted %d job(s).", len(qfiles))

    return qfiles


def cosolvent_species_for(config, source_label, fallback):
    """Every cosolvent resname present in the simulation labelled *source_label*.

    A ``SimulationEntry`` may list several cosolvents. Only the analysed species was
    ever masked out, so any second species stayed in the MMGBSA receptor and its
    interaction energy was folded into the result.
    """
    for sim in getattr(config, "simulations", []) or []:
        if sim.label == source_label:
            species = [c for c in sim.cosolvents]
            if fallback not in species:
                species.append(fallback)
            return species
    logger.warning(
        "No simulation labelled '%s' in the config; stripping only %s from the MMGBSA "
        "receptor. Any other cosolvent species in that run will remain in it.",
        source_label, fallback,
    )
    return [fallback]


def metals_blocking_decomp(universe, ligand_resid, frame=None):
    """Metal residues that would abort a decomposition run, with their ligand distance.

    :return: ``[(resname, resid, min_distance_to_ligand_ang), ...]``, nearest first.
    """
    present = [r for r in universe.residues
               if r.resname in DECOMP_UNSUPPORTED_RESNAMES]
    if not present:
        return []
    if frame is not None and getattr(universe, "trajectory", None) is not None:
        universe.trajectory[frame]
    ligand = universe.select_atoms(f"resid {ligand_resid}")
    if len(ligand) == 0:
        raise ValueError(
            f"resid {ligand_resid} selects no atoms, so the metal-to-ligand distance "
            f"cannot be measured."
        )
    out = []
    for res in present:
        dist = float(np.linalg.norm(
            res.atoms.positions[:, None, :] - ligand.positions[None, :, :],
            axis=-1).min())
        if not np.isfinite(dist):
            # A topology-only Universe has no real coordinates. Refusing here is the
            # point: an unmeasurable distance must not slip past the proximity gate.
            raise ValueError(
                f"Distance from {res.resname} {res.resid} to resid {ligand_resid} is "
                f"not finite — the Universe has no coordinates. Load it with a "
                f"trajectory before checking metal proximity."
            )
        out.append((res.resname, int(res.resid), dist))
    return sorted(out, key=lambda t: t[2])


def post_strip_ligand_index(universe, strip_resnames, excluded_resids, keep_resid):
    """1-based index of *keep_resid* in the topology AFTER the strip mask is applied.

    ``ante-MMPBSA.py`` applies ``-s`` (strip) FIRST and then resolves ``-n`` (the ligand)
    against the already-stripped complex, which is renumbered from 1. Passing the probe's
    ORIGINAL resid selects nothing there: on FosAKP, formamide 289 lives at position 279
    of the 279-residue stripped complex, so ``-n :289`` produced an EMPTY ligand.prmtop,
    an untouched receptor.prmtop, and MMPBSA then died in ``Strip('')`` with
    "Cannot pass no mask to Strip!" because it derives ligand_mask from that prmtop.

    :param universe: MDAnalysis Universe of the solvated topology.
    :param strip_resnames: residue NAMES removed by the strip mask (solvent, ions).
    :param excluded_resids: resids of the other probe copies, also removed.
    :param keep_resid: the ligand's resid in the original topology.
    :return: its 1-based residue index in the stripped complex.
    :raises ValueError: if the ligand is itself stripped, which would be silent nonsense.
    """
    index = 0
    for res in universe.residues:
        resid = int(res.resid)
        if res.resname in strip_resnames or resid in excluded_resids:
            continue
        index += 1
        if resid == keep_resid:
            return index
    raise ValueError(
        f"resid {keep_resid} does not survive the strip mask, so it cannot be the "
        f"MMGBSA ligand. Check the strip mask and the probe resname."
    )


def _build_mmgbsa_inputs(config, target, tag, tag_dir, args):
    """Extract frame trajectories and describe one prepare_mmgbsa_batch call per molecule.

    One MMGBSA job covers exactly one probe molecule, because MMPBSA takes a single
    ligand mask. Each molecule therefore gets its own subdirectory so their frame
    trajectories, MMPBSA inputs and results never overwrite one another.
    """
    jobs = select_mmgbsa_jobs(target, args)
    if not jobs:
        logger.warning("Target %s has no MMGBSA-eligible molecule; skipping that leg.",
                       tag)
        return None

    import MDAnalysis as mda

    mm_root = os.path.join(tag_dir, "mmgbsa")
    specs = []
    for occ, selections in jobs:
        sysname = f"{tag}_{occ.probe_resname}{occ.probe_resid}_{occ.source_label}"
        mm_dir = os.path.join(mm_root, f"{occ.probe_resname}{occ.probe_resid}")
        os.makedirs(mm_dir, exist_ok=True)
        traj = os.path.join(mm_dir, "frames.dcd")
        write_frame_trajectory(selections, traj)

        u = mda.Universe(occ.topology)
        species = cosolvent_species_for(config, occ.source_label, occ.probe_resname)
        all_resids = set()
        for resname in species:
            all_resids.update(
                int(r) for r in u.select_atoms(f"resname {resname}").resids)
        others = amber_exclude_mask(all_resids, keep=occ.probe_resid)
        strip = SOLVENT_STRIP + others

        # Per-residue decomposition forces gbsa=2 (ICOSA), which sander cannot atom-type
        # for transition metals — it aborts with "bad atom type: Mn". Drop them for
        # decomposition runs only, and refuse when one sits close enough to the ligand
        # that removing it would quietly change the answer.
        decomp_strip = []
        if getattr(args, "decomp", True):
            # `u` is topology-only and has no coordinates, so the proximity check needs
            # a universe with frames. Use the first selected frame of the first record.
            probe_occ, probe_frames = selections[0]
            coord_u = mda.Universe(probe_occ.topology, probe_occ.trajectory)
            blocking = metals_blocking_decomp(coord_u, occ.probe_resid,
                                              frame=probe_frames[0])
            too_close = [m for m in blocking if m[2] < DECOMP_METAL_MIN_DIST_ANG]
            if too_close:
                name, resid, dist = too_close[0]
                raise ValueError(
                    f"{name} {resid} is {dist:.2f} A from {occ.probe_resname} "
                    f"{occ.probe_resid}. Per-residue decomposition cannot run with it "
                    f"present (sander cannot atom-type it), but dropping a metal that "
                    f"close would change the interaction energy. Re-run with "
                    f"--no-decomp to keep the metal, or raise the risk knowingly."
                )
            for name, resid, dist in blocking:
                decomp_strip.append(name)
                logger.warning(
                    "%s: dropping %s %d (%.1f A from the ligand) so per-residue "
                    "decomposition can run; it stays in a --no-decomp run.",
                    sysname, name, resid, dist,
                )
            if decomp_strip:
                strip = strip + ":" + ":".join(sorted(set(decomp_strip)))

        # ante-MMPBSA resolves the ligand against the STRIPPED complex, which is
        # renumbered from 1, so the original resid would select nothing there.
        # decomp_strip must be included: dropping the metals shifts every residue after
        # them, so omitting it here would point the ligand mask at the wrong residue.
        ligand_index = post_strip_ligand_index(
            u,
            strip_resnames=({n for n in SOLVENT_STRIP.strip(":").split(":") if n}
                            | set(decomp_strip)),
            excluded_resids=all_resids - {occ.probe_resid},
            keep_resid=occ.probe_resid,
        )
        logger.info(
            "%s: %s %d is residue %d of the stripped complex (ante-MMPBSA ligand mask).",
            sysname, occ.probe_resname, occ.probe_resid, ligand_index,
        )

        mmpbsa_in = os.path.join(mm_dir, "mmgbsa.in")
        with open(mmpbsa_in, "w") as fh:
            fh.write(MMPBSA_IN_TEMPLATE.format(
                strip_mask=strip,
                igb=int(getattr(args, "igb", DEFAULT_IGB)),
                saltcon=float(getattr(args, "saltcon", DEFAULT_SALTCON)),
                decomp=DECOMP_BLOCK if getattr(args, "decomp", True) else ""))

        specs.append({
            "sysname": sysname,
            "prmtop": occ.topology,
            "trajectory": traj,
            "ligand_amber_selection": f":{ligand_index}",
            "strip_amber_selection": strip,
            "mmpbsa_in": mmpbsa_in,
            "output_folder": mm_dir,
            "radii": radii_for_igb(getattr(args, "igb", DEFAULT_IGB),
                                   getattr(args, "radii", None)),
        })
    return specs


def _load_field_maps(checkpoint_dir, cosolvents):
    """``{cosolvent: (array, origin, delta)}`` from the merged AGFE maps in *checkpoint_dir*.

    Mirrors ``MultiSimulationReport._load_merged_field_maps``: binding-site scoring fuses
    site features over the maps when it has them and falls back to a member-count-biased
    best-of-members maximum when it does not, so omitting them here would let
    refine_hotspots rank a different top-N than ``binding_sites.csv`` reports.
    """
    import numpy as _np
    from gridData import Grid

    maps = {}
    for cos in cosolvents:
        path = os.path.join(checkpoint_dir, f"map_agfe_{cos}.dx")
        if not os.path.isfile(path):
            continue
        g = Grid(path)
        maps[cos] = (_np.asarray(g.grid), _np.asarray(g.origin, dtype=float),
                     _np.asarray(g.delta, dtype=float))
    return maps


def _rank_targets(config, results, args):
    """Top-N refinement targets as ``[(target, is_binding_site), ...]``."""
    if args.target == "hotspots":
        flat = [h for sites in results.values() for h in sites]
        flat.sort(key=lambda h: (h.agfe_min if h.agfe_min is not None else 0.0))
        return [(h, False) for h in flat[:args.top_n]]

    from cosolvkit.analysis.sites.binding_sites import identify_binding_sites

    checkpoint_dir = (getattr(args, "checkpoint", None)
                      or os.path.join(config.out_path, "merged"))
    field_maps = _load_field_maps(checkpoint_dir, list(results))
    missing = [c for c in results if c not in field_maps]
    if missing:
        logger.warning(
            "No merged map_agfe_*.dx in %s for %s; binding-site scoring falls back to a "
            "best-of-members maximum that is biased by member count, so the targets "
            "chosen here may differ from the ranking in binding_sites.csv.",
            checkpoint_dir, ", ".join(sorted(missing)),
        )

    bs_cfg = config.binding_sites
    sites = identify_binding_sites(
        results,
        connectivity=bs_cfg.connectivity,
        weights=bs_cfg.weights,
        merge_tolerance_ang=bs_cfg.merge_tolerance_ang,
        probe_chemotype_overrides=bs_cfg.probe_chemotypes,
        field_maps=field_maps,
    )
    return [(s, True) for s in sites[:args.top_n]]
