#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Annotate hotspots with probe occupancy and emit AutoPath refinement jobs.
#
# Usage:
#   refine_hotspots --config analysis.yaml --top-n 5 --mode both
#

import argparse
import glob
import json
import logging
import os

import pandas as pd

from cosolvkit.analysis.config import AnalysisConfig
from cosolvkit.analysis.sites.detect import HotspotDetector
from cosolvkit.analysis.sites.occupancy import OccupancyAnnotator, apply_chain_reference

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line surface for ``refine_hotspots``."""
    p = argparse.ArgumentParser(
        prog="refine_hotspots",
        description="Annotate detected hotspots with probe occupancy and generate "
                    "AutoPath MMGBSA / steered-MD jobs.",
    )
    p.add_argument("--config", required=True,
                   help="Analysis YAML the hotspots were detected from.")
    p.add_argument("--checkpoint", default=None,
                   help="Directory holding hotspot_checkpoints/ "
                        "(default: <out_path>/merged).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <out_path>/refine).")
    p.add_argument("--target", choices=["binding_sites", "hotspots"],
                   default="binding_sites",
                   help="What to refine (default: binding_sites).")
    p.add_argument("--top-n", type=int, default=5,
                   help="How many targets to refine (default: 5).")
    p.add_argument("--stride", type=int, default=1,
                   help="Trajectory stride for the occupancy scan (default: 1).")
    p.add_argument("--min-atoms", type=int, default=1,
                   help="Heavy atoms inside a blob that count as occupancy (default: 1).")
    p.add_argument("--gap-tolerance", type=int, default=0,
                   help="Sampled frames a molecule may miss without ending an episode.")
    p.add_argument("--annotate-only", action="store_true",
                   help="Stop after annotation; write no poses or jobs.")
    p.add_argument("--collect", action="store_true",
                   help="Collect finished MMGBSA/decomposition results from "
                        "<out>/*/mmgbsa/*/ and attach them to the hotspot checkpoint "
                        "(mmgbsa_delta_total on the Hotspot, per-residue decomposition "
                        "on its PocketResidues), instead of annotating or generating "
                        "jobs. Safe to re-run: re-collecting replaces rather than "
                        "duplicates each result.")
    p.add_argument("--mode", choices=["mmgbsa", "smd", "lie", "both"], default="both",
                   help="Which refinement leg to run (default: both = mmgbsa+smd). "
                        "'lie' runs inline and needs no SLURM job and no --collect.")
    p.add_argument("--lie-n-frames", type=int, default=0,
                   help="Frames per LIE calculation; 0 (default) uses every frame the "
                        "molecule occupied the site, since LIE costs about a second.")
    p.add_argument("--lie-cutoff", type=float, default=6.0,
                   help="Angstrom shell around the ligand forming the LIE environment "
                        "(default: 6.0). Water is kept: LIE scores the ligand against "
                        "its whole surroundings.")
    p.add_argument("--lie-exclude-probes", action="store_true",
                   help="Also exclude the other cosolvent copies from the LIE "
                        "environment. A measured no-op on boxes built with cosolvent "
                        "repulsive forces, which keep probes apart; matters when they "
                        "can aggregate.")
    p.add_argument("--submit", action="store_true",
                   help="sbatch the generated qfiles instead of only writing them.")
    p.add_argument("--slurm-template", default=None,
                   help="Optional SLURM template file. Placeholders {{NAME}}, {{SCRIPT}}, "
                        "{{WORKDIR}} and {{PYTHON}} are substituted; a template must cd "
                        "to {{WORKDIR}} itself. Omitted, a built-in block is written.")
    p.add_argument("--igb", type=int, default=5, choices=[1, 2, 5, 7, 8],
                   help="Generalized Born model for MMGBSA (default: 5).")
    p.add_argument("--radii", default=None,
                   help="Amber radius set. Default follows --igb (igb 2/5 -> mbondi2, "
                        "8 -> mbondi3, 7 -> bondi, 1 -> mbondi); overriding to a "
                        "non-matching set warns, because it changes the energies "
                        "without erroring.")
    p.add_argument("--saltcon", type=float, default=0.15,
                   help="Salt concentration in M for the GB model (default: 0.15).")
    p.add_argument("--no-decomp", dest="decomp", action="store_false", default=True,
                   help="Skip per-residue MMGBSA decomposition. Decomposition forces "
                        "gbsa=2 (ICOSA), which sander cannot atom-type for transition "
                        "metals, so a metalloprotein either drops the metal or uses "
                        "this flag to keep it.")
    p.add_argument("--python-exe", dest="python_exe", default=None,
                   help="Interpreter the generated SLURM jobs run (default: the one "
                        "running this command). Must be able to import autopath and "
                        "MDAnalysis; a bare 'python' on a compute node usually cannot.")
    p.add_argument("--mmgbsa-n-frames", type=int, default=20,
                   help="Frames to extract per MMGBSA job (default: 20).")
    p.add_argument("--mmgbsa-n-molecules", type=int, default=1,
                   help="Occupying molecules to emit a job for, best first (default: 1). "
                        "Each gets its own subdirectory under the target's mmgbsa/.")
    p.add_argument("--mmgbsa-frame-strategy", choices=["random", "cluster"],
                   default="random",
                   help="How to pick frames within a record (default: random). "
                        "'cluster' is reserved and not implemented yet.")
    p.add_argument("--mmgbsa-source", default=None,
                   help="Restrict MMGBSA frame selection to this simulation label.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for random frame selection (default: 0).")
    return p


def occupancy_rows(results, gap_tolerance=0):
    """Flatten ``{cosolvent: [Hotspot]}`` into one row per ProbeOccupancy record.

    *gap_tolerance* must be the value pose selection uses, or the episodes reported here
    disagree with the episode the chosen pose actually came from. ``topology`` and
    ``trajectory`` are carried on every row because a frame index is trajectory-local and
    therefore meaningless without the file it indexes.
    """
    rows = []
    for cosolvent, hotspots in results.items():
        for h in hotspots:
            for occ in h.probe_occupancy:
                episodes = occ.episodes(gap_tolerance)
                longest = occ.longest_episode(gap_tolerance)
                rows.append({
                    "site_id": h.site_id,
                    "cosolvent": cosolvent,
                    "source_label": occ.source_label,
                    "topology": occ.topology,
                    "trajectory": occ.trajectory,
                    "probe_resname": occ.probe_resname,
                    "probe_resid": occ.probe_resid,
                    "probe_resindex": occ.probe_resindex,
                    "n_frames_bound": occ.n_frames_bound,
                    "occupancy_fraction": round(occ.occupancy_fraction, 4),
                    "n_episodes": len(episodes),
                    "longest_start": longest[0] if longest else None,
                    "longest_end": longest[1] if longest else None,
                })
    return rows


def annotate(config, checkpoint_dir, args):
    """Load the checkpoint, run the occupancy scan, and persist the result."""
    cosolvents = []
    for sim in config.simulations:
        for c in sim.cosolvents:
            if c not in cosolvents:
                cosolvents.append(c)

    results = HotspotDetector.load_checkpoint(checkpoint_dir, cosolvents)
    annotator = OccupancyAnnotator(
        config.simulations,
        stride=args.stride,
        min_atoms_in_blob=args.min_atoms,
        logger=logger,
    )
    annotator.annotate(results)

    for sim in config.simulations:
        if not sim.chain_reference:
            continue
        import MDAnalysis as mda
        reference = mda.Universe(sim.chain_reference)
        topology = mda.Universe(sim.topology)
        for c in sim.cosolvents:
            if c in results:
                apply_chain_reference(results[c], reference, topology)

    HotspotDetector.save_checkpoint(results, checkpoint_dir)
    return results


def _cosolvent_list(config):
    """Every cosolvent named anywhere in the config, in first-seen order."""
    cosolvents = []
    for sim in config.simulations:
        for c in sim.cosolvents:
            if c not in cosolvents:
                cosolvents.append(c)
    return cosolvents


def _parse_target_tag(tag):
    """Invert :func:`cosolvkit.cli.refine_hotspots_jobs.target_tag`.

    :return: ``(is_binding_site, cosolvent_or_none, site_id)``. ``cosolvent`` is
        ``None`` for a binding-site tag, which carries no cosolvent of its own.
    :raises ValueError: if *tag* is not one of the two shapes that function writes.
    """
    if tag.startswith("bs_"):
        return True, None, int(tag[len("bs_"):])
    if tag.startswith("hs_"):
        rest = tag[len("hs_"):]
        cosolvent, site_id_str = rest.rsplit("_", 1)
        return False, cosolvent, int(site_id_str)
    raise ValueError(f"{tag!r} is not a refine_hotspots target directory name.")


def _find_target_hotspot(results, is_bs, cosolvent, site_id, probe_resname, probe_resid,
                         source_labels):
    """The :class:`Hotspot` in *results* that owns the molecule an mmgbsa/ job refined.

    Checkpoints only ever persist ``Hotspot`` objects — a binding site is a virtual
    grouping of them, recomputed at generation time, never itself checkpointed — so a
    ``bs_*`` job is resolved down to whichever member hotspot actually held this
    molecule. A hotspot's occupancy is always of its own cosolvent (see
    ``OccupancyAnnotator._scan``), so *probe_resname* alone narrows the search to
    ``results[probe_resname]``.
    """
    if not is_bs:
        for h in results.get(cosolvent, []):
            if h.site_id == site_id:
                return h
        return None

    candidates = []
    for h in results.get(probe_resname, []):
        occs = [o for o in h.probe_occupancy
                if o.probe_resname == probe_resname and o.probe_resid == probe_resid]
        if occs:
            candidates.append((h, occs))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    scored = sorted(
        candidates,
        key=lambda hc: (
            -len({o.source_label for o in hc[1]} & set(source_labels)),
            -sum(o.n_frames_bound for o in hc[1]),
        ),
    )
    logger.warning(
        "%s %d occupies %d hotspots of cosolvent %s; attaching to site_id %d (best "
        "source-label / frame-count match).",
        probe_resname, probe_resid, len(candidates), probe_resname,
        scored[0][0].site_id,
    )
    return scored[0][0]


def _representative_occupancy(hotspot, probe_resname, probe_resid):
    """The occupancy record ``_build_mmgbsa_inputs`` would have used as ``occ``.

    Same tie-break as ``select_mmgbsa_jobs``'s ``representative`` — reproducing it is
    what lets this reconstruct the correct source topology and strip-mask inputs for a
    job that may have pooled frames from several source records of the same molecule.
    """
    occs = [o for o in hotspot.probe_occupancy
            if o.probe_resname == probe_resname and o.probe_resid == probe_resid]
    if not occs:
        return None
    return max(occs, key=lambda o: (o.n_frames_bound, o.source_label))


def _attach_mmgbsa_result(hotspot, result):
    """Add *result* to the hotspot, replacing any earlier result from the same file.

    Keyed on ``results_path`` (one file per molecule per job) rather than appended
    unconditionally, so re-running ``--collect`` on the same jobs does not duplicate.
    """
    hotspot.mmgbsa = [r for r in hotspot.mmgbsa if r.results_path != result.results_path]
    hotspot.mmgbsa.append(result)


def _attach_decomposition(hotspot, df):
    """Write matching rows of *df* onto this hotspot's PocketResidues, by ORIGINAL resid.

    Assignment, not append: re-collecting overwrites a residue's previous
    decomposition rather than accumulating it.
    """
    if df is None or not hotspot.pocket_residues:
        return
    group_cols = [c for c in df.columns if c.endswith("_Avg")]
    by_resid = {int(row["resid"]): row for _, row in df.iterrows()}
    for pr in hotspot.pocket_residues:
        row = by_resid.get(int(pr.resid))
        if row is None:
            continue
        pr.mmgbsa_decomposition = {
            col[:-len("_Avg")]: {
                "average": float(row[col]),
                "std_dev": float(row[f"{col[:-len('_Avg')]}_StdDev"]),
                "std_err": float(row[f"{col[:-len('_Avg')]}_StdErr"]),
            }
            for col in group_cols
        }
        pr.mmgbsa_location = str(row["location"])


def collect_results(config, checkpoint_dir, out_dir):
    """Walk finished MMGBSA jobs under *out_dir* and attach them to the checkpoint.

    Reads every ``<out_dir>/<tag>/mmgbsa/<PROBE><resid>/`` directory ``generate_jobs``
    could have written, skipping (with a log line, not an error) any that have not
    finished MMPBSA yet. Writes ``mmgbsa_results.csv`` (one row per molecule refined)
    and ``mmgbsa_decomposition.csv`` (one row per residue per result, carrying both the
    stripped-complex index and the mapped ORIGINAL resid so the mapping stays
    auditable), and re-saves the hotspot checkpoint with the results attached.

    Idempotent: re-running replaces each hotspot's earlier result for the same
    ``results_path`` rather than appending a duplicate — the same principle
    :func:`annotate` uses for occupancy.

    :return: the updated ``{cosolvent: [Hotspot]}``.
    """
    from cosolvkit.analysis.sites.mmgbsa_results import (
        collect_decomposition, collect_job, parse_probe_dirname, parse_strip_mask,
        read_strip_mask,
    )

    results = HotspotDetector.load_checkpoint(checkpoint_dir, _cosolvent_list(config))

    import MDAnalysis as mda

    result_rows = []
    decomp_rows = []

    tag_dirs = sorted(d for d in glob.glob(os.path.join(out_dir, "*"))
                      if os.path.isdir(os.path.join(d, "mmgbsa")))

    for tag_dir in tag_dirs:
        tag = os.path.basename(tag_dir)
        try:
            is_bs, cosolvent, site_id = _parse_target_tag(tag)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", tag_dir, exc)
            continue

        for job_dir in sorted(glob.glob(os.path.join(tag_dir, "mmgbsa", "*"))):
            if not os.path.isdir(job_dir):
                continue
            try:
                probe_resname, probe_resid = parse_probe_dirname(
                    os.path.basename(job_dir))
            except ValueError as exc:
                logger.warning("Skipping %s: %s", job_dir, exc)
                continue

            source_labels = []
            frames_json = os.path.join(job_dir, "frames.json")
            if os.path.isfile(frames_json):
                with open(frames_json) as fh:
                    source_labels = [f["source_label"]
                                     for f in json.load(fh).get("frames", [])]

            hotspot = _find_target_hotspot(results, is_bs, cosolvent, site_id,
                                           probe_resname, probe_resid, source_labels)
            if hotspot is None:
                logger.warning(
                    "%s: no hotspot in the checkpoint occupies %s %d; skipping.",
                    job_dir, probe_resname, probe_resid,
                )
                continue

            representative = _representative_occupancy(hotspot, probe_resname,
                                                        probe_resid)
            if representative is None:
                logger.warning(
                    "%s: hotspot site_id=%s has no occupancy record for %s %d; "
                    "skipping.", job_dir, hotspot.site_id, probe_resname, probe_resid,
                )
                continue

            result = collect_job(job_dir, probe_resname, probe_resid,
                                 representative.source_label)
            if result is None:
                continue

            _attach_mmgbsa_result(hotspot, result)
            result_rows.append({
                "tag": tag,
                "target_type": "binding_site" if is_bs else "hotspot",
                "hotspot_site_id": hotspot.site_id,
                "cosolvent": hotspot.cosolvent,
                "probe_resname": result.probe_resname,
                "probe_resid": result.probe_resid,
                "source_label": result.source_label,
                "delta_total": result.delta_total,
                "std_dev": result.std_dev,
                "std_err": result.std_err,
                "n_frames": result.n_frames,
                "results_path": result.results_path,
            })

            try:
                strip_resnames, excluded_resids = parse_strip_mask(
                    read_strip_mask(job_dir))
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "%s: cannot read strip_mask (%s); skipping decomposition.",
                    job_dir, exc,
                )
                continue

            universe = mda.Universe(representative.topology)
            try:
                df = collect_decomposition(job_dir, universe, strip_resnames,
                                           excluded_resids)
            except ValueError:
                logger.exception("%s: decomposition mapping failed; skipping.", job_dir)
                continue
            if df is None:
                continue

            _attach_decomposition(hotspot, df)
            value_cols = [c for c in df.columns
                         if c.endswith(("_Avg", "_StdDev", "_StdErr"))]
            for _, row in df.iterrows():
                decomp_rows.append({
                    "tag": tag,
                    "target_type": "binding_site" if is_bs else "hotspot",
                    "hotspot_site_id": hotspot.site_id,
                    "cosolvent": hotspot.cosolvent,
                    "probe_resname": probe_resname,
                    "probe_resid": probe_resid,
                    "resname": row["resname"],
                    "stripped_resid": int(row["stripped_resid"]),
                    "resid": int(row["resid"]),
                    "location": row["location"],
                    **{c: row[c] for c in value_cols},
                })

    HotspotDetector.save_checkpoint(results, checkpoint_dir)

    results_csv = os.path.join(out_dir, "mmgbsa_results.csv")
    pd.DataFrame(result_rows).to_csv(results_csv, index=False)
    logger.info("Wrote %d MMGBSA result(s) to %s.", len(result_rows), results_csv)

    decomp_csv = os.path.join(out_dir, "mmgbsa_decomposition.csv")
    pd.DataFrame(decomp_rows).to_csv(decomp_csv, index=False)
    logger.info("Wrote %d decomposition row(s) to %s.", len(decomp_rows), decomp_csv)

    return results


def lie_rows(results):
    """One row per LieResult across every hotspot in *results*."""
    rows = []
    for cosolvent, hotspots in results.items():
        for h in hotspots:
            for r in h.lie:
                rows.append({
                    "site_id": h.site_id,
                    "cosolvent": cosolvent,
                    "probe_resname": r.probe_resname,
                    "probe_resid": r.probe_resid,
                    "source_label": r.source_label,
                    "eelec": round(r.eelec, 4),
                    "vdw": round(r.vdw, 4),
                    "total": round(r.total, 4),
                    "std_dev": round(r.std_dev, 4),
                    "std_err": round(r.std_err, 4),
                    "n_frames": r.n_frames,
                    "cutoff": r.cutoff,
                    "exclude_probes": r.exclude_probes,
                    "results_path": r.results_path,
                })
    return rows


def run_lie(config, results, checkpoint_dir, out_dir, args):
    """Compute LIE for the top targets INLINE and attach it to the checkpoint.

    Unlike the MMGBSA leg there is no qfile and no second stage: ``compute_LIE`` runs
    pytraj in-process in about a second per molecule, so queueing it would cost more
    than running it. Re-running replaces the records for a molecule rather than
    appending, matching how annotation behaves.
    """
    import MDAnalysis as mda

    from cosolvkit.analysis.sites.lie import compute_lie, probe_exclude_mask
    from cosolvkit.analysis.sites.mmgbsa import write_frame_trajectory
    from cosolvkit.cli.refine_hotspots_jobs import (
        _rank_targets, select_mmgbsa_jobs, target_tag,
    )

    targets = _rank_targets(config, results, args)
    n_done = 0
    for target, is_bs in targets:
        tag = target_tag(target, is_bs)
        jobs = select_mmgbsa_jobs(target, args, n_frames=args.lie_n_frames)
        if not jobs:
            logger.warning("Target %s has no occupancy record; skipping LIE.", tag)
            continue

        members = target.member_hotspots if is_bs else [target]
        for occ, selections in jobs:
            lie_dir = os.path.join(out_dir, tag, "lie",
                                   f"{occ.probe_resname}{occ.probe_resid}")
            os.makedirs(lie_dir, exist_ok=True)
            traj = os.path.join(lie_dir, "frames.dcd")
            write_frame_trajectory(selections, traj)

            exclude = None
            if args.lie_exclude_probes:
                universe = mda.Universe(occ.topology)
                species = [c for sim in config.simulations
                           if sim.label == occ.source_label for c in sim.cosolvents]
                exclude = probe_exclude_mask(universe, species or [occ.probe_resname],
                                             keep_resid=occ.probe_resid)

            result = compute_lie(occ, traj, lie_dir, cutoff=args.lie_cutoff,
                                 exclude_mask=exclude)
            if result is None:
                continue
            # Attach only to the member hotspot(s) that actually recorded this
            # molecule. A BindingSite's probe_occupancy is the concatenation over its
            # members, so writing to every member would duplicate one result across
            # blobs the molecule never entered and double-count any aggregate.
            key = (result.source_label, result.probe_resname, result.probe_resid)
            owners = [h for h in members
                      if any((o.source_label, o.probe_resname, o.probe_resid) == key
                             for o in h.probe_occupancy)]
            if not owners:
                logger.warning("No member hotspot of %s owns %s %d; attaching to all "
                               "%d members.", tag, result.probe_resname,
                               result.probe_resid, len(members))
                owners = members
            # Purge this molecule from EVERY hotspot before re-attaching, not just from
            # the owners. Otherwise a record written by an earlier pass — with a
            # different stride, or before the ownership rule narrowed — survives on a
            # hotspot the molecule never occupied, and the pass is not idempotent.
            for hs_list in results.values():
                for h in hs_list:
                    if h.lie:
                        h.lie = [r for r in h.lie
                                 if (r.source_label, r.probe_resname,
                                     r.probe_resid) != key]
            for h in owners:
                h.lie.append(result)
            n_done += 1
            logger.info("%s: LIE %s %d = %.2f +/- %.2f kcal/mol over %d frames "
                        "(EELEC %.2f, VDW %.2f).", tag, result.probe_resname,
                        result.probe_resid, result.total, result.std_err,
                        result.n_frames, result.eelec, result.vdw)

    HotspotDetector.save_checkpoint(results, checkpoint_dir)
    rows = lie_rows(results)
    csv_path = os.path.join(out_dir, "lie_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Computed %d LIE result(s); wrote %d row(s) to %s.",
                n_done, len(rows), csv_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    config = AnalysisConfig.from_yaml(args.config)
    # Resolve once and write it back, so job generation reads the merged maps from the
    # same directory the hotspots were loaded from.
    args.checkpoint = checkpoint_dir = (args.checkpoint
                                        or os.path.join(config.out_path, "merged"))
    # Absolute from here on. The generated SLURM script cds into the target directory
    # before running the driver, so a relative --out would resolve the driver path
    # against the new working directory and double it
    # (<out>/bs_1/<out>/bs_1/run_autopath.py). Every path baked into a generated file
    # has to survive that cd.
    out_dir = os.path.abspath(args.out or os.path.join(config.out_path, "refine"))
    os.makedirs(out_dir, exist_ok=True)

    if args.collect:
        collect_results(config, checkpoint_dir, out_dir)
        return 0

    results = annotate(config, checkpoint_dir, args)

    rows = occupancy_rows(results, gap_tolerance=args.gap_tolerance)
    csv_path = os.path.join(out_dir, "hotspot_occupancy.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote %d occupancy records to %s.", len(rows), csv_path)

    if args.annotate_only:
        return 0

    if args.mode == "lie":
        run_lie(config, results, checkpoint_dir, out_dir, args)
        return 0

    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs
    generate_jobs(config, results, out_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
