#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Linear Interaction Energy for a bound probe, via autopath's ProteinLigandAnalyzer.
#

import logging
import os

import numpy as np

from cosolvkit.analysis.core.models import LieResult

logger = logging.getLogger(__name__)

# autopath's own default: monatomic counter-ions only. Water is deliberately NOT here —
# LIE scores the ligand against its whole surroundings, and the solvent term is what
# makes the alpha/beta form approximate binding rather than a bare contact energy.
DEFAULT_ION_EXCLUDE = ":Na+,Cl-,NA,CL,K,K+"


def _analyzer(topology, trajectory, ligand_resname, outdir):
    """Build a ProteinLigandAnalyzer. Imported lazily so cosolvkit needn't need autopath."""
    from autopath.ap_PLIP import ProteinLigandAnalyzer

    return ProteinLigandAnalyzer(
        top=topology,
        trajs=[trajectory],
        ligand_mda_selection=f"resname {ligand_resname}",
        outdir=outdir,
    )


def probe_exclude_mask(universe, probe_resnames, keep_resid, base=DEFAULT_ION_EXCLUDE):
    """Extend the ion exclude mask with every probe copy except *keep_resid*.

    Whether this changes anything depends on how the box was BUILT. cosolvkit can apply
    a pairwise repulsive force between cosolvent molecules
    (``CosolventSystem.add_repulsive_forces``), and the 6E22 example does exactly that
    for BEN-BEN at sigma 4.0 A. On such a system no second probe sits inside the cutoff
    and this mask is a measured no-op — on both 6E22 and FosAKP the environment was
    byte-identical with and without it. Without that build-time repulsion, probes can
    aggregate and their probe-probe terms would otherwise enter the LIE.
    """
    resids = set()
    for resname in probe_resnames:
        resids.update(int(r) for r in
                      universe.select_atoms(f"resname {resname}").resids)
    others = sorted(r for r in resids if r != int(keep_resid))
    if not others:
        return base
    return base + "," + ",".join(str(r) for r in others)


def compute_lie(occupancy, trajectory, outdir, cutoff=6.0, exclude_mask=None,
                lie_options="nopbc cutvdw 10.0 cutelec 10.0", analyzer_factory=None):
    """Run LIE for one probe molecule over one purpose-built frame trajectory.

    :param occupancy: a :class:`ProbeOccupancy`; supplies the prmtop, the probe resname
        and the probe resid used as the Amber ligand mask.
    :param trajectory: DCD holding exactly the frames to score (see
        :func:`cosolvkit.analysis.sites.mmgbsa.write_frame_trajectory`).
    :param outdir: directory for ``LIE_results.csv`` and the component plot.
    :param cutoff: Angstrom; residues within this of the ligand form the environment.
    :param exclude_mask: Amber mask removed from the environment. Defaults to ions only,
        which keeps water in.
    :param analyzer_factory: test seam; ``(topology, trajectory, resname, outdir)``.
    :return: a :class:`LieResult`, or None when the run produced no frames.
    """
    os.makedirs(outdir, exist_ok=True)
    factory = analyzer_factory or _analyzer
    analyzer = factory(occupancy.topology, trajectory, occupancy.probe_resname, outdir)

    df = analyzer.compute_LIE(
        prmtop=occupancy.topology,
        use_residues=None,                       # auto-select the cutoff shell
        ligand_amber_selection=occupancy.amber_mask,
        exclude_amber_selection=exclude_mask or DEFAULT_ION_EXCLUDE,
        cutoff=cutoff,
    )
    if df is None or len(df) == 0:
        logger.warning("LIE produced no frames for %s %d in %s; skipping.",
                       occupancy.probe_resname, occupancy.probe_resid, trajectory)
        return None

    total = np.asarray(df["Total"], dtype=float)
    n = len(total)
    std = float(total.std(ddof=1)) if n > 1 else 0.0
    return LieResult(
        probe_resname=occupancy.probe_resname,
        probe_resid=int(occupancy.probe_resid),
        source_label=occupancy.source_label,
        eelec=float(np.asarray(df["EELEC"], dtype=float).mean()),
        vdw=float(np.asarray(df["VDW"], dtype=float).mean()),
        total=float(total.mean()),
        std_dev=std,
        std_err=std / np.sqrt(n) if n > 1 else 0.0,
        n_frames=n,
        cutoff=float(cutoff),
        exclude_probes=bool(exclude_mask),
        results_path=os.path.join(os.path.abspath(outdir), "LIE_results.csv"),
    )
