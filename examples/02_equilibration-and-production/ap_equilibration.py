"""Equilibrate a CosolvKit-built cosolvent system with AutoPath.

This is the EQUILIBRATION phase of examples/02. It deliberately SKIPS AutoPath's
own system-building step: the system was already assembled and parametrized by
CosolvKit (examples/01_build_cosolvent_system), which writes ``system.pdb`` and
``system.xml`` (OpenMM engine) into the output directory. We load those two
files directly and hand them to ``autopath.Equilibration``.

Restraints are protein-only (see cosolvent_equilibration.json): the cosolvents
and water are left free so they equilibrate around the restrained protein.

Run on a GPU node in the ``autopath`` env (see run_equilibration.q), e.g.:

    python ap_equilibration.py --system-dir ../../scratch/6E22_benzene
"""

import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import Equilibration
from autopath.utils import load_system, setup_logging, compute_rmsd
from openmm.app import PDBFile


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Equilibrate a CosolvKit-built cosolvent system with AutoPath "
                    "(no build step; consumes system.pdb + system.xml).",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument(
        "-d", "--system-dir", dest="system_dir", required=True,
        help="Input directory with the CosolvKit-generated system.pdb and system.xml "
             "(e.g. ../01_build_cosolvent_system/6E22_benzene). Read-only.",
    )
    parser.add_argument(
        "-n", "--name", dest="name", required=False, default=None,
        help="Run id (used in output filenames). Default: basename of --system-dir.",
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", required=False, default=None,
        help="Output directory for equilibration results. Default: ./<name> (i.e. inside "
             "this 02 example dir), so outputs do NOT pollute the 01 build directory.",
    )
    parser.add_argument(
        "-p", "--protocol", dest="protocol", required=False,
        default="cosolvent_equilibration.json",
        help="Path to the JSON equilibration protocol (default: cosolvent_equilibration.json).",
    )
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    system_dir = os.path.abspath(args.system_dir)
    run_id = args.name if args.name is not None else os.path.basename(system_dir.rstrip("/"))
    # Outputs go to the 02 dir (default ./<name>), separate from the read-only 01 input.
    out_dir = os.path.abspath(args.out_dir if args.out_dir is not None else run_id)
    os.makedirs(out_dir, exist_ok=True)
    equilibration_scheme = args.protocol

    logger = setup_logging(f"{out_dir}/autopath.log", log_level="INFO")
    logger.info(f"Starting equilibration for '{run_id}': input {system_dir} -> output {out_dir}")

    ########################################################################################
    #################################### Load CosolvKit system #############################
    ########################################################################################
    # CosolvKit already built and parametrized the system; we just load its output.
    # (This replaces AutoPath's PDBPreprocessor + SystemPreparation build step.)
    system_pdb_file = os.path.join(system_dir, "system.pdb")
    system_xml_file = os.path.join(system_dir, "system.xml")
    for f in (system_pdb_file, system_xml_file):
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"Expected CosolvKit output {f} not found. "
                f"Run examples/01_build_cosolvent_system first (OpenMM engine)."
            )

    topology = PDBFile(system_pdb_file).topology
    system = load_system(system_xml_file)

    ########################################################################################
    ###################################### Equilibration ###################################
    ########################################################################################
    equilibration = Equilibration(
        system=system,
        topology=topology,
        protocol_fname=equilibration_scheme,
        is_membrane=False,
        restrained_minimization=False,
        out_dir=f"{out_dir}/equilibration",
    )
    # Writes system_equil_{run_id}.xml + checkpoint_equil_{run_id}.chk (the handoff
    # to the production phase) plus equilibration_{run_id}.dcd and CSV/plots.
    equilibration.run(pdb_file=system_pdb_file, run_id=run_id)

    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################
    equilibrated_traj = f"{out_dir}/equilibration/equilibration_{run_id}.dcd"

    # Wrap, image and backbone-align the equilibration trajectory.
    # Topology is system.pdb (the OpenMM engine emits no prmtop).
    traj = md.load(equilibrated_traj, top=system_pdb_file)
    traj = traj.center_coordinates()
    traj = traj.image_molecules()
    try:
        backbone = traj.topology.select("backbone")
        traj = traj.superpose(traj[0], atom_indices=backbone)
    except Exception as e:
        logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
    aligned = equilibrated_traj.replace(".dcd", "_aligned.dcd")
    traj.save(aligned)
    os.remove(equilibrated_traj)
    logger.info(f"Aligned equilibration trajectory saved to {aligned}")

    # Protein RMSD over the equilibration (no ligand: a cosolvent box has none).
    u_eq = mda.Universe(system_pdb_file, aligned, in_memory=True)
    rmsd_eq = compute_rmsd(
        u_eq, u_eq,
        alig_select="backbone",
        groupselections={"protein": "protein and not name H*"},
        plots_outdir=f"{out_dir}/equilibration",
    )
    rmsd_eq.to_csv(f"{out_dir}/equilibration/RMSD_{run_id}.csv", index=False)
    logger.info("Equilibration complete.")
    return


if __name__ == "__main__":
    main()
