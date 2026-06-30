"""Production (vanilla) MD of an equilibrated cosolvent system with AutoPath.

This is the PRODUCTION phase of examples/02. It continues from the equilibration
phase (ap_equilibration.py), reading the equilibrated OpenMM system and the
checkpoint (positions + velocities + box) that equilibration wrote, and runs
unbiased conventional MD with ``autopath.VanillaMD``.

No restraints are applied (cosolvents and protein move freely). Run on a GPU
node in the ``autopath`` env (see run_production.q), e.g.:

    python ap_production.py --system-dir ../../scratch/6E22_benzene --replica rep1
"""

import os
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import VanillaMD
from autopath.utils import load_system, setup_logging, compute_rmsd
from openmm.app import PDBFile


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Run production (vanilla) MD of an equilibrated CosolvKit/AutoPath system.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument(
        "-d", "--system-dir", dest="system_dir", required=True,
        help="Input dir with the CosolvKit system.pdb (topology), e.g. "
             "../01_build_cosolvent_system/6E22_benzene. Read-only.",
    )
    parser.add_argument(
        "-n", "--name", dest="name", required=False, default=None,
        help="Run id used during equilibration. Default: basename of --system-dir.",
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", required=False, default=None,
        help="Output directory holding the equilibration/ results and where the MD/ "
             "outputs are written. Must match ap_equilibration.py's --out-dir. "
             "Default: ./<name> (inside this 02 example dir).",
    )
    parser.add_argument(
        "-r", "--replica", dest="replica", required=False, default=None,
        help="Replica label, to differentiate multiple production runs of the same system.",
    )
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    system_dir = os.path.abspath(args.system_dir)
    name = args.name if args.name is not None else os.path.basename(system_dir.rstrip("/"))
    run_id = name if args.replica is None else f"{name}_{args.replica}"
    # Equilibration outputs + MD outputs live in the 02 dir, not the read-only 01 input.
    out_dir = os.path.abspath(args.out_dir if args.out_dir is not None else name)

    logger = setup_logging(f"{out_dir}/autopath.log", log_level="INFO")

    ########################################################################################
    ######################################## Vanilla MD ####################################
    ########################################################################################
    MD_TIME = 100             # ns
    TIMESTEP = 0.004          # 4 fs
    TEMPERATURE = 300         # K
    SAVE_FREQ = 25000         # ~0.1 ns at 4 fs
    RESTART_VELOCITIES = False # resample velocities from the checkpoint
    RESTRAINED_ATOMS = None   # cosolvent MD: nothing restrained in production

    # Topology comes from the CosolvKit system.pdb in the 01 input dir
    # (OpenMM engine emits no prmtop).
    system_pdb_file = os.path.join(system_dir, "system.pdb")
    topology = PDBFile(system_pdb_file).topology

    # Handoff from the equilibration phase (written to out_dir by ap_equilibration.py).
    equil_system = os.path.join(out_dir, "equilibration", f"system_equil_{name}.xml")
    checkpoint = os.path.join(out_dir, "equilibration", f"checkpoint_equil_{name}.chk")
    for f in (system_pdb_file, equil_system, checkpoint):
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"Expected input {f} not found. Run ap_equilibration.py first "
                f"(with the same --name '{name}' and --out-dir)."
            )
    system = load_system(equil_system)

    vanilla_md = VanillaMD(
        system=system,
        topology=topology,
        restrained_atoms=RESTRAINED_ATOMS,
        timestep=TIMESTEP,
        temperature=TEMPERATURE,
        save_freq=SAVE_FREQ,
        out_dir=f"{out_dir}/MD",
    )
    vanilla_md.run(
        checkpoint_file=checkpoint,
        pdb_file=None,
        run_id=run_id,
        MD_time=MD_TIME,
        restart_velocities=RESTART_VELOCITIES,
    )

    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################
    traj_fname = f"{out_dir}/MD/MD_{run_id}.dcd"

    # Wrap, image and backbone-align the production trajectory (top = system.pdb).
    traj = md.load(traj_fname, top=system_pdb_file)
    traj = traj.center_coordinates()
    traj = traj.image_molecules()
    try:
        backbone = traj.topology.select("backbone")
        traj = traj.superpose(traj[0], atom_indices=backbone)
    except Exception as e:
        logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
    aligned = traj_fname.replace(".dcd", "_aligned.dcd")
    traj.save(aligned)
    os.remove(traj_fname)
    logger.info(f"Aligned production trajectory saved to {aligned}")

    # Protein RMSD (no ligand in a cosolvent box).
    u = mda.Universe(system_pdb_file, aligned, in_memory=True)
    rmsd_df = compute_rmsd(
        u, u,
        alig_select="backbone",
        groupselections={"protein": "protein and not name H*"},
        plots_outdir=f"{out_dir}/MD",
        suffix=run_id,
    )
    rmsd_df.to_csv(f"{out_dir}/MD/{run_id}_rmsd.csv", index=False)
    logger.info("Production MD complete.")
    return


if __name__ == "__main__":
    main()
