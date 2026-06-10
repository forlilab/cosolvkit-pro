import os
import sys
import argparse
from cosolvkit.simulation import run_simulation
from cosolvkit.analysis.utils import setup_logging

#TODO When Autopath is published, this class should be a wrapper around Autopath classses API.
# It should coordinate the dispatch of multiple simulation replicas and optionally monitor convergence
# using the analysis class.
def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs an OpenMM MD simulation from a cosolvkit-prepared system.",
                                     epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2023 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""")

    parser.add_argument('--pdb', dest='pdb', required=True,
                        help='path to the system PDB file (system.pdb)')
    parser.add_argument('--system', dest='system', required=True,
                        help='path to the serialized OpenMM system XML file (system.xml)')
    parser.add_argument('--output_dir', dest='output_dir', default='results',
                        help='directory where trajectory and statistics will be saved (default: results)')
    parser.add_argument('--membrane', action='store_true', default=False,
                        help='use membrane barostat for membrane-protein systems')
    parser.add_argument('--num_simulation_steps', type=int, default=25000000,
                        help='total number of MD simulation steps (default: 25000000 = 100 ns at 4 fs)')
    parser.add_argument('--traj_write_freq', type=int, default=25000,
                        help='frequency of writing trajectory frames (default: 25000)')
    parser.add_argument('--time_step', type=float, default=0.004,
                        help='MD timestep in ps (default: 0.004)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='simulation temperature in K (default: 300.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for reproducibility (default: None)')

    return parser.parse_args()


def main():
    args = cmd_lineparser()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(level="INFO", filepath=os.path.join(args.output_dir, "simulation.log"))

    logger.info("Starting MD simulation")
    run_simulation(
        pdb_fname=args.pdb,
        system_fname=args.system,
        membrane_protein=args.membrane,
        traj_write_freq=args.traj_write_freq,
        time_step=args.time_step,
        temperature=args.temperature,
        simulation_steps=args.num_simulation_steps,
        results_path=args.output_dir,
        seed=args.seed,
    )
    logger.info("Simulation finished.")
    return


if __name__ == "__main__":
    sys.exit(main())
