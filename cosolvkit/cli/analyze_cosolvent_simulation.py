import sys
import argparse
from cosolvkit.analysis import Report
from cosolvkit.utils import setup_logging

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit analysis on the output of a MD simulation")
    parser.add_argument('-tr','--trajectory', dest='traj_file', required=True,
                        action='store', help='path to the trajectory file from MD simulation')
    parser.add_argument('-tp','--topology', dest='top_file', required=True,
                        action='store', help='path to the topology file from MD simulation')
    parser.add_argument('-o', '--out_path', dest='out_path', required=True,
                        action='store', help='path where to store output files')
    parser.add_argument('-c', '--cosolvents', dest='cosolvents', required=True,
                        action='store', help='list of cosolvents resname to analyze separated by commas')
    parser.add_argument('-at', '--atomtypes', dest='atomtypes_file', required=False,
                        action='store_true', help='turn on the use of atom types for density map generation (default: False)')
    parser.add_argument('-atfname', '--atomtypes_file', dest='atomtypes_file', required=False,
                        action='store', help='path to the file containing the definition of atom types for density map generation')
    parser.add_argument('-l', '--statistics_file', dest='statistics_file', required=False,
                    action='store', help='path to the log file from MD simulation')
    return parser.parse_args()

def main():
    args = cmd_lineparser()

    logger = setup_logging(level="INFO", filepath=f"{args.out_path}/cosolvkit_analysis.log")

    report = Report(args.statistics_file, 
                    args.traj_file, 
                    args.top_file, 
                    args.cosolvents.split(','), 
                    args.out_path)
    
    report.generate_report(equilibration=False, rmsf=False, rdf=False)
    
    report.generate_density_maps(
                                use_atomtypes=args.atomtypes_file,
                                atomtypes_definitions=args.atomtypes_file,
                                temperature=300)
    
    report.generate_pymol_session(reference_pdb=f'{args.out_path}/averaged_trajectory.pdb')
    
    report.generate_hotspot_report(
        min_cluster_voxels=2, 
        agfe_cutoff=-1.0,
        top_n_plot=10
    )

    return

if __name__ == "__main__":
    sys.exit(main())