import sys
import os
import argparse
from cosolvkit.analysis.utils import setup_logging


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs cosolvkit analysis on the output of a MD simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YAML-driven (recommended):
  analyze_cosolvent_simulation -cfg analysis.yaml

  # Generate a fully-commented YAML template:
  analyze_cosolvent_simulation --generate-config analysis.yaml

  # Legacy single-trajectory mode (backward-compatible):
  analyze_cosolvent_simulation -tr traj.xtc -tp system.prmtop -c BEN,ACE -o results/
""",
    )

    # ---- YAML mode (primary) ----------------------------------------
    parser.add_argument(
        '-cfg', '--config',
        dest='config',
        default=None,
        metavar='CONFIG_YAML',
        help='Path to YAML analysis config file (primary interface).',
    )
    parser.add_argument(
        '--generate-config',
        dest='generate_config',
        default=None,
        metavar='OUTPUT_PATH',
        help='Write a fully-commented YAML template to OUTPUT_PATH and exit.',
    )

    # ---- Legacy single-trajectory flags (backward compatibility) -----
    legacy = parser.add_argument_group(
        'Legacy single-trajectory flags',
        'Kept for backward compatibility. When -cfg is given these are ignored.',
    )
    legacy.add_argument('-tr', '--trajectory',     dest='traj_file',       default=None,
                        help='Path to the trajectory file from MD simulation.')
    legacy.add_argument('-tp', '--topology',       dest='top_file',        default=None,
                        help='Path to the topology file from MD simulation.')
    legacy.add_argument('-o',  '--out_path',       dest='out_path',        default=None,
                        help='Path where to store output files.')
    legacy.add_argument('-c',  '--cosolvents',     dest='cosolvents',      default=None,
                        help='Cosolvent resnames to analyse, separated by commas.')
    legacy.add_argument('-at', '--atomtypes',      dest='use_atomtypes',   default=False,
                        action='store_true',
                        help='Enable atom-type-based density analysis.')
    legacy.add_argument('-atfname', '--atomtypes_file', dest='atomtypes_file', default=None,
                        help='Path to a custom atom-types JSON file.')
    legacy.add_argument('-l',  '--statistics_file', dest='statistics_file', default=None,
                        help='Path to the MD log/statistics file.')
    legacy.add_argument('--consensus', dest='consensus', default=False,
                        action='store_true',
                        help='Run cross-probe consensus detection after hotspot detection.')
    legacy.add_argument('--jaccard-threshold', dest='jaccard_threshold', default=0.05,
                        type=float,
                        help='Minimum Jaccard voxel-mask overlap to link two sites '
                             'in the consensus graph (default 0.05).')
    legacy.add_argument('--load-hotspot-checkpoint', dest='load_hotspot_checkpoint',
                        default=False, action='store_true',
                        help='Skip hotspot detection and load results from the '
                             'previously saved checkpoint under '
                             'out_path/hotspot_checkpoints/. '
                             'Useful to re-run consensus with different parameters.')

    return parser.parse_args()


def _run_from_yaml(config_path: str, logger):
    from cosolvkit.analysis.analysis_config import AnalysisConfig
    from cosolvkit.analysis.multi_report import MultiReport

    cfg = AnalysisConfig.from_yaml(config_path)
    runner = MultiReport(cfg)
    runner.run()


def _run_legacy(args, logger):
    """Single-trajectory mode using the original flag interface."""
    from cosolvkit.analysis import Report

    if not all([args.traj_file, args.top_file, args.out_path, args.cosolvents]):
        logger.error(
            "Legacy mode requires -tr, -tp, -o, and -c. "
            "Use -cfg for YAML-driven analysis."
        )
        sys.exit(1)

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    report = Report(
        statistics_file=args.statistics_file,
        traj_file=args.traj_file,
        top_file=args.top_file,
        cosolvent_names=args.cosolvents.split(','),
        out_path=out_path,
    )

    report.generate_report(equilibration=False, rmsf=True, rdf=False)

    report.generate_density_maps(
        use_atomtypes=args.use_atomtypes,
        atomtypes_definitions=args.atomtypes_file,
        temperature=300,
    )

    report.generate_pymol_session(
        reference_pdb=os.path.join(out_path, 'averaged_trajectory.pdb')
    )

    if args.load_hotspot_checkpoint:
        hotspot_results = report.load_hotspot_checkpoint()
    else:
        hotspot_results = report.generate_hotspot_report(
            min_cluster_voxels=20,
            agfe_cutoff=-1.0,
            top_n_plot=10,
        )

    if args.consensus:
        report.generate_consensus_report(
            probe_results=hotspot_results,
            jaccard_threshold=args.jaccard_threshold,
        )


def main():
    args = cmd_lineparser()

    # --generate-config: write template and exit
    if args.generate_config is not None:
        from cosolvkit.analysis.analysis_config import AnalysisConfig
        AnalysisConfig.generate_template(args.generate_config)
        print(f"Template written to: {args.generate_config}")
        sys.exit(0)

    # Determine out_path for early logger setup
    if args.config:
        import yaml
        with open(args.config) as fh:
            _raw = yaml.safe_load(fh) or {}
        out_path = _raw.get('out_path', 'results')
        # Resolve relative to config file location (same logic as AnalysisConfig)
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), out_path)
    else:
        out_path = args.out_path or 'results'

    os.makedirs(out_path, exist_ok=True)
    logger = setup_logging(
        level="INFO",
        filepath=os.path.join(out_path, "cosolvkit_analysis.log"),
    )

    if args.config:
        _run_from_yaml(args.config, logger)
    else:
        logger.warning(
            "No -cfg/--config YAML file provided; running in legacy single-trajectory mode. "
            "To migrate: analyze_cosolvent_simulation --generate-config analysis.yaml"
        )
        _run_legacy(args, logger)


if __name__ == "__main__":
    sys.exit(main())
