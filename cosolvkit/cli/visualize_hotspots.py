#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# CLI entry point for the interactive hotspot dashboard
#

import sys
import argparse


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Launch the CoSolvKit interactive hotspot dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize results from an analysis output directory (merged/ auto-detected):
  visualize_hotspots -d results/

  # Provide a reference PDB explicitly:
  visualize_hotspots -d results/ -p averaged_trajectory.pdb

  # Run on a custom port:
  visualize_hotspots -d results/ --port 8888

  # SSH / HPC usage — run on the HPC, then in a LOCAL terminal:
  #   ssh -L 8050:localhost:8050 user@hpc-hostname -N
  # The dashboard prints this command automatically when it detects SSH.

  # Set the AGFE cutoff label shown in the voxel view title:
  visualize_hotspots -d results/ --agfe-cutoff -1.5
""",
    )

    parser.add_argument(
        "-d", "--directory",
        dest="out_path",
        required=True,
        metavar="OUT_PATH",
        help=(
            "Analysis output directory containing hotspot_sites_*.csv files "
            "and .dx maps. A merged/ subdirectory is used automatically if present."
        ),
    )
    parser.add_argument(
        "-p", "--pdb",
        dest="pdb_path",
        default=None,
        metavar="PDB_FILE",
        help=(
            "Path to the reference PDB file. When omitted, the dashboard "
            "searches for averaged_trajectory.pdb in the output directory tree."
        ),
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=8050,
        metavar="PORT",
        help="Port for the Dash development server (default: 8050).",
    )
    parser.add_argument(
        "--agfe-cutoff",
        dest="agfe_cutoff",
        type=float,
        default=-1.0,
        metavar="CUTOFF",
        help=(
            "AGFE cutoff (kcal/mol) shown in voxel plot titles "
            "(default: -1.0). Must match the value used during analysis."
        ),
    )
    parser.add_argument(
        "--host",
        dest="host",
        default="0.0.0.0",
        metavar="HOST",
        help=(
            "Network interface to bind to (default: 0.0.0.0 — all interfaces). "
            "Required for SSH port forwarding; use 127.0.0.1 to restrict to localhost only."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run Dash in debug mode (hot-reloading, verbose error pages).",
    )

    return parser.parse_args()


def main():
    args = cmd_lineparser()

    try:
        from cosolvkit.hotspot_dashboard import HotspotDashboard
    except ImportError as exc:
        print(f"Import error: {exc}")
        print(
            "Install the required dashboard dependencies with:\n"
            "  pip install dash dash-bio"
        )
        sys.exit(1)

    dashboard = HotspotDashboard(
        out_path=args.out_path,
        pdb_path=args.pdb_path,
        port=args.port,
        agfe_cutoff=args.agfe_cutoff,
    )
    dashboard.run_server(host=args.host, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())
