#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# VMD session generation
#

import os
import logging
from typing import Union


def generate_vmd_session(out_path: str,
                          topology: str,
                          trajectory: str,
                          density_files: Union[str, list] = None):
    """Generate a VMD session script to visualize the trajectory and density.

    :param out_path: directory where the script is written.
    :type out_path: str
    :param topology: path to the topology file.
    :type topology: str
    :param trajectory: path to the trajectory file.
    :type trajectory: str
    :param density_files: list of .dx density files to load.
    :type density_files: Union[str, list]
    """
    logger = logging.getLogger(__name__)

    # FIXME at some point like for pymol
    isovalue = 1.0
    output_vmd_file = os.path.join(out_path, "vmd_session.vmd")

    topology_abs_path = os.path.abspath(topology)
    trajectory_abs_path = os.path.abspath(trajectory)

    vmd_script = f"""
# VMD visualization script

# Load topology and trajectory
mol new {topology_abs_path} type parm7
mol addfile {trajectory_abs_path} type netcdf waitfor all

# Set up protein visualization
mol delrep 0 top
mol representation NewCartoon
mol color Structure
mol selection "protein"
mol material Opaque
mol addrep top"""

    for i, density in enumerate(density_files or []):
        density_dx_abs_path = os.path.abspath(density)
        vmd_script += f"""

# Load density map
mol new {density_dx_abs_path} type dx waitfor all
mol representation Isosurface {isovalue} 0 0 0 1
mol color ColorID {i}
mol material Transparent
mol addrep top"""

    vmd_script += f"""

color Display Background white

save_state {output_vmd_file}
"""

    with open(output_vmd_file, "w") as f:
        f.write(vmd_script)

    logger.info(f"VMD session script saved as {output_vmd_file}")
