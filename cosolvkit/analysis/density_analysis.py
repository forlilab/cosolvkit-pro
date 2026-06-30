# cosolvkit/analysis/density_analysis.py
"""Back-compat shim. Code moved to core/grid.py and viz/{pymol,vmd}.py."""
from cosolvkit.analysis.core.grid import (  # noqa: F401
    BOLTZMANN_CONSTANT_KB,
    GridAnalysis, combine_dx_maps, combine_dx_maps_with_resampling,
    _read_dx, _grids_spatially_aligned, _grid_free_energy,
    _smooth_grid_free_energy, _grid_density, _subset_grid, _export,
)
from cosolvkit.analysis.viz.pymol import generate_pymol_session  # noqa: F401
from cosolvkit.analysis.viz.vmd import generate_vmd_session  # noqa: F401
