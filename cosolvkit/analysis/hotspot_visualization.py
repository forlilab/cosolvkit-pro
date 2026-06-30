# cosolvkit/analysis/hotspot_visualization.py
"""Back-compat shim. Code moved to viz/plotly.py and viz/pymol.py."""
from cosolvkit.analysis.viz.plotly import (  # noqa: F401
    plot_hotspot_clustering_3d, plot_sp_raw, plot_sp_fits,
)
from cosolvkit.analysis.viz.pymol import (  # noqa: F401
    visualise_clustering, add_hotspots_to_pymol_session,
    generate_consensus_pockets_session, generate_pharmacophore_session,
)
