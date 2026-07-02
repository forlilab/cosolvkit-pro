"""Back-compat shim. Moved to config.py."""
from cosolvkit.analysis.config import *  # noqa: F401,F403
from cosolvkit.analysis.config import (  # noqa: F401
    AnalysisConfig,
    ClusteringConfig,
    BindingSitesConfig,
    DensityMapsConfig,
    HotspotsConfig,
    MiscConfig,
    PyMolConfig,
    ReportConfig,
    SimulationEntry,
    CheckpointConfig,
)
