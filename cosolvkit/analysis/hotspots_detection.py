# cosolvkit/analysis/hotspots_detection.py
"""Back-compat shim. BindingSite -> core/models.py; HotspotDetector -> sites/detect.py."""
from cosolvkit.analysis.core.models import BindingSite  # noqa: F401
from cosolvkit.analysis.sites.detect import (  # noqa: F401
    HotspotDetector, save_checkpoint, load_checkpoint,
)
