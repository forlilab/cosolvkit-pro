# cosolvkit/analysis/hotspots_detection.py
"""Back-compat shim. Hotspot -> core/models.py; HotspotDetector -> sites/detect.py."""
from cosolvkit.analysis.core.models import Hotspot  # noqa: F401
from cosolvkit.analysis.sites.detect import (  # noqa: F401
    HotspotDetector, save_checkpoint, load_checkpoint,
)
