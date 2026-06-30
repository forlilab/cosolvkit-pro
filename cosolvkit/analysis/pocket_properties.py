# cosolvkit/analysis/pocket_properties.py
"""Back-compat shim. Scoring -> core/scoring.py, models -> core/models.py,
calculator/helpers -> sites/properties.py."""
from cosolvkit.analysis.core.models import PocketResidue  # noqa: F401
from cosolvkit.analysis.core.scoring import (  # noqa: F401
    compute_composite_score, _get_site_value, _CORE_ATTR_ALIASES,
)
from cosolvkit.analysis.sites.properties import (  # noqa: F401
    PocketPropertyCalculator, set_residue_embeddings,
    _serialize_regionprop_value, _is_xyz, _build_selection,
    _single_exp, _bi_exp, _r2, REGIONPROPS_ALL,
)
