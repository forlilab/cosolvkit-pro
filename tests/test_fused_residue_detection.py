"""A topology whose resid field wrapped must be detected, because resid selections go wrong.

The PDB resid field is 4 characters, so it maxes out at 9999. In a real FosAKP box with 11,924
waters, loading ``system.pdb`` gives 9,999 correct 3-atom water residues **plus one
5,775-atom pseudo-residue** holding 1,925 fused waters; the matching ``system.prmtop`` gives all
11,924 cleanly.

What this does and does not break, measured on that box:

* Position histograms do not care, so **AGFE maps were unaffected**.
* Survival probability runs with ``residues=False``, and waterdynamics then keys on
  ``atoms.ids``, which are unique (40,444 of 40,444). So **SP is not corrupted either**.
* What does break: **316 resids map to more than one residue**, so any ``resid``-based
  selection silently picks up atoms from unrelated molecules. ``_build_selection`` builds
  ``sphzone R resid N`` for resid-style zones, which would then be centred on the wrong atoms.
  Per-residue quantities (a probe's radius of gyration, ``residues=True`` analyses) are wrong
  for the fused residue too.

So the fused pseudo-residue is the *detectable symptom* of a wrapped topology. What is tested
here is the detection itself and the warn/no-warn decision; the exact wording of the message is
not pinned.
"""

import logging

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.sites.properties import (
    _zone_is_resid_based,
    detect_fused_residues,
    warn_if_fused_residues,
)

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


def _waters(atom_resindex, n_residues, resname="HOH"):
    """Universe of 3-atom waters grouped according to atom_resindex."""
    n_atoms = len(atom_resindex)
    u = mda.Universe.empty(
        n_atoms, n_residues=n_residues, n_segments=1,
        atom_resindex=atom_resindex, residue_segindex=[0] * n_residues,
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["O", "H1", "H2"] * (n_atoms // 3))
    u.add_TopologyAttr("resname", [resname] * n_residues)
    u.add_TopologyAttr("resid", list(range(1, n_residues + 1)))
    pos = np.tile(np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3), (2, 1, 1))
    u.load_new(pos, order="fac", format=MemoryReader,
               dimensions=np.array([[50.0] * 3 + [90.0] * 3] * 2))
    return u


def _clean(n_waters=5):
    return _waters(sum(([i] * 3 for i in range(n_waters)), []), n_waters)


def _fused(n_clean=4, n_in_fused=3):
    """n_clean normal waters plus one residue holding n_in_fused waters."""
    idx = sum(([i] * 3 for i in range(n_clean)), []) + [n_clean] * (3 * n_in_fused)
    return _waters(idx, n_clean + 1)


def test_zone_is_resid_based_discriminates_resids_from_xyz():
    """Only resid-style zones are exposed to the ambiguity, so the warning is gated on them.

    The load-bearing case is the 3-element zone: all-int means resids, floats mean a point.
    """
    assert _zone_is_resid_based(7) is True
    assert _zone_is_resid_based([12, 13, 14]) is True
    assert _zone_is_resid_based([1, 2, 3]) is True
    assert _zone_is_resid_based([1.5, 2.5, 3.5]) is False
    assert _zone_is_resid_based((0.0, 0.0, 0.0)) is False


def test_fused_residue_is_detected_with_its_size():
    u = _fused(n_clean=4, n_in_fused=3)
    fused = detect_fused_residues(u.select_atoms("resname HOH"))
    assert len(fused) == 1
    resid, n_atoms = fused[0]
    assert n_atoms == 9, "3 waters fused into one residue"
    assert resid == 5


def test_clean_and_degenerate_selections_are_not_flagged():
    """No false positives: a clean box, an empty selection (a probe absent from the box), and
    a single-residue selection, where the modal size IS that residue and it must not flag itself.
    """
    assert detect_fused_residues(_clean().select_atoms("resname HOH")) == []
    assert detect_fused_residues(_clean().select_atoms("resname NOPE")) == []
    assert detect_fused_residues(_clean(n_waters=1).select_atoms("resname HOH")) == []


def test_uniformly_large_residues_are_not_flagged():
    """A 12-atom probe is normal; only residues exceeding the modal size are suspect."""
    n = 4
    idx = sum(([i] * 12 for i in range(n)), [])
    u = mda.Universe.empty(12 * n, n_residues=n, n_segments=1, atom_resindex=idx,
                           residue_segindex=[0] * n, trajectory=True)
    u.add_TopologyAttr("resname", ["BEN"] * n)
    u.add_TopologyAttr("resid", list(range(1, n + 1)))
    assert detect_fused_residues(u.select_atoms("resname BEN")) == []


def test_warn_flags_a_fused_topology_and_stays_silent_on_a_clean_one(caplog):
    """Callers branch on the return value, and a warning on a clean box would send someone
    re-running analyses that were fine."""
    with caplog.at_level(logging.WARNING):
        flagged = warn_if_fused_residues(_fused().select_atoms("resname HOH"),
                                         logger=logging.getLogger("t"),
                                         context="zone selection")
    assert flagged is True
    assert caplog.text != ""

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        clean_flagged = warn_if_fused_residues(_clean().select_atoms("resname HOH"),
                                               logger=logging.getLogger("t"))
    assert clean_flagged is False
    assert caplog.text == ""
