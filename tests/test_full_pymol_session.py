"""A single session showing hotspots (per probe) and binding sites (per rank), with labels.

The pipeline already had two disconnected sessions: `visualise_clustering` per cosolvent, and
`generate_binding_site_session` whose sites sat at top level rather than under a parent group. This
builds one session with the two-level hierarchy the analysis actually has:

    hotspots
        hs_BEN, hs_PHN, ...          one subgroup per probe
            hs_BEN_r1_dens           carved AGFE density of that hotspot
            hs_BEN_r1_lab            pseudoatom carrying its properties as a label
    bindingSites
        bs_1, bs_2, ...              one subgroup per ranked site
            bs_1_pocket              the site's union mask
            bs_1_BEN, bs_1_PHN       the per-probe hotspot densities that were merged into it
            bs_1_lab                 site properties

Hotspots are shown UNFILTERED: the shape filter is a scoring decision, and a session whose purpose
is inspecting what the filter would remove must not have already removed it.

Every label object ends in ``_lab`` so the whole set toggles with ``disable *_lab`` /
``enable *_lab``; PyMol allows an object in only one group, so a separate "labels" group would
have to steal them from their probe/site groups.

These tests exercise the emitted .pml script, which is generated whether or not PyMol is
importable, so they run in environments without it.
"""

import numpy as np
import pytest


class _HS:
    def __init__(self, rank, centroid, props=None, n_voxels=40):
        self.rank = rank
        self.site_id = rank
        self.centroid = np.asarray(centroid, dtype=float)
        self.agfe_min = -1.5
        self.n_voxels = n_voxels
        self.properties = props or {}
        self.voxel_mask = np.ones((3, 3, 3), dtype=bool)
        self.grid_origin = np.zeros(3)
        self.grid_delta = np.full(3, 0.8)


class _BS:
    def __init__(self, rank, members, cosolvents, centroid=(0, 0, 0)):
        self.rank = rank
        self.site_id = rank
        self.centroid = np.asarray(centroid, dtype=float)
        self.member_hotspots = members
        self.cosolvents = list(cosolvents)
        self.n_cosolvents = len(cosolvents)
        self.volume = 120.0
        self.combined = 4.2
        self.probe_coverage = 0.5
        self.residence = None
        self.voxel_mask = np.ones((3, 3, 3), dtype=bool)
        self.grid_origin = np.zeros(3)
        self.grid_delta = np.full(3, 0.8)


@pytest.fixture
def scene():
    ben = {"BEN": [_HS(1, (1, 1, 1), {"geom_solidity": 0.81, "field_sharpness": 0.42}),
                   _HS(2, (5, 5, 5), {"geom_solidity": 0.95})]}
    phn = {"PHN": [_HS(1, (1.2, 1.1, 0.9), {"geom_solidity": 0.77})]}
    probe_results = {**ben, **phn}
    sites = [_BS(1, [ben["BEN"][0], phn["PHN"][0]], ["BEN", "PHN"]),
             _BS(2, [ben["BEN"][1]], ["BEN"], centroid=(5, 5, 5))]
    return probe_results, sites


def _pml(tmp_path, scene, **kw):
    from cosolvkit.analysis.viz.pymol import write_full_session_script
    probe_results, sites = scene
    p = tmp_path / "full_session.pml"
    write_full_session_script(probe_results, sites, str(p),
                              density_dir=str(tmp_path), reference_pdb=None, **kw)
    return p.read_text()


def test_two_top_level_groups_exist(tmp_path, scene):
    s = _pml(tmp_path, scene)
    assert "group hotspots," in s
    assert "group bindingSites," in s


def test_one_hotspot_subgroup_per_probe(tmp_path, scene):
    s = _pml(tmp_path, scene)
    assert "group hs_BEN," in s
    assert "group hs_PHN," in s
    # and those subgroups are nested under the parent
    assert "group hotspots, hs_BEN hs_PHN" in s or "group hotspots, hs_PHN hs_BEN" in s


def test_one_binding_site_subgroup_per_rank_nested_under_the_parent(tmp_path, scene):
    s = _pml(tmp_path, scene)
    assert "group bs_1," in s and "group bs_2," in s
    assert "group bindingSites, bs_1 bs_2" in s


def test_each_site_carries_its_member_probe_densities(tmp_path, scene):
    """Site 1 merged a BEN and a PHN hotspot, so both must appear inside it."""
    s = _pml(tmp_path, scene)
    assert "bs_1_BEN" in s and "bs_1_PHN" in s
    assert "bs_1_pocket" in s
    # site 2 had only BEN
    assert "bs_2_BEN" in s and "bs_2_PHN" not in s


def test_hotspots_are_not_filtered(tmp_path, scene):
    """Both BEN hotspots appear, including the solidity 0.95 one a filter would drop."""
    s = _pml(tmp_path, scene)
    assert "hs_BEN_r1" in s and "hs_BEN_r2" in s


def test_labels_are_toggleable_as_a_set(tmp_path, scene):
    s = _pml(tmp_path, scene)
    assert "hs_BEN_r1_lab" in s
    assert "bs_1_lab" in s
    assert "disable *_lab" in s     # documented off switch
    assert "enable *_lab" in s


def test_labels_carry_the_properties(tmp_path, scene):
    s = _pml(tmp_path, scene)
    assert "solidity" in s and "0.81" in s          # hotspot property
    assert "sharpness" in s and "0.42" in s
    assert "rank" in s


def test_missing_properties_do_not_break_the_label(tmp_path, scene):
    """The second BEN hotspot has no field_sharpness; it must still get a label."""
    s = _pml(tmp_path, scene)
    assert "hs_BEN_r2_lab" in s


def test_labels_start_hidden_when_asked(tmp_path, scene):
    s = _pml(tmp_path, scene, labels_on=False)
    assert "disable *_lab" in s.split("# --- final state")[-1]


def test_paths_are_absolute_so_the_script_replays_from_anywhere(tmp_path, scene):
    """A .pml with relative paths silently loads nothing unless PyMol's cwd happens to match
    the directory it was generated in. That is also the version-independent way to view the
    session: a .pse written by PyMol 3.x may not restore mesh objects in an older PyMol."""
    import os
    s = _pml(tmp_path, scene)
    for line in s.splitlines():
        if line.startswith("load "):
            path = line.split(" ", 1)[1].split(",")[0].strip()
            assert os.path.isabs(path), f"relative path in .pml: {path}"
