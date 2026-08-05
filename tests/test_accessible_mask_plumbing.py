"""The accessible-volume mask must be written where the detector looks for it.

It was not. `GridAnalysis._build_accessible_mask` wrote to
``getattr(self, "_out_dir", None) or os.getcwd()`` and **nothing in the package ever set
``_out_dir``** -- `report.py` constructs `GridAnalysis(...)` with no output path. So every probe of
every replica wrote the same fixed filename into the current working directory, each overwriting
the last, while `detect.py` looked in `self.out_path` and never found one. Measured on the FosAKP
tree: **zero** masks under `analysis_v3`, and a single stray
``benchmarking/solvent_accessible_map.dx`` in the launch directory.

Consequence: `accessible_fraction` was never populated by the pipeline, despite carrying a
non-zero default weight -- the feature contributed nothing and only tripped the dead-weight guard.
The 0.724 AUC that justified the weight came from benchmarking scripts that set ``_out_dir``
themselves.

Two fixes, tested here:

1. ``GridAnalysis(out_dir=...)`` and the mask named per probe, so replicas (different out_dir) and
   probes (different filename) stop colliding.
2. Replica masks combined by MAJORITY VOTE -- a voxel is accessible if it was seen in more than
   half the replicas. Union would make the accessible volume grow with replica count, so the
   feature would not be comparable between runs with different numbers of replicas; majority is
   count-stable and drops single-visit noise, consistent with the rest of the pipeline.
"""

import os

import numpy as np
import pytest
from gridData import Grid


def _mask_grid(values, origin=(0.0, 0.0, 0.0), delta=0.8):
    return Grid(np.asarray(values, dtype=float),
                origin=np.asarray(origin, dtype=float),
                delta=np.full(3, float(delta)))


class TestMajorityCombine:

    def test_majority_needs_more_than_half(self, tmp_path):
        """3 of 5 accessible -> True; 2 of 5 -> False."""
        from cosolvkit.analysis.core.grid import combine_accessible_masks
        paths = []
        # voxel [0,0,0]: seen by 3/5. voxel [1,1,1]: seen by 2/5.
        seen_a = [True, True, True, False, False]
        seen_b = [True, True, False, False, False]
        for i in range(5):
            m = np.zeros((3, 3, 3))
            m[0, 0, 0] = float(seen_a[i])
            m[1, 1, 1] = float(seen_b[i])
            p = tmp_path / f"m{i}.dx"
            _mask_grid(m).export(str(p))
            paths.append(str(p))
        out = combine_accessible_masks(paths, out_fname=str(tmp_path / "merged.dx"))
        assert out[0, 0, 0], "3/5 should be accessible"
        assert not out[1, 1, 1], "2/5 should not be accessible"

    def test_is_stable_against_replica_count(self, tmp_path):
        """The same fraction of sightings gives the same answer at 3 and at 9 replicas.

        This is the property union lacks.
        """
        from cosolvkit.analysis.core.grid import combine_accessible_masks
        def build(n, n_seen, tag):
            paths = []
            for i in range(n):
                m = np.zeros((2, 2, 2))
                m[0, 0, 0] = float(i < n_seen)
                p = tmp_path / f"{tag}{i}.dx"
                _mask_grid(m).export(str(p))
                paths.append(str(p))
            return combine_accessible_masks(
                paths, out_fname=str(tmp_path / f"{tag}.dx"))
        assert build(3, 1, "a")[0, 0, 0] == build(9, 3, "b")[0, 0, 0]   # 1/3 both ways
        assert build(3, 2, "c")[0, 0, 0] == build(9, 6, "d")[0, 0, 0]   # 2/3 both ways

    def test_handles_differing_extents_on_the_shared_lattice(self, tmp_path):
        """Replica grids differ in shape; masks must still combine without interpolation."""
        from cosolvkit.analysis.core.grid import combine_accessible_masks
        paths = []
        for i, (shape, origin) in enumerate([((4, 4, 4), (0.0, 0.0, 0.0)),
                                             ((5, 5, 5), (-0.8, -0.8, -0.8)),
                                             ((4, 4, 4), (0.0, 0.0, 0.0))]):
            m = np.ones(shape)
            p = tmp_path / f"e{i}.dx"
            _mask_grid(m, origin=origin).export(str(p))
            paths.append(str(p))
        out = combine_accessible_masks(paths, out_fname=str(tmp_path / "e.dx"))
        assert out.dtype == bool and out.any()

    def test_single_mask_passes_through(self, tmp_path):
        from cosolvkit.analysis.core.grid import combine_accessible_masks
        m = np.zeros((3, 3, 3)); m[1, 1, 1] = 1.0
        p = tmp_path / "one.dx"
        _mask_grid(m).export(str(p))
        out = combine_accessible_masks([str(p)], out_fname=str(tmp_path / "o.dx"))
        assert out[1, 1, 1] and out.sum() == 1


class TestGridAnalysisWritesWhereAsked:

    def test_out_dir_is_honoured_and_named_per_probe(self, tmp_path):
        """The regression: no out_dir meant the mask went to the cwd under one shared name."""
        pytest.importorskip("MDAnalysis")
        from tests.test_grid_extent import _universe
        from cosolvkit.analysis.core.grid import GridAnalysis
        u = _universe(probe_offset=0.0)
        an = GridAnalysis(u.select_atoms("resname BEN"), gridsize=1.0,
                          use_atomtypes=False, verbose=False, out_dir=str(tmp_path))
        an.run()
        written = sorted(os.path.basename(p) for p in tmp_path.glob("solvent_accessible_map*.dx"))
        assert written == ["solvent_accessible_map_BEN.dx"], \
            f"mask not written per probe into out_dir; found {written}"

    def test_out_dir_is_exposed_on_the_instance(self, tmp_path):
        pytest.importorskip("MDAnalysis")
        from tests.test_grid_extent import _universe
        from cosolvkit.analysis.core.grid import GridAnalysis
        an = GridAnalysis(_universe().select_atoms("resname BEN"), gridsize=1.0,
                          use_atomtypes=False, verbose=False, out_dir=str(tmp_path))
        assert an._out_dir == str(tmp_path)
