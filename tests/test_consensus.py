"""Tests for CrossProbeConsensusDetector and ConsensusSite.

All test inputs are synthetic BindingSite objects built via the make_binding_site
fixture — no trajectory or .dx files required.
"""

import json
import os

import numpy as np
import pytest

from cosolvkit.consensus_detection import CrossProbeConsensusDetector


# ---------------------------------------------------------------------------
# Jaccard helpers
# ---------------------------------------------------------------------------

class TestComputeJaccard:

    def _detector(self, probe_results):
        return CrossProbeConsensusDetector(probe_results, jaccard_threshold=0.05)

    def test_perfect_overlap_is_one(self, make_binding_site):
        s = make_binding_site()
        d = self._detector({"BEN": [s]})
        assert d._compute_jaccard(s, s) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self, make_binding_site):
        s1 = make_binding_site(blob_slices=(slice(0, 5), slice(0, 5), slice(0, 5)))
        s2 = make_binding_site(blob_slices=(slice(15, 20), slice(15, 20), slice(15, 20)))
        d = self._detector({"BEN": [s1], "ACE": [s2]})
        assert d._compute_jaccard(s1, s2) == pytest.approx(0.0)

    def test_half_overlap(self, make_binding_site):
        mask_a = np.zeros((10, 1, 1), dtype=bool)
        mask_a[:10] = True
        mask_b = np.zeros((10, 1, 1), dtype=bool)
        mask_b[5:] = True
        s1 = make_binding_site(shape=(10, 1, 1), blob_slices=np.s_[:10])
        s2 = make_binding_site(shape=(10, 1, 1), blob_slices=np.s_[5:])
        s1.voxel_mask = mask_a
        s2.voxel_mask = mask_b
        d = self._detector({"BEN": [s1], "ACE": [s2]})
        # intersection = 5, union = 10 → Jaccard = 0.5
        assert d._compute_jaccard(s1, s2) == pytest.approx(0.5, abs=0.01)

    def test_different_shapes_resampling_path(self, make_binding_site):
        """Sites with different grid shapes should use the resampling path."""
        s1 = make_binding_site(shape=(20, 20, 20), blob_slices=np.s_[5:15, 5:15, 5:15])
        s2 = make_binding_site(shape=(10, 10, 10), blob_slices=np.s_[2:8, 2:8, 2:8])
        # Set grid metadata so map_coordinates path is used
        s1.grid_origin = np.array([0.0, 0.0, 0.0])
        s1.grid_delta = np.array([0.5, 0.5, 0.5])
        s2.grid_origin = np.array([0.0, 0.0, 0.0])
        s2.grid_delta = np.array([1.0, 1.0, 1.0])
        d = self._detector({"BEN": [s1], "ACE": [s2]})
        jaccard = d._compute_jaccard(s1, s2)
        assert jaccard >= 0.0
        assert jaccard <= 1.0


# ---------------------------------------------------------------------------
# build_overlap_graph
# ---------------------------------------------------------------------------

class TestBuildOverlapGraph:

    def test_intra_probe_sites_never_connected(self, make_binding_site):
        s1 = make_binding_site(cosolvent="BEN", rank=1)
        s2 = make_binding_site(cosolvent="BEN", rank=2,
                                blob_slices=np.s_[5:10, 5:10, 5:10])
        d = CrossProbeConsensusDetector({"BEN": [s1, s2]}, jaccard_threshold=0.0)
        g = d.build_overlap_graph()
        assert g.number_of_edges() == 0

    def test_overlapping_cross_probe_sites_are_connected(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")  # same blob position → overlap = 1.0
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.5,
        )
        g = d.build_overlap_graph()
        assert g.number_of_edges() == 1

    def test_non_overlapping_sites_not_connected(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN",
                                   blob_slices=np.s_[0:5, 0:5, 0:5])
        s_ace = make_binding_site(cosolvent="ACE",
                                   blob_slices=np.s_[15:20, 15:20, 15:20])
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
        )
        g = d.build_overlap_graph()
        assert g.number_of_edges() == 0

    def test_jaccard_below_threshold_gives_no_edge(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=1.1,  # impossible threshold
        )
        g = d.build_overlap_graph()
        assert g.number_of_edges() == 0

    def test_edge_weight_is_jaccard_value(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
        )
        g = d.build_overlap_graph()
        edges = list(g.edges(data=True))
        assert len(edges) == 1
        assert "weight" in edges[0][2]
        assert edges[0][2]["weight"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# detect_communities
# ---------------------------------------------------------------------------

class TestDetectCommunities:

    def test_overlapping_probes_form_one_community(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")  # same location
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
        )
        sites = d.detect_communities()
        assert len(sites) == 1
        assert sites[0].n_probes == 2

    def test_non_overlapping_probes_form_separate_communities(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN", blob_slices=np.s_[0:5, 0:5, 0:5])
        s_ace = make_binding_site(cosolvent="ACE", blob_slices=np.s_[15:20, 15:20, 15:20])
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
        )
        sites = d.detect_communities()
        assert len(sites) == 2

    def test_three_probes_one_overlapping_pair(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")  # overlaps BEN
        s_nma = make_binding_site(cosolvent="NMA", blob_slices=np.s_[14:19, 14:19, 14:19])
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace], "NMA": [s_nma]},
            jaccard_threshold=0.05,
        )
        sites = d.detect_communities()
        # BEN+ACE form 1 community; NMA isolated → 2 communities total
        assert len(sites) == 2

    def test_probe_coverage_is_correct(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
            score_weights={"coverage": 1.0, "favorability": 0.0, "volume": 0.0},
        )
        sites = d.detect_communities()
        top = sites[0]
        assert top.probe_coverage == pytest.approx(top.n_probes / top.total_probes)

    def test_consensus_score_is_finite_and_nonneg(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector({"BEN": [s_ben], "ACE": [s_ace]})
        sites = d.detect_communities()
        for s in sites:
            assert np.isfinite(s.consensus_score)
            assert s.consensus_score >= 0.0

    def test_top_site_rank_one(self, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        sites = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]}
        ).detect_communities()
        assert sites[0].consensus_rank == 1

    def test_consensus_centroid_between_member_centroids(self, make_binding_site):
        c1 = np.array([2.0, 2.0, 2.0])
        c2 = np.array([8.0, 8.0, 8.0])
        s_ben = make_binding_site(cosolvent="BEN", centroid=c1,
                                   blob_slices=np.s_[2:5, 2:5, 2:5])
        s_ace = make_binding_site(cosolvent="ACE", centroid=c2,
                                   blob_slices=np.s_[2:5, 2:5, 2:5])  # same mask → Jaccard=1
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            jaccard_threshold=0.05,
        )
        sites = d.detect_communities()
        cc = sites[0].consensus_centroid
        # weighted centroid must lie between the two member centroids
        for i in range(3):
            assert min(c1[i], c2[i]) <= cc[i] <= max(c1[i], c2[i])

    def test_single_probe_no_inter_probe_edges(self, make_binding_site):
        """Single probe → all sites are isolated nodes → each forms its own community."""
        s1 = make_binding_site(cosolvent="BEN", rank=1, blob_slices=np.s_[0:5, 0:5, 0:5])
        s2 = make_binding_site(cosolvent="BEN", rank=2, blob_slices=np.s_[14:19, 14:19, 14:19])
        d = CrossProbeConsensusDetector({"BEN": [s1, s2]}, jaccard_threshold=0.05)
        sites = d.detect_communities()
        # Pin current behavior: 2 isolated nodes → 2 singleton communities
        assert len(sites) == 2

    def test_greedy_modularity_method(self, make_binding_site):
        pytest.importorskip("networkx")
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector(
            {"BEN": [s_ben], "ACE": [s_ace]},
            community_method="greedy_modularity",
        )
        sites = d.detect_communities()
        assert len(sites) >= 1


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

class TestExportResults:

    def _run_and_export(self, tmp_path, make_binding_site):
        s_ben = make_binding_site(cosolvent="BEN")
        s_ace = make_binding_site(cosolvent="ACE")
        d = CrossProbeConsensusDetector({"BEN": [s_ben], "ACE": [s_ace]})
        sites = d.detect_communities()
        d.export_results(sites, out_path=str(tmp_path))
        return sites

    def test_csv_written(self, tmp_path, make_binding_site):
        self._run_and_export(tmp_path, make_binding_site)
        assert (tmp_path / "consensus_sites.csv").exists()

    def test_json_written(self, tmp_path, make_binding_site):
        self._run_and_export(tmp_path, make_binding_site)
        assert (tmp_path / "consensus_sites.json").exists()

    def test_pharmacophore_json_written(self, tmp_path, make_binding_site):
        self._run_and_export(tmp_path, make_binding_site)
        assert (tmp_path / "consensus_sites_pharmacophore.json").exists()

    def test_csv_row_count_matches_sites(self, tmp_path, make_binding_site):
        import pandas as pd
        sites = self._run_and_export(tmp_path, make_binding_site)
        df = pd.read_csv(tmp_path / "consensus_sites.csv")
        assert len(df) == len(sites)

    def test_json_is_valid(self, tmp_path, make_binding_site):
        self._run_and_export(tmp_path, make_binding_site)
        with open(tmp_path / "consensus_sites.json") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_pharmacophore_has_expected_structure(self, tmp_path, make_binding_site):
        self._run_and_export(tmp_path, make_binding_site)
        with open(tmp_path / "consensus_sites_pharmacophore.json") as f:
            data = json.load(f)
        for entry in data:
            assert "consensus_rank" in entry
            assert "pharmacophore" in entry
            assert "member_cosolvents" in entry

    def test_empty_sites_no_crash(self, tmp_path, make_binding_site):
        d = CrossProbeConsensusDetector({"BEN": [], "ACE": []})
        d.export_results([], out_path=str(tmp_path))
        # Should not raise; CSV is empty but present
        assert (tmp_path / "consensus_sites.csv").exists()
