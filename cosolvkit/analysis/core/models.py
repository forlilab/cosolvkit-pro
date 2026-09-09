#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Data model classes for analysis.
# Dependency leaf: only third-party imports (numpy, dataclasses, typing).
# Must NOT import from cosolvkit.analysis.sites, core.grid, or core.scoring.
#

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _round_or_none(value, ndigits=4):
    """Round *value* for CSV/JSON export; ``None``/non-finite pass through as ``None``.

    Sites not derived from an AGFE density map legitimately have no affinity/shape values.
    """
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return round(fval, ndigits) if np.isfinite(fval) else None


@dataclass
class ProbeOccupancy:
    """Frames in which one probe molecule occupied one hotspot, in one source file.

    Identity is ``(source_label, probe_resname, probe_resid)``. A hotspot visited by
    several copies of a probe holds one record per copy.
    """

    source_label:     str
    topology:         str
    trajectory:       str
    probe_resname:    str
    probe_resid:      int
    probe_resindex:   int
    frames:           List[int]
    n_frames_scanned: int
    stride:           int = 1

    @property
    def n_frames_bound(self) -> int:
        return len(self.frames)

    @property
    def occupancy_fraction(self) -> float:
        return (self.n_frames_bound / self.n_frames_scanned
                if self.n_frames_scanned else 0.0)

    @property
    def amber_mask(self) -> str:
        """Amber residue mask selecting this molecule, e.g. ``":279"``."""
        return f":{self.probe_resid}"

    def episodes(self, gap_tolerance: int = 0) -> List[Tuple[int, int]]:
        """Inclusive frame bounds of contiguous residence.

        Two sampled frames are contiguous when they differ by at most
        ``stride * (1 + gap_tolerance)``. Derived from ``frames`` on every call — never
        stored, because storing it would duplicate state.
        """
        if not self.frames:
            return []
        max_gap = self.stride * (1 + gap_tolerance)
        ordered = sorted(self.frames)
        out = []
        start = prev = ordered[0]
        for f in ordered[1:]:
            if f - prev > max_gap:
                out.append((start, prev))
                start = f
            prev = f
        out.append((start, prev))
        return out

    def longest_episode(self, gap_tolerance: int = 0) -> Optional[Tuple[int, int]]:
        """Widest episode; ties resolve to the earliest."""
        eps = self.episodes(gap_tolerance)
        if not eps:
            return None
        return max(eps, key=lambda e: (e[1] - e[0], -e[0]))

    def to_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "topology": self.topology,
            "trajectory": self.trajectory,
            "probe_resname": self.probe_resname,
            "probe_resid": int(self.probe_resid),
            "probe_resindex": int(self.probe_resindex),
            "frames": [int(f) for f in self.frames],
            "n_frames_scanned": int(self.n_frames_scanned),
            "stride": int(self.stride),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeOccupancy":
        return cls(
            source_label=str(d["source_label"]),
            topology=str(d["topology"]),
            trajectory=str(d["trajectory"]),
            probe_resname=str(d["probe_resname"]),
            probe_resid=int(d["probe_resid"]),
            probe_resindex=int(d["probe_resindex"]),
            frames=[int(f) for f in d["frames"]],
            n_frames_scanned=int(d["n_frames_scanned"]),
            stride=int(d.get("stride", 1)),
        )


@dataclass
class MmgbsaResult:
    """An MMGBSA interaction energy for one probe molecule in one hotspot.

    A single-trajectory interaction energy over the frames that molecule actually
    occupied the site — useful for RANKING hotspots, not as an absolute affinity.
    ``components`` holds the raw Differences table keyed by MMPBSA component name
    (``VDWAALS``, ``EEL``, ``EGB``, ``ESURF``, ``DELTA G gas``, ``DELTA G solv``).
    """

    probe_resname:  str
    probe_resid:    int       # resid in the ORIGINAL topology, not the stripped complex
    source_label:   str
    delta_total:    float     # kcal/mol
    std_dev:        float
    std_err:        float
    n_frames:       int
    components:     Dict[str, Dict[str, float]] = field(default_factory=dict)
    results_path:   Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "probe_resname": self.probe_resname,
            "probe_resid": int(self.probe_resid),
            "source_label": self.source_label,
            "delta_total": float(self.delta_total),
            "std_dev": float(self.std_dev),
            "std_err": float(self.std_err),
            "n_frames": int(self.n_frames),
            "components": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.components.items()
            },
            "results_path": self.results_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MmgbsaResult":
        return cls(
            probe_resname=str(d["probe_resname"]),
            probe_resid=int(d["probe_resid"]),
            source_label=str(d["source_label"]),
            delta_total=float(d["delta_total"]),
            std_dev=float(d["std_dev"]),
            std_err=float(d["std_err"]),
            n_frames=int(d["n_frames"]),
            components={
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in (d.get("components") or {}).items()
            },
            results_path=d.get("results_path"),
        )


@dataclass
class LieResult:
    """A Linear Interaction Energy for one probe molecule in one hotspot.

    NOT a binding free energy. These are the raw ligand-environment interaction terms
    that ``ProteinLigandAnalyzer.compute_LIE`` returns, in kcal/mol, with no alpha/beta
    coefficients applied and no explicit desolvation term. The field is ``total``, not
    ``delta_total``, to keep that distinct from :class:`MmgbsaResult`.

    The environment is the residues within ``cutoff`` of the ligand INCLUDING WATER —
    solvent is what makes the alpha/beta form approximate binding rather than a bare
    contact energy — so these numbers are not comparable to an MMGBSA total, which
    scores the whole receptor and subtracts a solvation leg.
    """

    probe_resname:  str
    probe_resid:    int       # resid in the ORIGINAL prmtop numbering
    source_label:   str
    eelec:          float     # kcal/mol, mean over frames
    vdw:            float     # kcal/mol, mean over frames
    total:          float     # eelec + vdw
    std_dev:        float
    std_err:        float
    n_frames:       int
    cutoff:         float = 6.0
    n_env_residues: int = 0
    exclude_probes: bool = False
    results_path:   Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "probe_resname": self.probe_resname,
            "probe_resid": int(self.probe_resid),
            "source_label": self.source_label,
            "eelec": float(self.eelec),
            "vdw": float(self.vdw),
            "total": float(self.total),
            "std_dev": float(self.std_dev),
            "std_err": float(self.std_err),
            "n_frames": int(self.n_frames),
            "cutoff": float(self.cutoff),
            "n_env_residues": int(self.n_env_residues),
            "exclude_probes": bool(self.exclude_probes),
            "results_path": self.results_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LieResult":
        return cls(
            probe_resname=str(d["probe_resname"]),
            probe_resid=int(d["probe_resid"]),
            source_label=str(d["source_label"]),
            eelec=float(d["eelec"]),
            vdw=float(d["vdw"]),
            total=float(d["total"]),
            std_dev=float(d["std_dev"]),
            std_err=float(d["std_err"]),
            n_frames=int(d["n_frames"]),
            cutoff=float(d.get("cutoff", 6.0)),
            n_env_residues=int(d.get("n_env_residues", 0)),
            exclude_probes=bool(d.get("exclude_probes", False)),
            results_path=d.get("results_path"),
        )


@dataclass
class PoseRef:
    """A single frame identifying one probe molecule bound in one hotspot."""

    source_label:   str
    topology:       str
    trajectory:     str
    frame:          int
    probe_resname:  str
    probe_resid:    int
    probe_resindex: int
    episode:        Tuple[int, int]

    @property
    def amber_mask(self) -> str:
        return f":{self.probe_resid}"


# ---------------------------------------------------------------------------
# PocketResidue — per-residue data attached to a Hotspot
# ---------------------------------------------------------------------------

@dataclass
class PocketResidue:
    """A protein residue that lines a cosolvent hotspot cavity.

    Populated incrementally by :class:`PocketPropertyCalculator` (identity/proximity,
    then ``rmsf``, ``cosolvent_contacts``) and by externally injected embeddings.

    Attributes
    ----------
    resid : int
        PDB residue number (MDAnalysis ``resid``).
    resindex : int
        Universe-internal index (stable across trajectory frames).
    resname : str
        Three-letter residue code.
    chain : str
        Segment / chain ID.
    n_contact_voxels : int
        Distinct blob voxels within the cutoff of any heavy atom.
    min_dist_ang : float
        Distance in Å to the nearest blob voxel.
    contact_fraction : float
        ``n_contact_voxels / total_blob_voxels``.
    rmsf : float or None
        Cα RMSF in Å.
    embedding : np.ndarray or None
        PLM feature vector, shape ``(n_dims,)``.
    embedding_model : str or None
        Name of the model that produced ``embedding``.
    cosolvent_contacts : dict
        ``{source_label: {cosolvent_name: {cosolvent_resid: [frame_index, ...]}}}``.
        Source-keyed because a frame index is ambiguous across replicas.
    chain_source : str
        Where ``chain`` came from — ``"topology"`` (segid) or ``"chain_reference"``.
    properties : dict
        Extensible bag for arbitrary extra scalar properties.
    mmgbsa_decomposition : dict
        Per-residue MMGBSA energy decomposition, keyed by energy group name (e.g.
        ``"van_der_Waals"``, ``"Electrostatic"``) with ``{"average", "std_dev",
        "std_err"}`` values in kcal/mol. Populated by
        :mod:`cosolvkit.analysis.sites.mmgbsa_results`. ``resid`` here is this
        residue's ORIGINAL topology numbering — the stripped-complex index used by
        the decomposition table has already been mapped back before assignment.
    mmgbsa_location : str or None
        ``'R'`` (receptor) or ``'L'`` (ligand) side of the decomposition table this
        residue was reported on.
    """

    resid:            int
    resindex:         int
    resname:          str
    chain:            str
    n_contact_voxels: int
    min_dist_ang:     float
    contact_fraction: float

    rmsf:            Optional[float]       = None
    embedding:       Optional[np.ndarray]  = None
    embedding_model: Optional[str]         = None

    cosolvent_contacts: Dict[str, Dict[str, Dict[int, List[int]]]] = field(
        default_factory=dict
    )
    chain_source: str = "topology"     # "topology" | "chain_reference"
    properties: Dict[str, Any] = field(default_factory=dict)
    mmgbsa_decomposition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mmgbsa_location: Optional[str] = None

    def _iter_sources(self, source=None):
        """Yield the ``{resname: {resid: frames}}`` maps for one source, or all of them."""
        if source is None:
            return list(self.cosolvent_contacts.values())
        entry = self.cosolvent_contacts.get(source)
        return [entry] if entry is not None else []

    def contact_frames(self, cosolvent_name: str, source=None) -> List[int]:
        """Sorted frames where any molecule of *cosolvent_name* contacted this residue."""
        all_frames = set()
        for by_probe in self._iter_sources(source):
            for frames in by_probe.get(cosolvent_name, {}).values():
                all_frames.update(frames)
        return sorted(all_frames)

    def contact_molecules(self, cosolvent_name: str, source=None) -> List[int]:
        """Sorted resids of the molecules that ever contacted this residue."""
        resids = set()
        for by_probe in self._iter_sources(source):
            resids.update(by_probe.get(cosolvent_name, {}).keys())
        return sorted(resids)

    def n_contact_events(self, cosolvent_name: str, source=None) -> int:
        """Total (molecule, frame) pairs — a proxy for raw contact frequency."""
        return sum(
            len(frames)
            for by_probe in self._iter_sources(source)
            for frames in by_probe.get(cosolvent_name, {}).values()
        )

    def to_dict(self) -> dict:
        """JSON-safe dict: ``embedding`` becomes a list and contact resids are stringified."""
        return {
            "resid": self.resid,
            "resindex": self.resindex,
            "resname": self.resname,
            "chain": self.chain,
            "n_contact_voxels": self.n_contact_voxels,
            "min_dist_ang": round(float(self.min_dist_ang), 4),
            "contact_fraction": round(float(self.contact_fraction), 4),
            "rmsf": round(float(self.rmsf), 4) if self.rmsf is not None else None,
            "embedding": (
                [float(v) for v in self.embedding] if self.embedding is not None
                else None
            ),
            "embedding_model": self.embedding_model,
            "chain_source": self.chain_source,
            "cosolvent_contacts": {
                source: {
                    cosolvent: {str(rid): frames for rid, frames in mol_dict.items()}
                    for cosolvent, mol_dict in by_probe.items()
                }
                for source, by_probe in self.cosolvent_contacts.items()
            },
            "properties": self.properties,
            "mmgbsa_decomposition": {
                group: {kk: float(vv) for kk, vv in vals.items()}
                for group, vals in self.mmgbsa_decomposition.items()
            },
            "mmgbsa_location": self.mmgbsa_location,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PocketResidue":
        """Reconstruct from :meth:`to_dict` output."""
        pr = cls(
            resid=int(d["resid"]),
            resindex=int(d["resindex"]),
            resname=str(d["resname"]),
            chain=str(d["chain"]),
            n_contact_voxels=int(d["n_contact_voxels"]),
            min_dist_ang=float(d["min_dist_ang"]),
            contact_fraction=float(d["contact_fraction"]),
        )
        pr.rmsf = float(d["rmsf"]) if d.get("rmsf") is not None else None
        emb = d.get("embedding")
        pr.embedding = (
            np.array(emb, dtype=np.float32) if emb is not None else None
        )
        pr.embedding_model = d.get("embedding_model")
        pr.chain_source = str(d.get("chain_source", "topology"))
        raw = d.get("cosolvent_contacts", {})
        pr.cosolvent_contacts = {
            source: {
                cosolvent: {int(rid): list(frames) for rid, frames in mol_dict.items()}
                for cosolvent, mol_dict in by_probe.items()
            }
            for source, by_probe in raw.items()
        }
        pr.properties = dict(d.get("properties", {}))
        pr.mmgbsa_decomposition = {
            group: {kk: float(vv) for kk, vv in vals.items()}
            for group, vals in (d.get("mmgbsa_decomposition") or {}).items()
        }
        pr.mmgbsa_location = d.get("mmgbsa_location")
        return pr

    def __repr__(self) -> str:
        return (
            f"PocketResidue({self.resname}{self.resid}, chain={self.chain!r}, "
            f"min_dist={self.min_dist_ang:.2f}Å, rmsf={self.rmsf})"
        )


# ---------------------------------------------------------------------------
# Hotspot — a detected cosolvent binding hotspot
# ---------------------------------------------------------------------------

class Hotspot:
    """A binding hotspot detected from cosolvent AGFE density maps.

    Normally built by :class:`HotspotDetector`. Downstream analyses attach extra data
    through the ``properties`` dict rather than by subclassing. Every AGFE-derived
    argument is optional, so the class also serves as a plain occupancy-region record
    for sites that did not come from a density map; ``cosolvent``, ``centroid`` and
    ``voxel_mask`` alone are enough for hotspot grouping.
    """

    def __init__(self, rank, site_id, cosolvent, n_voxels=0, centroid=None,
                 agfe_min=None, agfe_mean_top_pct=None, voxel_mask=None,
                 favorable_atomtypes=None, per_type_agfe=None):
        self.rank = rank                            # 1 = most-negative agfe_min
        self.site_id = site_id                      # label from scipy.ndimage.label
        self.cosolvent = cosolvent                  # residue name
        self.n_voxels = n_voxels
        self.centroid = centroid                    # (3,), Angstroms
        self.agfe_min = agfe_min                    # kcal/mol, or None
        self.agfe_mean_top_pct = agfe_mean_top_pct  # kcal/mol, or None
        self.voxel_mask = voxel_mask                # boolean 3D, same shape as the AGFE grid
        self.favorable_atomtypes = (list(favorable_atomtypes)
                                    if favorable_atomtypes else [])
        self.per_type_agfe = dict(per_type_agfe) if per_type_agfe else {}  # min AGFE per type
        self.properties = {}
        self.pocket_residues = []                     # populated on demand
        self.probe_occupancy = []                     # List[ProbeOccupancy]
        self.mmgbsa = []                               # List[MmgbsaResult]
        self.lie = []                                  # List[LieResult]
        # Grid metadata, set by HotspotDetector.detect() after construction; required to
        # combine voxel masks across grids (e.g. binding-site grouping).
        self.grid_origin = None                       # np.ndarray (3,), Angstroms
        self.grid_delta = None                        # np.ndarray (3,), Angstroms per voxel

    def add_property(self, name, value):
        """Attach an arbitrary named property."""
        self.properties[name] = value

    @property
    def n_probe_molecules(self):
        return len(self.probe_occupancy)

    @property
    def total_residence_frames(self):
        return sum(o.n_frames_bound for o in self.probe_occupancy)

    def occupancy_by_probe(self):
        """``{probe_resname: [ProbeOccupancy, ...]}``."""
        out = {}
        for occ in self.probe_occupancy:
            out.setdefault(occ.probe_resname, []).append(occ)
        return out

    def best_pose(self, gap_tolerance=0):
        """The most convincing bound frame, or None when nothing ever occupied this site.

        Selects the record with the widest residence episode; ties resolve on higher
        occupancy fraction, then source label, then resid, so the result is deterministic.
        The frame returned is the *sampled* frame nearest the episode midpoint — with a
        stride the arithmetic midpoint need not be a frame that was scanned.
        """
        best = None
        best_key = None
        for occ in self.probe_occupancy:
            ep = occ.longest_episode(gap_tolerance)
            if ep is None:
                continue
            key = (-(ep[1] - ep[0]), -occ.occupancy_fraction,
                   occ.source_label, occ.probe_resid)
            if best_key is None or key < best_key:
                best_key, best = key, (occ, ep)
        if best is None:
            return None
        occ, ep = best
        midpoint = (ep[0] + ep[1]) / 2.0
        frame = min((f for f in occ.frames if ep[0] <= f <= ep[1]),
                    key=lambda f: (abs(f - midpoint), f))
        return PoseRef(
            source_label=occ.source_label,
            topology=occ.topology,
            trajectory=occ.trajectory,
            frame=int(frame),
            probe_resname=occ.probe_resname,
            probe_resid=occ.probe_resid,
            probe_resindex=occ.probe_resindex,
            episode=ep,
        )

    def to_row(self):
        """Flat scalar dict for CSV/TSV export. Contains no list or dict values."""
        cent = self.centroid if self.centroid is not None else (None, None, None)
        pose = self.best_pose()
        d = {
            "rank": self.rank,
            "site_id": self.site_id,
            "cosolvent": self.cosolvent,
            "n_voxels": self.n_voxels,
            "centroid_x": _round_or_none(cent[0], 3),
            "centroid_y": _round_or_none(cent[1], 3),
            "centroid_z": _round_or_none(cent[2], 3),
            "agfe_min": _round_or_none(self.agfe_min),
            "agfe_mean_top_pct": _round_or_none(self.agfe_mean_top_pct),
            "favorable_atomtypes": ",".join(self.favorable_atomtypes),
            "n_probe_molecules": self.n_probe_molecules,
            "total_residence_frames": self.total_residence_frames,
            "best_probe": pose.probe_resname if pose else None,
            "best_source": pose.source_label if pose else None,
            "best_frame": pose.frame if pose else None,
            "mmgbsa_delta_total": _round_or_none(
                min((r.delta_total for r in self.mmgbsa), default=None)
            ),
            "mmgbsa_n_results": len(self.mmgbsa),
            # LIE is an interaction energy, not a dG: reported as "total" so it is
            # not mistaken for the MMGBSA column beside it.
            "lie_total": _round_or_none(
                min((r.total for r in self.lie), default=None)),
            "lie_n_results": len(self.lie),
        }
        d.update({f"agfe_{k}": _round_or_none(v) for k, v in self.per_type_agfe.items()})
        d.update({k: v for k, v in self.properties.items()
                  if not isinstance(v, (list, dict))})
        return d

    # Retained because the dashboard, the PyMOL session writer and visualize_hotspots
    # all consume the flat shape.
    def to_dict(self):
        """Alias of :meth:`to_row`."""
        return self.to_row()

    def to_record(self):
        """Canonical nested dict — the single source of truth for persistence."""
        rec = self.to_row()
        rec["schema"] = 2
        rec["properties"] = self.properties
        rec["pocket_residues"] = [r.to_dict() for r in self.pocket_residues]
        rec["probe_occupancy"] = [o.to_dict() for o in self.probe_occupancy]
        rec["mmgbsa"] = [m.to_dict() for m in self.mmgbsa]
        rec["lie"] = [r.to_dict() for r in self.lie]
        return rec

    @classmethod
    def from_record(cls, d, voxel_mask, grid_origin, grid_delta):
        """Rebuild a Hotspot from :meth:`to_record` output plus its voxel mask.

        Dispatches on ``schema``. Schema 1 is the pre-unification flat dict with a nested
        ``_properties`` key, read so that checkpoints already on disk keep loading.

        :param d: record as produced by :meth:`to_record`.
        :param voxel_mask: 3-D boolean array of shape ``(nx, ny, nz)``.
        :param grid_origin: shape ``(3,)`` AGFE grid origin, Angstroms.
        :param grid_delta: shape ``(3,)`` voxel spacing, Angstroms.
        """
        schema = int(d.get("schema", 1))
        if schema not in (1, 2):
            raise ValueError(
                f"Unrecognised hotspot record schema {schema!r}; this build reads 1 and 2."
            )

        favorable_atomtypes = (
            d["favorable_atomtypes"].split(",") if d.get("favorable_atomtypes") else []
        )
        reserved = {"agfe_min", "agfe_mean_top_pct"}
        per_type_agfe = {
            k[5:]: float(v) for k, v in d.items()
            if k.startswith("agfe_") and k not in reserved and v is not None
        }
        site = cls(
            rank=int(d["rank"]),
            site_id=int(d["site_id"]),
            cosolvent=str(d["cosolvent"]),
            n_voxels=int(d["n_voxels"]),
            centroid=np.array([d["centroid_x"], d["centroid_y"], d["centroid_z"]],
                              dtype=float),
            agfe_min=float(d["agfe_min"]),
            agfe_mean_top_pct=float(d["agfe_mean_top_pct"]),
            voxel_mask=voxel_mask,
            favorable_atomtypes=favorable_atomtypes,
            per_type_agfe=per_type_agfe,
        )
        site.properties = dict(d.get("properties", d.get("_properties", {})))
        site.pocket_residues = [
            PocketResidue.from_dict(r) for r in d.get("pocket_residues", [])
        ]
        site.probe_occupancy = [
            ProbeOccupancy.from_dict(o) for o in d.get("probe_occupancy", [])
        ]
        site.mmgbsa = [MmgbsaResult.from_dict(m) for m in d.get("mmgbsa", [])]
        site.lie = [LieResult.from_dict(r) for r in d.get("lie", [])]
        site.grid_origin = np.asarray(grid_origin, dtype=float)
        site.grid_delta = np.asarray(grid_delta, dtype=float)
        return site

    def extract_surface(self, agfe_array, level=0.0, spacing=(1.0, 1.0, 1.0)):
        """Generate a surface mesh for this hotspot using marching cubes.

        Optional visualization helper, not part of hotspot detection.

        :param agfe_array: 3-D AGFE array, same shape as this site's voxel mask.
        :param level: iso-surface level passed to ``marching_cubes``.
        :param spacing: voxel spacing in Angstroms per dimension.
        :return: ``(verts, faces, normals, values)``, or ``None`` if extraction fails
            (scikit-image missing, degenerate surface).
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            return None
        try:
            vol = np.where(self.voxel_mask, agfe_array, np.nan)
            return marching_cubes(vol, level=level, spacing=spacing,
                                  allow_degenerate=False)
        except Exception:
            return None

    def __repr__(self):
        agfe = ("n/a" if self.agfe_min is None
                else f"{float(self.agfe_min):.3f} kcal/mol")
        top = ("n/a" if self.agfe_mean_top_pct is None
               else f"{float(self.agfe_mean_top_pct):.3f}")
        return (
            f"Hotspot(rank={self.rank}, cosolvent={self.cosolvent!r}, "
            f"n_voxels={self.n_voxels}, agfe_min={agfe}, "
            f"agfe_mean_top_pct={top})"
        )


# ---------------------------------------------------------------------------
# BindingSite — a pocket formed by one or more hotspots (any cosolvent)
# ---------------------------------------------------------------------------

class BindingSite:
    """A binding pocket formed by one or more :class:`Hotspot` objects.

    Geometry (volume/shape/centroid) is the pocket's own, computed on the union of the
    member hotspot masks; affinity/kinetics/chemistry are roll-ups over the members.
    ``combined`` and ``rank`` are set by
    :func:`cosolvkit.analysis.core.scoring.score_binding_sites`.

    Every aggregate except ``site_id`` is optional, so the class can also describe a
    pocket not derived from AGFE densities, where only members, mask and centroid are
    known. The chemistry roll-ups default to values derived from ``member_hotspots``.
    """

    def __init__(self, site_id, member_hotspots=None, voxel_mask=None, centroid=None,
                 agfe_min=None, agfe_mean_top_pct=None, volume=None,
                 solidity=None, extent=None, axis_major_length=None,
                 axis_minor_length=None,
                 favorable_atomtypes=None, pharmacophore=None, residence=None,
                 residence_metric=None,
                 cosolvents=None, n_total_cosolvents=None,
                 pocket_residues=None, grid_origin=None, grid_delta=None,
                 probe_chemotypes=None, n_total_probe_chemotypes=None):
        self.site_id = site_id
        self.member_hotspots = (list(member_hotspots)
                                if member_hotspots is not None else [])
        self.voxel_mask = voxel_mask                    # boolean 3D, on the common grid
        self.centroid = centroid                        # (3,), Angstroms
        self.agfe_min = agfe_min                        # best (most-negative) across members
        self.agfe_mean_top_pct = agfe_mean_top_pct
        self.volume = volume                            # Angstrom^3, union mask
        self.solidity = solidity
        self.extent = extent
        self.axis_major_length = axis_major_length
        self.axis_minor_length = axis_minor_length
        self.favorable_atomtypes = (list(favorable_atomtypes)
                                    if favorable_atomtypes else [])  # union over members
        self.pharmacophore = dict(pharmacophore) if pharmacophore else {}
        self.residence = residence                      # max of the kinetics metric, or None
        # Which survival-probability metric ``residence`` holds. Recorded because the
        # default is a FRACTION still bound at long lag, not a time, and would otherwise
        # be misread as a residence time.
        self.residence_metric = residence_metric
        if cosolvents is None:
            cosolvents = sorted({h.cosolvent for h in self.member_hotspots
                                 if h.cosolvent is not None})
        self.cosolvents = cosolvents                    # sorted, unique
        self.n_total_cosolvents = (len(cosolvents) if n_total_cosolvents is None
                                   else n_total_cosolvents)
        # Chemotype classes spanned by the probes hitting this site, and how many the
        # whole panel could express. Distinct from n_cosolvents (a plain probe count):
        # several probes of the same kind span fewer classes than two opposite ones.
        if probe_chemotypes is None:
            from cosolvkit.analysis.core.chemotypes import probe_chemotypes as _pc
            probe_chemotypes = _pc(self.cosolvents)
        self.probe_chemotypes = list(probe_chemotypes)
        self.n_total_probe_chemotypes = n_total_probe_chemotypes
        self.pocket_residues = pocket_residues if pocket_residues is not None else []
        self.grid_origin = grid_origin
        self.grid_delta = grid_delta
        self.properties = {}
        self.combined = None                            # set by score_binding_sites
        self.rank = None

    def add_property(self, name, value):
        """Attach an arbitrary named property."""
        self.properties[name] = value

    @property
    def n_hotspots(self):
        return len(self.member_hotspots)

    @property
    def n_cosolvents(self):
        return len(self.cosolvents)

    @property
    def probe_coverage(self):
        """Fraction of the panel's probes that hit this site (probe count, not chemistry)."""
        return (self.n_cosolvents / self.n_total_cosolvents
                if self.n_total_cosolvents else 0.0)

    @property
    def n_probe_chemotypes(self):
        return len(self.probe_chemotypes)

    @property
    def probe_chemotype_coverage(self):
        """Fraction of the panel's chemotype classes represented at this site.

        Falls back to the full class list as the denominator when the panel's own
        class count is unknown.
        """
        total = self.n_total_probe_chemotypes
        if not total:
            from cosolvkit.analysis.core.chemotypes import CHEMOTYPE_CLASSES
            total = len(CHEMOTYPE_CLASSES)
        return self.n_probe_chemotypes / total if total else 0.0

    @property
    def member_hotspot_ids(self):
        return [h.site_id for h in self.member_hotspots]

    @property
    def probe_occupancy(self):
        """Every member hotspot's occupancy records, concatenated."""
        return [occ for h in self.member_hotspots for occ in h.probe_occupancy]

    @property
    def n_probe_molecules(self):
        return len(self.probe_occupancy)

    @property
    def total_residence_frames(self):
        return sum(o.n_frames_bound for o in self.probe_occupancy)

    def best_pose(self, gap_tolerance=0):
        """The best bound frame across member hotspots, or None if none was occupied."""
        poses = [h.best_pose(gap_tolerance) for h in self.member_hotspots]
        poses = [p for p in poses if p is not None]
        if not poses:
            return None
        return min(poses, key=lambda p: (-(p.episode[1] - p.episode[0]),
                                         p.source_label, p.probe_resid))

    def to_row(self):
        """Flat dict for CSV/JSON export (binding_sites.csv schema)."""
        cent = self.centroid if self.centroid is not None else (None, None, None)
        pose = self.best_pose()
        d = {
            "site_id": self.site_id,
            "rank": self.rank,
            "combined": _round_or_none(self.combined),
            "cosolvents": ",".join(self.cosolvents),
            "n_cosolvents": self.n_cosolvents,
            "probe_coverage": _round_or_none(self.probe_coverage),
            "probe_chemotypes": ",".join(self.probe_chemotypes),
            "n_probe_chemotypes": self.n_probe_chemotypes,
            "probe_chemotype_coverage": _round_or_none(self.probe_chemotype_coverage),
            "n_hotspots": self.n_hotspots,
            "member_hotspot_ids": ",".join(str(i) for i in self.member_hotspot_ids),
            "centroid_x": _round_or_none(cent[0], 3),
            "centroid_y": _round_or_none(cent[1], 3),
            "centroid_z": _round_or_none(cent[2], 3),
            "agfe_min": _round_or_none(self.agfe_min),
            "agfe_mean_top_pct": _round_or_none(self.agfe_mean_top_pct),
            "volume": _round_or_none(self.volume, 3),
            "solidity": _round_or_none(self.solidity),
            "extent": _round_or_none(self.extent),
            "axis_major_length": _round_or_none(self.axis_major_length),
            "axis_minor_length": _round_or_none(self.axis_minor_length),
            "favorable_atomtypes": ",".join(self.favorable_atomtypes),
            "n_chemotypes": len(self.favorable_atomtypes),
            "residence": _round_or_none(self.residence),
            "residence_metric": self.residence_metric,
            # Mean over member hotspots, matching how `score_binding_sites` reads it. Exported
            # because it carries a non-zero DEFAULT weight: without this column the feature was
            # scored but invisible in binding_sites.csv, so nothing downstream (the dashboard
            # reranker, any external analysis) could see or re-weight it. None when the
            # accessible-volume mask was unavailable.
            "accessible_fraction": _round_or_none(self._mean_member_property(
                "accessible_fraction"), 4),
            "n_probe_molecules": self.n_probe_molecules,
            "total_residence_frames": self.total_residence_frames,
            "best_probe": pose.probe_resname if pose else None,
        }
        d.update({k: v for k, v in self.properties.items()
                  if not isinstance(v, (list, dict))})
        return d

    # Retained: binding_sites.csv writers and the dashboard consume the flat shape.
    def to_dict(self):
        """Alias of :meth:`to_row`."""
        return self.to_row()

    def _mean_member_property(self, name):
        """Mean of *name* over member hotspots that carry it, or None if none do.

        The MEAN rather than the max: a best-of-members summary is inflated by member count
        (rho -0.82 on the FosAKP benchmark).
        """
        vals = [h.properties[name] for h in (self.member_hotspots or [])
                if h.properties is not None and h.properties.get(name) is not None]
        return float(np.mean(vals)) if vals else None

    def __repr__(self):
        agfe = "n/a" if self.agfe_min is None else f"{float(self.agfe_min):.3f}"
        return (f"BindingSite(site_id={self.site_id}, rank={self.rank}, "
                f"n_hotspots={self.n_hotspots}, cosolvents={self.cosolvents}, "
                f"agfe_min={agfe})")
