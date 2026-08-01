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
from typing import Any, Dict, List, Optional


def _round_or_none(value, ndigits=4):
    """Round *value* for CSV/JSON export, passing ``None``/non-finite through as ``None``.

    Sites that do not come from an AGFE density map (e.g. crystallographic ligand
    positions used as ground truth) legitimately have no affinity or shape values.
    """
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return round(fval, ndigits) if np.isfinite(fval) else None


# ---------------------------------------------------------------------------
# PocketResidue — per-residue data attached to a Hotspot
# ---------------------------------------------------------------------------

@dataclass
class PocketResidue:
    """A protein residue that lines a cosolvent hotspot cavity.

    Populated incrementally by :class:`PocketPropertyCalculator` methods:

    * :meth:`find_pocket_residues` — identity + proximity fields
    * :meth:`annotate_residue_rmsf` — ``rmsf``
    * :meth:`compute_cosolvent_contacts` — ``cosolvent_contacts``
    * :func:`set_residue_embeddings` — ``embedding`` / ``embedding_model``

    Attributes
    ----------
    resid : int
        PDB residue number (MDAnalysis ``resid``).
    resindex : int
        Universe-internal index (stable across trajectory frames).
    resname : str
        Three-letter residue code, e.g. ``"LEU"``.
    chain : str
        Segment / chain ID.
    n_contact_voxels : int
        Number of distinct blob voxels within *cutoff* Å of any heavy atom.
    min_dist_ang : float
        Distance in Å to the nearest blob voxel.
    contact_fraction : float
        ``n_contact_voxels / total_blob_voxels``.
    rmsf : float or None
        Cα RMSF in Å (set by :meth:`annotate_residue_rmsf`).
    embedding : np.ndarray or None
        PLM feature vector, shape ``(n_dims,)`` (injected externally).
    embedding_model : str or None
        Name of the model that produced ``embedding``.
    cosolvent_contacts : dict
        ``{cosolvent_name: {cosolvent_resid: [frame_index, ...]}}`` — for each
        cosolvent molecule (identified by its MDAnalysis ``resid``), the sorted
        list of trajectory frame indices where it was within *contact_cutoff*
        of any heavy atom of this residue.
    properties : dict
        Extensible bag for arbitrary extra scalar properties.
    """

    # Required positional fields
    resid:            int
    resindex:         int
    resname:          str
    chain:            str
    n_contact_voxels: int
    min_dist_ang:     float
    contact_fraction: float

    # Optional, populated by separate methods
    rmsf:            Optional[float]       = None
    embedding:       Optional[np.ndarray]  = None
    embedding_model: Optional[str]         = None

    cosolvent_contacts: Dict[str, Dict[int, List[int]]] = field(
        default_factory=dict
    )
    properties: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def contact_frames(self, cosolvent_name: str) -> List[int]:
        """Sorted union of frames where ANY molecule of *cosolvent_name* contacted this residue."""
        mol_dict = self.cosolvent_contacts.get(cosolvent_name, {})
        all_frames: set = set()
        for frames in mol_dict.values():
            all_frames.update(frames)
        return sorted(all_frames)

    def contact_resids(self, cosolvent_name: str) -> List[int]:
        """Sorted list of cosolvent molecule resids that ever contacted this residue."""
        return sorted(self.cosolvent_contacts.get(cosolvent_name, {}).keys())

    def n_contact_events(self, cosolvent_name: str) -> int:
        """Total (molecule, frame) pairs — proxy for raw contact frequency."""
        return sum(
            len(frames)
            for frames in self.cosolvent_contacts.get(cosolvent_name, {}).values()
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-safe dict representation.

        * ``embedding`` is serialized as ``list[float]`` (or ``None``).
        * ``cosolvent_contacts`` keys are stringified for JSON compatibility.
        """
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
            "cosolvent_contacts": {
                cosolvent: {str(rid): frames for rid, frames in mol_dict.items()}
                for cosolvent, mol_dict in self.cosolvent_contacts.items()
            },
            "properties": self.properties,
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
        raw = d.get("cosolvent_contacts", {})
        pr.cosolvent_contacts = {
            cosolvent: {int(rid): list(frames) for rid, frames in mol_dict.items()}
            for cosolvent, mol_dict in raw.items()
        }
        pr.properties = dict(d.get("properties", {}))
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

    Stores all computed scores and an extensible ``properties`` dict so that
    downstream analyses (e.g. residence time, pharmacophore annotation) can
    attach extra data without subclassing.

    Normally constructed by :class:`HotspotDetector`. Every AGFE-derived argument
    is optional, so the class doubles as a plain occupancy-region record for sites
    that come from somewhere other than a density map — e.g. a crystallographic
    ligand copy used as benchmark ground truth. Such a hotspot carries only
    ``cosolvent`` (the chemical species), ``centroid`` and ``voxel_mask``, which is
    all :func:`cosolvkit.analysis.sites.binding_sites.group_hotspots` needs.
    """

    def __init__(self, rank, site_id, cosolvent, n_voxels=0, centroid=None,
                 agfe_min=None, agfe_mean_top_pct=None, voxel_mask=None,
                 favorable_atomtypes=None, per_type_agfe=None):
        self.rank = rank                            # int; 1 = most-negative agfe_min
        self.site_id = site_id                      # label from scipy.ndimage.label
        self.cosolvent = cosolvent                  # str residue name
        self.n_voxels = n_voxels                    # int
        self.centroid = centroid                    # np.ndarray (3,), Angstroms
        self.agfe_min = agfe_min                    # float, kcal/mol, or None
        self.agfe_mean_top_pct = agfe_mean_top_pct  # float, kcal/mol, or None
        self.voxel_mask = voxel_mask                # boolean 3D ndarray, same shape as AGFE grid
        self.favorable_atomtypes = (list(favorable_atomtypes)
                                    if favorable_atomtypes else [])  # List[str]
        self.per_type_agfe = dict(per_type_agfe) if per_type_agfe else {}  # min AGFE per type
        self.properties = {}                          # extensible user properties
        self.pocket_residues = []                     # List[PocketResidue], populated on demand
        # Grid spatial metadata — set by HotspotDetector.detect() after construction.
        # Required for cross-grid voxel-mask computations (e.g. binding-site grouping).
        self.grid_origin = None                       # np.ndarray (3,), Angstroms
        self.grid_delta = None                        # np.ndarray (3,), Angstroms per voxel

    def add_property(self, name, value):
        """Attach an arbitrary property (e.g. ``site.add_property('residence_time_ns', 12.4)``)."""
        self.properties[name] = value

    @classmethod
    def from_dict(cls, d, voxel_mask, grid_origin, grid_delta):
        """Reconstruct a Hotspot from a serialized dict and its voxel mask.

        This is the inverse of the data written by
        :meth:`HotspotDetector.save_checkpoint`.  It is not intended for use
        with the human-readable CSV/JSON exports (those do not contain the
        voxel mask).

        Parameters
        ----------
        d : dict
            Metadata dict as produced by :meth:`to_dict` plus an optional
            ``_properties`` key carrying the extensible properties dict.
        voxel_mask : np.ndarray
            3-D boolean array of shape ``(nx, ny, nz)``.
        grid_origin : np.ndarray
            Shape ``(3,)`` origin of the AGFE grid in Angstroms.
        grid_delta : np.ndarray
            Shape ``(3,)`` voxel spacing in Angstroms.
        """
        favorable_atomtypes = (
            d["favorable_atomtypes"].split(",")
            if d.get("favorable_atomtypes")
            else []
        )
        per_type_agfe = {
            k[5:]: float(v)
            for k, v in d.items()
            if k.startswith("agfe_") and k not in ("agfe_min", "agfe_mean_top_pct")
        }
        site = cls(
            rank=int(d["rank"]),
            site_id=int(d["site_id"]),
            cosolvent=str(d["cosolvent"]),
            n_voxels=int(d["n_voxels"]),
            centroid=np.array([d["centroid_x"], d["centroid_y"], d["centroid_z"]], dtype=float),
            agfe_min=float(d["agfe_min"]),
            agfe_mean_top_pct=float(d["agfe_mean_top_pct"]),
            voxel_mask=voxel_mask,
            favorable_atomtypes=favorable_atomtypes,
            per_type_agfe=per_type_agfe,
        )
        site.properties = dict(d.get("_properties", {}))
        site.pocket_residues = [
            PocketResidue.from_dict(r) for r in d.get("pocket_residues", [])
        ]
        site.grid_origin = np.asarray(grid_origin, dtype=float)
        site.grid_delta = np.asarray(grid_delta, dtype=float)
        return site

    def to_dict(self):
        """Flat dict for CSV/JSON export. Includes base scores and ``properties``."""
        cent = self.centroid if self.centroid is not None else (None, None, None)
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
        }
        d.update({f"agfe_{k}": _round_or_none(v) for k, v in self.per_type_agfe.items()})
        d.update(self.properties)
        if self.pocket_residues:
            d["pocket_residues"] = [r.to_dict() for r in self.pocket_residues]
        return d

    def extract_surface(self, agfe_array, level=0.0, spacing=(1.0, 1.0, 1.0)):
        """Generate a surface mesh for this hotspot using marching cubes.

        This is an optional visualization helper and is not called during
        normal hotspot detection.  Returns ``None`` silently on failure.

        Parameters
        ----------
        agfe_array : np.ndarray
            3-D AGFE array with the same shape as this site's voxel mask.
        level : float
            Iso-surface level passed to ``marching_cubes`` (default 0.0).
        spacing : tuple of float
            Voxel spacing in each dimension, e.g. ``(0.5, 0.5, 0.5)`` for a
            0.5 Å grid.

        Returns
        -------
        tuple or None
            ``(verts, faces, normals, values)`` from
            ``skimage.measure.marching_cubes``, or ``None`` if the extraction
            fails (e.g. scikit-image not available, degenerate surface).
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

    Geometry (volume/shape/centroid) is the pocket's own, computed on the
    union of member hotspot masks; affinity/kinetics/chemistry are roll-ups of
    the members. ``combined`` and ``rank`` are set by
    :func:`cosolvkit.analysis.core.scoring.score_binding_sites`.

    Every aggregate except ``site_id`` is optional so the class can also describe a
    pocket that was not derived from AGFE densities — e.g. a ground-truth site built
    from crystallographic ligand copies, where only the members, mask and centroid
    are known. ``cosolvents`` / ``n_total_cosolvents`` default to values derived from
    ``member_hotspots``.
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
                                if member_hotspots is not None else [])  # list[Hotspot]
        self.voxel_mask = voxel_mask                    # boolean 3D ndarray on the common grid
        self.centroid = centroid                        # np.ndarray (3,), Angstroms
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
        # default is ``sp_plateau`` (a FRACTION still bound at long lag), not a time —
        # the column would otherwise be silently misread as a residence time.
        self.residence_metric = residence_metric
        # Default the chemistry roll-ups from the members rather than requiring them.
        if cosolvents is None:
            cosolvents = sorted({h.cosolvent for h in self.member_hotspots
                                 if h.cosolvent is not None})
        self.cosolvents = cosolvents                    # sorted unique list[str]
        self.n_total_cosolvents = (len(cosolvents) if n_total_cosolvents is None
                                   else n_total_cosolvents)
        # Chemotype classes spanned by the probes that hit this site, and how many the
        # whole panel could express. Distinct from n_cosolvents: four hydrophobes span
        # fewer classes than one anion plus one cation.
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
        """Attach an arbitrary property, e.g. the fused field features written by
        :func:`cosolvkit.analysis.core.site_features.fused_site_features`."""
        self.properties[name] = value

    @property
    def n_hotspots(self):
        return len(self.member_hotspots)

    @property
    def n_cosolvents(self):
        return len(self.cosolvents)

    @property
    def probe_coverage(self):
        return (self.n_cosolvents / self.n_total_cosolvents
                if self.n_total_cosolvents else 0.0)

    @property
    def n_probe_chemotypes(self):
        return len(self.probe_chemotypes)

    @property
    def probe_chemotype_coverage(self):
        """Fraction of the panel's chemotype classes represented at this site.

        Falls back to the full class list as the denominator when the panel's own
        class count was not supplied.
        """
        total = self.n_total_probe_chemotypes
        if not total:
            from cosolvkit.analysis.core.chemotypes import CHEMOTYPE_CLASSES
            total = len(CHEMOTYPE_CLASSES)
        return self.n_probe_chemotypes / total if total else 0.0

    @property
    def member_hotspot_ids(self):
        return [h.site_id for h in self.member_hotspots]

    def to_dict(self):
        """Flat dict for CSV/JSON export (binding_sites.csv schema)."""
        cent = self.centroid if self.centroid is not None else (None, None, None)
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
        }
        d.update(self.properties)
        return d

    def __repr__(self):
        agfe = "n/a" if self.agfe_min is None else f"{float(self.agfe_min):.3f}"
        return (f"BindingSite(site_id={self.site_id}, rank={self.rank}, "
                f"n_hotspots={self.n_hotspots}, cosolvents={self.cosolvents}, "
                f"agfe_min={agfe})")
