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
        ``{cosolvent_name: {cosolvent_resid: [frame_index, ...]}}`` — sorted frames in
        which each cosolvent molecule was within the contact cutoff of this residue.
    properties : dict
        Extensible bag for arbitrary extra scalar properties.
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

    cosolvent_contacts: Dict[str, Dict[int, List[int]]] = field(
        default_factory=dict
    )
    properties: Dict[str, Any] = field(default_factory=dict)

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
        # Grid metadata, set by HotspotDetector.detect() after construction; required to
        # combine voxel masks across grids (e.g. binding-site grouping).
        self.grid_origin = None                       # np.ndarray (3,), Angstroms
        self.grid_delta = None                        # np.ndarray (3,), Angstroms per voxel

    def add_property(self, name, value):
        """Attach an arbitrary named property."""
        self.properties[name] = value

    @classmethod
    def from_dict(cls, d, voxel_mask, grid_origin, grid_delta):
        """Reconstruct a Hotspot from checkpointed metadata plus its voxel mask.

        Inverse of :meth:`HotspotDetector.save_checkpoint`, not of the CSV/JSON
        exports (those carry no voxel mask).

        :param d: metadata as produced by :meth:`to_dict`, optionally with a
            ``_properties`` key holding the extensible properties dict.
        :param voxel_mask: 3-D boolean array of shape ``(nx, ny, nz)``.
        :param grid_origin: shape ``(3,)`` AGFE grid origin, Angstroms.
        :param grid_delta: shape ``(3,)`` voxel spacing, Angstroms.
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
            # Mean over member hotspots, matching how `score_binding_sites` reads it. Exported
            # because it carries a non-zero DEFAULT weight: without this column the feature was
            # scored but invisible in binding_sites.csv, so nothing downstream (the dashboard
            # reranker, any external analysis) could see or re-weight it. None when the
            # accessible-volume mask was unavailable.
            "accessible_fraction": _round_or_none(self._mean_member_property(
                "accessible_fraction"), 4),
        }
        d.update(self.properties)
        return d

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
