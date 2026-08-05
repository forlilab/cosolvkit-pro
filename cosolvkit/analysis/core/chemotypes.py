#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Chemotype classes for cosolvent probes.
#
# Assignments are verified against PROBE_REFERENCE_SMILES in
# tests/test_probe_chemotype_chemistry.py, which is where the reasoning lives.
#

CHEMOTYPE_CLASSES = (
    "aromatic",
    "aliphatic",        # saturated carbon skeleton; NOT a synonym for hydrophobic
    "hydrophobic",      # apolar surface
    "polar",            # aprotic, dipole-dominated (see the table); orthogonal to donor/acceptor
    "hbond_donor",
    "hbond_acceptor",
    "anionic",
    "cationic",
)

# Chemotype classes per cosolvent residue name, for the standard CoSolvKit probe panel.
# A probe belongs to every class it can express, so entries are usually multi-class.
# Brittle if a resname changes; resnames are CoSolvKit labels and some collide with PDB
# component IDs for unrelated molecules, which matters when matching crystallographic ligands.
#
# `aliphatic` is applied for >=2 alkyl carbons or a branched alkyl; a lone methyl does not count
# (so MOH/ACM/ALD/ACN/ACT/MAM/MTM/MIM are untagged, EOH/IPA/NMA/DMS/DTM are). EDO has two carbons
# but both bear hydroxyls, so it presents no apolar surface.
#
# `polar` means APROTIC and dipole-dominated: a large permanent dipole with no H-bond donor, so the
# tag stays orthogonal to hbond_donor/hbond_acceptor instead of restating them. Protic dipolar
# probes (FMD/ACM/NMA, alcohols) are described by their donor/acceptor tags, and the charged probes
# by anionic/cationic; applying `polar` to those would make it near-universal and uninformative.
DEFAULT_PROBE_CHEMOTYPES = {
    # aromatics
    "BEN": ("aromatic", "hydrophobic"),                     # benzene
    "FBZ": ("aromatic", "hydrophobic"),                     # fluorobenzene
    "PHN": ("aromatic", "hbond_donor", "hbond_acceptor"),   # phenol
    "PRM": ("aromatic", "hbond_acceptor"),                  # pyrimidine
    "PYR": ("aromatic", "hbond_acceptor"),                  # pyridine
    "IMD": ("aromatic", "hbond_donor", "hbond_acceptor"),   # imidazole
    "IMI": ("aromatic", "hbond_donor", "hbond_acceptor"),   # imidazole (alt resname)
    "MIM": ("aromatic", "hbond_acceptor"),                  # 1-methylimidazole: no N-H
    "IMP": ("aromatic", "hbond_donor", "cationic"),         # imidazolium: both N-H, nothing accepts
    "THP": ("aromatic", "hydrophobic"),                     # thiophene: no aliphatic carbon
    "THZ": ("aromatic", "hbond_acceptor"),                  # thiazole
    "TTZ": ("aromatic", "hbond_acceptor", "anionic"),       # tetrazolate
    # aliphatic / apolar
    "PRP": ("aliphatic", "hydrophobic"),                    # propane
    # alcohols / polyols
    "MOH": ("hbond_donor", "hbond_acceptor"),               # methanol
    "EOH": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # ethanol
    "IPA": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # isopropanol
    "EDO": ("hbond_donor", "hbond_acceptor"),               # ethylene glycol
    # carbonyl / amide / nitrile
    "ALD": ("polar", "hbond_acceptor"),                     # acetaldehyde, 2.7 D
    "ACN": ("polar", "hbond_acceptor"),                     # acetonitrile, 3.9 D
    "FMD": ("hbond_donor", "hbond_acceptor"),               # formamide
    "ACM": ("hbond_donor", "hbond_acceptor"),               # acetamide
    "NMA": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # N-methylacetamide
    "DMS": ("aliphatic", "polar", "hbond_acceptor"),        # DMSO, 4.0 D
    # charged
    "ACT": ("anionic", "hbond_acceptor"),                   # acetate
    "MAM": ("cationic", "hbond_donor"),                     # methylammonium
    "MTA": ("cationic", "hbond_donor"),                     # methylammonium (alt resname)
    # Neutral amines. pKa ~10.6, so >99.9% protonated at pH 7.4 -- keep only if a neutral amine
    # probe is deliberate; MAM/MTA are the cationic counterparts.
    "MTM": ("hbond_donor", "hbond_acceptor"),               # methylamine (neutral)
    "DTM": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # dimethylamine (neutral)
}

# Reference structure per probe, so assignments can be checked against the molecule rather than
# the comment beside it. Not used at runtime and not the parameterisation source.
PROBE_REFERENCE_SMILES = {
    "BEN": "c1ccccc1",
    "FBZ": "Fc1ccccc1",
    "PHN": "Oc1ccccc1",
    "PRM": "c1cncnc1",
    "PYR": "c1ccncc1",
    "IMD": "c1c[nH]cn1",
    "IMI": "c1c[nH]cn1",
    "MIM": "Cn1ccnc1",          # a 2-/4-methylimidazole would retain an N-H and be a donor
    "IMP": "c1c[nH]c[nH+]1",
    "THP": "c1ccsc1",
    "THZ": "c1cscn1",
    "TTZ": "c1nnn[n-]1",
    "PRP": "CCC",
    "MOH": "CO",
    "EOH": "CCO",
    "IPA": "CC(C)O",
    "EDO": "OCCO",
    "ALD": "CC=O",
    "ACN": "CC#N",
    "FMD": "NC=O",
    "ACM": "CC(N)=O",
    "NMA": "CNC(C)=O",
    "DMS": "CS(C)=O",
    "ACT": "CC(=O)[O-]",
    "MAM": "C[NH3+]",
    "MTA": "C[NH3+]",
    "MTM": "CN",
    "DTM": "CNC",
}


def resolve_probe_chemotypes(overrides=None):
    """Return the resname -> tuple(classes) mapping, with *overrides* applied.

    :param overrides: ``{resname: [class, ...]}``, taking precedence over the built-in
        table. An unknown class name raises rather than being silently dropped.
    :return: dict mapping upper-case resname to a tuple of chemotype classes.
    """
    mapping = {k: tuple(v) for k, v in DEFAULT_PROBE_CHEMOTYPES.items()}
    for resname, classes in (overrides or {}).items():
        if isinstance(classes, str):
            classes = [classes]
        bad = [c for c in classes if c not in CHEMOTYPE_CLASSES]
        if bad:
            raise ValueError(
                f"Unknown chemotype class(es) {bad} for probe {resname!r}; "
                f"valid classes: {list(CHEMOTYPE_CLASSES)}"
            )
        mapping[str(resname).strip().upper()] = tuple(classes)
    return mapping


def probe_chemotypes(cosolvents, mapping=None):
    """Sorted chemotype classes spanned by *cosolvents* (unknown resnames contribute none)."""
    mapping = mapping if mapping is not None else DEFAULT_PROBE_CHEMOTYPES
    found = set()
    for name in cosolvents or ():
        found.update(mapping.get(str(name).strip().upper(), ()))
    return [c for c in CHEMOTYPE_CLASSES if c in found]


def n_available_chemotypes(cosolvents, mapping=None):
    """How many classes the whole panel could express — the coverage denominator.

    Uses the panel, not ``len(CHEMOTYPE_CLASSES)``, so a panel missing a class is not
    permanently capped below 1.0 for a reason unrelated to any site.
    """
    n = len(probe_chemotypes(cosolvents, mapping))
    return n if n > 0 else len(CHEMOTYPE_CLASSES)
