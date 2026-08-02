#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Chemotype classes for cosolvent probes.
#
# Dependency leaf: standard library only. Must NOT import from cosolvkit.analysis.sites
# or core.grid.
#

CHEMOTYPE_CLASSES = (
    "aromatic",
    "aliphatic",
    "hbond_donor",
    "hbond_acceptor",
    "anionic",
    "cationic",
)

# Chemotype classes per cosolvent residue name, for the standard CoSolvKit probe panel.
# A probe belongs to every class it can express, so entries are usually multi-class.
# Chemotype coverage therefore differs from probe coverage: many probes from the same
# corner of chemical space span fewer classes than two probes from opposite corners.
DEFAULT_PROBE_CHEMOTYPES = {
    # aromatics
    "BEN": ("aromatic", "aliphatic"),                       # benzene
    "FBZ": ("aromatic", "aliphatic"),                       # fluorobenzene
    "PHN": ("aromatic", "hbond_donor", "hbond_acceptor"),   # phenol
    "PRM": ("aromatic", "hbond_acceptor"),                  # pyrimidine
    "PYR": ("aromatic", "hbond_acceptor"),                  # pyridine
    "IMD": ("aromatic", "hbond_donor", "hbond_acceptor"),   # imidazole
    "IMI": ("aromatic", "hbond_donor", "hbond_acceptor"),   # imidazole (alt resname)
    "MIM": ("aromatic", "hbond_acceptor"),                  # methylimidazole
    "IMP": ("aromatic", "hbond_donor", "cationic"),          # imidazolium
    "THP": ("aromatic", "aliphatic"),                       # thiophene
    "THZ": ("aromatic", "hbond_acceptor"),                  # thiazole
    "TTZ": ("aromatic", "hbond_acceptor", "anionic"),        # tetrazolate
    # aliphatic / apolar
    "PRP": ("aliphatic",),                                  # propane
    # alcohols / polyols
    "MOH": ("hbond_donor", "hbond_acceptor"),               # methanol
    "EOH": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # ethanol
    "IPA": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # isopropanol
    "EDO": ("hbond_donor", "hbond_acceptor"),               # ethylene glycol
    # carbonyl / amide / nitrile
    "ALD": ("hbond_acceptor",),                             # acetaldehyde
    "ACN": ("hbond_acceptor",),                             # acetonitrile
    "FMD": ("hbond_donor", "hbond_acceptor"),               # formamide
    "ACM": ("hbond_donor", "hbond_acceptor"),               # acetamide
    "NMA": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # N-methylacetamide
    "DMS": ("aliphatic", "hbond_acceptor"),                 # DMSO
    # charged
    "ACT": ("anionic", "hbond_acceptor"),                   # acetate
    "MAM": ("cationic", "hbond_donor"),                     # methylammonium
    "MTA": ("cationic", "hbond_donor"),                     # methylammonium (alt resname)
    "MTM": ("hbond_donor", "hbond_acceptor"),               # methylamine (neutral)
    "DTM": ("aliphatic", "hbond_donor", "hbond_acceptor"),  # dimethylamine
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
