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
# A probe belongs to every class it can express, so benzene is purely aromatic while
# phenol is aromatic AND an H-bond donor and acceptor.
#
# This is what distinguishes *probe chemotype* coverage from plain probe coverage: a site
# hit by benzene, phenol, propane and fluorobenzene is hit by four probes but spans only
# the hydrophobic/aromatic corner of chemical space, whereas a site hit by acetate and
# methylammonium spans two opposite charges with two probes.
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

    *overrides* is ``{resname: [class, ...]}``; it beats the built-in table, so a target
    with custom probes (or a probe whose default classification you disagree with) needs
    no code change. An unknown class name raises rather than being silently dropped.
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

    Uses the panel rather than ``len(CHEMOTYPE_CLASSES)`` so that a run with no anionic
    probe is not permanently capped below 1.0 for a reason unrelated to the site.
    """
    n = len(probe_chemotypes(cosolvents, mapping))
    return n if n > 0 else len(CHEMOTYPE_CLASSES)
