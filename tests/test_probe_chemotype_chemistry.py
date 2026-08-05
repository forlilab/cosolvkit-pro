"""Every chemotype assignment must agree with the probe's actual structure.

The table in ``core/chemotypes.py`` is hand-maintained, and the only thing tying an entry to a
molecule used to be the comment beside it. That let one straightforwardly wrong assignment sit
there: thiophene was tagged ``aliphatic`` despite having zero non-aromatic carbons (it is
``hydrophobic`` -- logP 1.75, the most lipophilic probe in the table).

These tests derive aromaticity, formal charge and H-bond donor/acceptor counts
from ``PROBE_REFERENCE_SMILES`` with RDKit and assert the table agrees, so the next probe added
cannot drift from its own structure.
"""

import pytest

from cosolvkit.analysis.core.chemotypes import (
    CHEMOTYPE_CLASSES, DEFAULT_PROBE_CHEMOTYPES, PROBE_REFERENCE_SMILES,
)

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit needed to verify chemistry")
from rdkit.Chem import Lipinski  # noqa: E402

PROBES = sorted(DEFAULT_PROBE_CHEMOTYPES)

# RDKit's Lipinski acceptor SMARTS counts divalent sulfur, so it reports 1 acceptor for thiophene.
# Chemically the ring sulfur's lone pairs are delocalised into the aromatic system and it is not a
# meaningful acceptor, so the table is right and RDKit is over-inclusive here.
HBA_EXCEPTIONS = {"THP"}


def _mol(res):
    m = Chem.MolFromSmiles(PROBE_REFERENCE_SMILES[res])
    assert m is not None, f"unparseable reference SMILES for {res}"
    return m


def test_every_probe_has_a_reference_structure():
    missing = set(DEFAULT_PROBE_CHEMOTYPES) - set(PROBE_REFERENCE_SMILES)
    extra = set(PROBE_REFERENCE_SMILES) - set(DEFAULT_PROBE_CHEMOTYPES)
    assert not missing, f"probes with no reference SMILES: {sorted(missing)}"
    assert not extra, f"reference SMILES with no chemotype entry: {sorted(extra)}"


def test_all_classes_are_known():
    for res, classes in DEFAULT_PROBE_CHEMOTYPES.items():
        unknown = [c for c in classes if c not in CHEMOTYPE_CLASSES]
        assert not unknown, f"{res} declares unknown class(es) {unknown}"


@pytest.mark.parametrize("res", PROBES)
def test_aromatic_tag_matches_structure(res):
    m = _mol(res)
    is_arom = any(a.GetIsAromatic() for a in m.GetAtoms())
    assert is_arom == ("aromatic" in DEFAULT_PROBE_CHEMOTYPES[res])


@pytest.mark.parametrize("res", PROBES)
def test_aliphatic_tag_requires_a_non_aromatic_carbon(res):
    """The thiophene bug: `aliphatic` is not a synonym for `hydrophobic`."""
    m = _mol(res)
    n_ali_c = sum(1 for a in m.GetAtoms()
                  if a.GetSymbol() == "C" and not a.GetIsAromatic())
    if "aliphatic" in DEFAULT_PROBE_CHEMOTYPES[res]:
        assert n_ali_c > 0, f"{res} is tagged aliphatic but has no non-aromatic carbon"


def test_there_is_no_halogen_class():
    """Removed: fluorobenzene was the only member and fluorine has no sigma-hole, so the class
    could never mean halogen-bond donation, and "contains a halogen" described no interaction.
    A consequence to keep in mind: FBZ and BEN now carry identical chemotypes, so swapping one for
    the other changes nothing a coverage feature can see."""
    assert "halogen" not in CHEMOTYPE_CLASSES
    for res, classes in DEFAULT_PROBE_CHEMOTYPES.items():
        assert "halogen" not in classes, f"{res} still carries the removed halogen tag"
    assert DEFAULT_PROBE_CHEMOTYPES["FBZ"] == DEFAULT_PROBE_CHEMOTYPES["BEN"]


@pytest.mark.parametrize("res", PROBES)
def test_polar_means_aprotic(res):
    """`polar` is reserved for dipole-dominated APROTIC probes, so it does not restate
    hbond_donor. A probe that donates an H-bond is described by that, not by `polar`."""
    if "polar" in DEFAULT_PROBE_CHEMOTYPES[res]:
        assert Lipinski.NumHDonors(_mol(res)) == 0, \
            f"{res} is tagged polar but donates an H-bond"
        assert Chem.GetFormalCharge(_mol(res)) == 0, \
            f"{res} is tagged polar but is charged; use anionic/cationic"


def test_polar_is_assigned():
    """It used to be declared and never used, so `probe_chemotypes` could not return it."""
    tagged = {r for r, c in DEFAULT_PROBE_CHEMOTYPES.items() if "polar" in c}
    assert tagged == {"ACN", "DMS", "ALD"}, f"unexpected polar set: {sorted(tagged)}"


def test_no_declared_class_is_dead():
    """Every class must be reachable, now that `polar` is assigned and `halogen` is gone."""
    used = set().union(*DEFAULT_PROBE_CHEMOTYPES.values())
    dead = [c for c in CHEMOTYPE_CLASSES if c not in used]
    assert not dead, f"declared but never assigned: {dead}"


@pytest.mark.parametrize("res", PROBES)
def test_charge_tags_match_formal_charge(res):
    q = Chem.GetFormalCharge(_mol(res))
    classes = DEFAULT_PROBE_CHEMOTYPES[res]
    assert (q < 0) == ("anionic" in classes), f"{res} formal charge {q}"
    assert (q > 0) == ("cationic" in classes), f"{res} formal charge {q}"
    assert not ("anionic" in classes and "cationic" in classes)


@pytest.mark.parametrize("res", PROBES)
def test_hbond_donor_tag_matches_structure(res):
    m = _mol(res)
    assert (Lipinski.NumHDonors(m) > 0) == \
        ("hbond_donor" in DEFAULT_PROBE_CHEMOTYPES[res]), \
        f"{res}: RDKit donors={Lipinski.NumHDonors(m)}"


@pytest.mark.parametrize("res", PROBES)
def test_hbond_acceptor_tag_matches_structure(res):
    if res in HBA_EXCEPTIONS:
        pytest.skip(f"{res}: RDKit counts divalent S as an acceptor; see HBA_EXCEPTIONS")
    m = _mol(res)
    assert (Lipinski.NumHAcceptors(m) > 0) == \
        ("hbond_acceptor" in DEFAULT_PROBE_CHEMOTYPES[res]), \
        f"{res}: RDKit acceptors={Lipinski.NumHAcceptors(m)}"


def test_cations_do_not_also_accept_and_anions_do_not_also_donate():
    """A fully protonated cation has no lone pair left to accept; a deprotonated anion has no H."""
    for res, classes in DEFAULT_PROBE_CHEMOTYPES.items():
        if "cationic" in classes:
            assert "hbond_acceptor" not in classes, f"{res} is cationic and tagged acceptor"
        if "anionic" in classes:
            assert "hbond_donor" not in classes, f"{res} is anionic and tagged donor"


def test_aliases_agree_with_their_canonical_probe():
    """Alt resnames must not drift from the probe they duplicate."""
    for alias, canonical in (("IMI", "IMD"), ("MTA", "MAM")):
        assert PROBE_REFERENCE_SMILES[alias] == PROBE_REFERENCE_SMILES[canonical]
        assert DEFAULT_PROBE_CHEMOTYPES[alias] == DEFAULT_PROBE_CHEMOTYPES[canonical], \
            f"{alias} and {canonical} are the same molecule but carry different chemotypes"
