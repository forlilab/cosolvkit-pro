import os
import logging
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule
from openmmforcefields.generators import (
    EspalomaTemplateGenerator,
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)
from openmmforcefields.generators.template_generators import SmallMoleculeTemplateGenerator

logger = logging.getLogger(__name__)

_FAMILY_ALIASES = {
    "OPENFF": "OPENFF",
    "ESPALOMA": "ESPALOMA",
    "GAFF": "GAFF",
}


def parse_small_molecule_ff(ff_str: str) -> Tuple[str, str]:
    """Parse a 'family-version' FF string into (full_name, family).

    Accepts strings of the form ``<family>-<version>`` (e.g. ``openff-2.3.0``,
    ``espaloma-0.3.2``, ``gaff-2.11``). The family prefix selects the
    openmmforcefields template generator; the full name (including version) is
    passed through to that generator.

    Returns
    -------
    tuple[str, str]
        ``(full_name, family)`` — e.g. ``("gaff-2.11", "GAFF")``.
    """
    if not isinstance(ff_str, str) or "-" not in ff_str:
        raise ValueError(
            "Small molecule forcefield must include a version suffix, "
            "e.g. 'espaloma-0.3.2', 'gaff-2.11', 'openff-2.3.0'. "
            f"Got: {ff_str!r}"
        )
    family_prefix = ff_str.split("-", 1)[0].upper()
    if family_prefix not in _FAMILY_ALIASES:
        raise ValueError(
            f"Unknown forcefield family {family_prefix!r}. "
            "Supported prefixes: espaloma-*, gaff-*, openff-*."
        )
    return ff_str, _FAMILY_ALIASES[family_prefix]


def load_molecule_from_file(path: str, smiles: Optional[str] = None) -> Molecule:
    """Load an OpenFF Molecule from a .sdf, .mol2, or .pdb file.

    Parameters
    ----------
    path : str
        Path to the ligand file.
    smiles : str, optional
        Reference SMILES used to assign correct bond orders after reading.
        Useful for PDB files or SDF files lacking bond-order information.

    Returns
    -------
    openff.toolkit.Molecule
    """
    ext = os.path.splitext(path)[1].lower()
    sanitize = smiles is None
    remove_hs = smiles is not None

    if ext == ".pdb":
        rdkit_mol = Chem.MolFromPDBFile(path, sanitize=sanitize, removeHs=remove_hs)
    elif ext in (".sdf", ".mol2"):
        rdkit_mol = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=remove_hs)[0]
    else:
        raise ValueError(
            f"Unsupported ligand format {ext!r}. Supported: .sdf, .mol2, .pdb"
        )

    if rdkit_mol is None:
        raise RuntimeError(f"RDKit could not read a molecule from {path!r}")

    if smiles is not None:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(f"Could not parse reference SMILES: {smiles!r}")
        rdkit_mol = AllChem.AssignBondOrdersFromTemplate(template, rdkit_mol)
        Chem.SanitizeMol(rdkit_mol)

    return Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)


def get_template_generator(
    molecules, ff_str: str
) -> SmallMoleculeTemplateGenerator:
    """Return the appropriate openmmforcefields template generator.

    Parameters
    ----------
    molecules : Molecule or list[Molecule]
        OpenFF Molecule(s) to register with the generator.
    ff_str : str
        Versioned FF string, e.g. ``"gaff-2.11"``, ``"espaloma-0.3.2"``,
        ``"openff-2.3.0"``.

    Returns
    -------
    SmallMoleculeTemplateGenerator
    """
    full_name, family = parse_small_molecule_ff(ff_str)

    if family == "ESPALOMA":
        return EspalomaTemplateGenerator(
            molecules=molecules,
            forcefield=full_name,
            template_generator_kwargs={
                "reference_forcefield": "openff_unconstrained-2.1.0",
                "charge_method": "nn",
            },
        )
    elif family == "GAFF":
        return GAFFTemplateGenerator(molecules=molecules, forcefield=full_name)
    elif family == "OPENFF":
        return SMIRNOFFTemplateGenerator(molecules=molecules, forcefield=full_name)
