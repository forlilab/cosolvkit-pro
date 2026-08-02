#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Utils functions
#

from typing import List, Tuple
import os
import logging
from openmm.app import *
from openmm import *
import pdbfixer
from openff.toolkit import Topology


MD_FORMAT_EXTENSIONS = {
    "AMBER": {"topology": ".prmtop", "position": ".rst7"},
    "GROMACS": {"topology": ".top", "position": ".gro"},
    "CHARMM": {"topology": ".psf", "position": ".crd"},
    "OPENMM": {"system": ".xml", "position": ".pdb", "topology": ".prmtop"}
}

class MutuallyExclusiveParametersError(Exception):
    """Raised when mutually exclusive parameters are supplied together."""
    pass

def setup_logging(level:str="INFO", filepath:str=None):

    """Set up the ``cosolvkit`` logger at an entry point, i.e. cli scripts.

    :param level: logging level name, defaults to "INFO"
    :type level: str
    :param filepath: optional log file; its directory is created if missing.
    :type filepath: str, optional
    :return: the configured logger
    :rtype: logging.Logger
    """
    outdir = os.path.dirname(filepath) if filepath else '.'
    os.makedirs(outdir, exist_ok=True)

    logger = logging.getLogger("cosolvkit")
    logger.setLevel(level.upper())

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    handlers = [logging.StreamHandler()]
    if filepath:
        handlers.append(logging.FileHandler(filepath, mode="a"))
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False  # keep records out of the root logger

    return logger

def fix_pdb(pdbfile: str, pdbxfile: str, keep_heterogens: bool=False) -> Tuple[Topology, List]:
    """Add missing atoms/residues/hydrogens and drop nonstandard residues.

    Terminal missing-residue gaps are not built back in.

    :param pdbfile: pdb string old format
    :type pdbfile: str
    :param pdbxfile: pdb string new format
    :type pdbxfile: str
    :param keep_heterogens: if False all heterogen atoms but waters are deleted, defaults to False
    :type keep_heterogens: bool, optional
    :return: new topology and positions
    :rtype: tuple[Topology, list]
    """
    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile, pdbxfile=pdbxfile)
    fixer.findMissingResidues()
    
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms() 
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    return fixer.topology, fixer.positions
    
def add_variants(topology: Topology, positions: list, variants: list=list()) -> Tuple[Topology, List]:
    """Adds variants for specific protonation states.

    :param topology: openmm topology
    :type topology: Topology
    :param positions: openmm positions
    :type positions: list
    :param variants: list of variants to apply for the protonation states, defaults to list()
    :type variants: list, optional
    :return: topology and positions with added protonation states
    :rtype: tuple[Topology, list]
    """
    modeller = Modeller(topology, positions)
    added_variants = modeller.addHydrogens(variants=variants)
    return modeller.topology, modeller.positions