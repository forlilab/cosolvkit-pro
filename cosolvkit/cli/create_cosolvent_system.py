import os
import io
import json
import time
import sys
import argparse
from collections import defaultdict
from cosolvkit.config import Config
from cosolvkit.utils import setup_logging, fix_pdb, add_variants, MD_FORMAT_EXTENSIONS
from cosolvkit.cosolvent_system import CosolventSystem, CosolventMembraneSystem
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Builds a cosolvent system for MD simulation.",
                                     epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2023 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""")
    
    parser.add_argument('-c', '--config', dest='config', required=True,
                        action='store', help='path to the json config file')
    parser.add_argument(
        "--iteratively_adjust_copies", action="store_true", default=False
    )
 
    return parser.parse_args()

def main():

    # Parse command line arguments
    args = cmd_lineparser()
    config_file = args.config

    # Load config file
    config = Config.from_config(config_file)
    os.makedirs(config.output_dir, exist_ok=True)

    # Set up logging
    logger=setup_logging(level="INFO", filepath=f"{config.output_dir}/cosolvkit.log")
    
    start = time.time()
    if (config.protein_path is not None and config.box_size is not None) or (config.protein_path is None and config.box_size is None):
        logger.error("Error! If the config file specifies a receptor, the box_size should be set to null and vice versa.")
        raise SystemExit("Error! If the config file specifies a receptor, the box_size should be set to null and vice versa.")

    if config.protein_path is not None:
        logger.info(f"Loading receptor file {config.protein_path}")
        try:
            with open(config.protein_path) as f:
                pdb_string = io.StringIO(f.read())
        except FileNotFoundError:
            logger.error(f"Error! File {config.protein_path} not found.")
            raise SystemExit(f"Error! File {config.protein_path} not found.")

        # Check if we need to clean the protein and add variants of residues
        if config.clean_protein:
            pdbfile = None
            pdbxfile = None
            if config.protein_path.endswith(".pdb"):
                pdbfile = pdb_string
            else:
                pdbxfile = pdb_string
            logger.info("Cleaning protein structure")
            protein_topology, protein_positions = fix_pdb(pdbfile=pdbfile,
                                                        pdbxfile=pdbxfile,
                                                        keep_heterogens=config.keep_heterogens)
        else:
            if not config.protein_path.endswith(".pdb"):
                pdb = PDBxFile(pdb_string)
            else:
                pdb = PDBFile(pdb_string)
            protein_topology, protein_positions = pdb.topology, pdb.positions

        # Call add_variants funtion to assing variants to the protein
        if len(config.variants.keys()) > 0:
            logger.info("Adding variants to the protein")
            variants_list = list()
            residues = list(protein_topology.residues())
            mapping = defaultdict(list)
            for r in residues:
                mapping[r.chain.id].append(int(r.id))

            for chain in mapping:
                for res_number in mapping[chain]:
                    key = f"{chain}:{res_number}"
                    if key in config.variants:
                        variants_list.append(config.variants[key])
                    else:
                        variants_list.append(None)
            protein_topology, protein_positions = add_variants(protein_topology, protein_positions, variants_list)

    else:
        assert config.box_size is not None, "box_size is None in the config"
        # Create empty modeller since there's nothing in the system yet
        config.box_size = config.box_size * openmmunit.angstrom
        protein_topology, protein_positions = Topology(), None

    protein_modeller = Modeller(protein_topology, protein_positions)

    engine = list(config.md_engine.keys())[0]

    # Check repulsive forces and md engine consistency
    if (engine.upper() != "OPENMM") and len(config.repulsive_forces) > 0:
        logger.warning("Custom repulsive forces will only work if the MD engine is OpenMM!")

    # Load cosolvents dictionary
    with open(config.cosolvents) as fi:
        cosolvents = json.load(fi)

    if config.membrane:
        logger.info("Building a membrane-cosolvent system")
        cosolv_system = CosolventMembraneSystem(cosolvents=cosolvents,
                                                forcefields=config.md_engine,
                                                small_molecule_ff=config.small_molecule_ff,
                                                ligands=config.ligands,
                                                simulation_format=engine,
                                                modeller=protein_modeller,
                                                padding=config.padding,
                                                box_size=config.box_size,
                                                lipid_type=config.lipid_type,
                                                lipid_patch_path=config.lipid_patch_path)
        cosolv_system.add_membrane(cosolvent_placement=config.memb_cosolv_placement,
                                positive_ion=config.positive_ion,
                                negative_ion=config.negative_ion,
                                waters_to_keep=config.waters_to_keep)
        cosolv_system.build(positive_ion=config.positive_ion, negative_ion=config.negative_ion, iteratively_adjust_copies=args.iteratively_adjust_copies)
    else:
        logger.info("Building cosolvent system..")
        cosolv_system = CosolventSystem(cosolvents=cosolvents,
                                        forcefields=config.md_engine,
                                        small_molecule_ff=config.small_molecule_ff,
                                        ligands=config.ligands,
                                        simulation_format=engine,
                                        modeller=protein_modeller,
                                        padding=config.padding,
                                        box_size=config.box_size)
        cosolv_system.build(solvent_smiles=config.solvent_smiles,
                            n_solvent_molecules=config.solvent_copies,
                            positive_ion=config.positive_ion,
                            negative_ion=config.negative_ion,
                            iteratively_adjust_copies=args.iteratively_adjust_copies)

    # add the repulsive forces if specified in the config file
    if len(config.repulsive_forces) > 0:
        cosolv_system.add_repulsive_forces(config.repulsive_forces)

    logger.info("Saving topology file")
    cosolv_system.save_topology(topology=cosolv_system.modeller.topology,
                                positions=cosolv_system.modeller.positions,
                                system=cosolv_system.system,
                                simulation_format=engine,
                                forcefield=cosolv_system.forcefield,
                                out_path=config.output_dir)
    logger.info(f"All done! System building took {(time.time() - start)/60:.2f} min.")
    return


if __name__ == "__main__":
    sys.exit(main())