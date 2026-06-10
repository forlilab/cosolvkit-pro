:: _cmdline:

cosolvkit command line interface
################################

The CosolvKit command line interface is the easiest method to create and simulate a cosolvent system. 
If this is your first time learning about CosolvKit, take a look at the page :ref:`Get started <get_started>`. 

CosolvKit inputs
**************************

The script `create_cosolvent_system.py` provides all the necessary tools to build a cosolvent system and optionally run an MD simulation with standard setup.
The main entry point of the script is the file `config.yaml` where all the necessary flags and command line options are specified.
A template for the `config.yaml` can be found in `cosolvkit/data/config.yaml`.


.. list-table:: CosolvKit config.yaml structure
    :widths: 25 15 70 25 15 15 15 15
    :header-rows: 1

    * - Argument
      - Type
      - Description
      - Default value
      - OPENMM
      - AMBER
      - GROMACS
      - CHARMM

    * - cosolvents
      - list
      - List of cosolvent molecules. Each entry requires ``name``, ``smiles``, ``resname``, and either ``concentration`` (mol/L) or ``copies`` (integer count).
      - no default
      -
      -
      -
      -
    * - md_engine
      - dict
      - Dict mapping engine name to list of forcefield XML files. Supported engines: [openmm, amber, gromacs, charmm]
      - no default
      -
      -
      -
      -
    * - small_molecule_ff
      - string
      - Force field for small molecules. Options: espaloma, gaff, smirnoff
      - espaloma
      -
      -
      -
      -
    * - protein_path
      - string
      - Path to the protein structure (PDB or PDBx). Set to null when using box_size.
      - null
      -
      -
      -
      -
    * - clean_protein
      - boolean
      - Flag indicating if cleaning the protein with ``PDBFixer``
      - true
      -
      -
      -
      -
    * - keep_heterogens
      - boolean
      - Flag indicating if keeping the heterogen atoms while cleaning the protein. Waters will be always kept.
      - false
      -
      -
      -
      -
    * - variants
      - dictionary
      - Dictionary of residues for which a variant is requested (different protonation state) in the form ``{"chain_id:res_id": "protonation_state"}``.
      - {}
      -
      -
      -
      -
    * - repulsive_forces
      - dict
      - Dict of pairwise repulsive forces: ``{"name": {"residueA": "BEN", "residueB": "PRP", "epsilon": 0.01, "sigma": 4.0}}``. epsilon in kcal/mol, sigma in Angstrom.
      - {}
      - ✔️
      - ✖️
      - ✖️
      - ✖️
    * - solvent_smiles
      - string
      - SMILES string of the solvent to use.
      - H2O
      -
      -
      -
      -
    * - solvent_copies
      - integer
      - If specified, the box won't be filled up with solvent, but will have the exact number of solvent molecules specified.
      - null
      -
      -
      -
      -
    * - positive_ion
      - string
      - Ion type for positive charge neutralisation.
      - Na+
      -
      -
      -
      -
    * - negative_ion
      - string
      - Ion type for negative charge neutralisation.
      - Cl-
      -
      -
      -
      -
    * - padding
      - float
      - Padding around the protein in Angstrom.
      - 10.0
      -
      -
      -
      -
    * - box_size
      - float
      - Box edge length in Angstrom. Required when no protein_path is given.
      - null
      -
      -
      -
      -
    * - ligands
      - dict
      - Dict mapping ligand name to path of SDF/MOL2 file.
      - {}
      -
      -
      -
      -
    * - membrane
      - boolean
      - Flag indicating if the system has a membrane or not.
      - false
      -
      -
      -
      -
    * - lipid_type
      - string
      - If membrane is true specify the lipid to use. Supported lipids: [POPC, POPE, DLPC, DLPE, DMPC, DOPC, DPPC]
      - POPC
      -
      -
      -
      -
    * - lipid_patch_path
      - string
      - Path to a pre-equilibrated lipid patch (mutually exclusive with lipid_type).
      - null
      -
      -
      -
      -
    * - memb_cosolv_placement
      - string
      - Which side of the membrane to place the cosolvents. Options: both, outside, inside
      - both
      -
      -
      -
      -
    * - waters_to_keep
      - list
      - List of residue indices of waters to preserve in membrane systems.
      - []
      -
      -
      -
      -
    * - output_dir
      - string
      - Path to the output directory for results.
      - no default
      -
      -
      -
      -


CosolvKit can be run with and without a protein receptor. Variants for protonation states can be specified as a YAML dictionary, and custom repulsive forces can be defined between specific cosolvent pairs (OpenMM only).

Post-processing pipeline
************************
The script `post_simulaiton_processing.py` takes care of analysing the MD simulation trajectories and produces RDF plots as well as densities analysis as PyMol sessions.
To access help message type:

.. code-block:: bash

    $ post_simulation_processing.py --help

The script is based on the `Report class` and the following functions:

    - log_file: is the statistics.csv or whatever log_file produced during the simulation. At least Volume, Temperature and Pot_e should be reported on this log file.
    - traj_file: trajectory file
    - top_file: topology file
    - cosolvents_file: json file describing the cosolvents

    generate_report():
        - out_path: where to save the results. 3 folders will be created:
            - report
                - autocorrelation
                - rdf
    generate_density_maps():
        - out_path: where to save the results.
        - analysis_selection_string: selection string of cosolvents you want to analyse. This follows MDAnalysis selection strings style. If no selection string, one density file for each cosolvent will be created.

    generate_pymol_report()
        - selection_string: important residues to select and show in the PyMol session.

.. figure:: img/rdf_BEN_C1x.png
   :alt: RDF plot example

   Example of an RDF plot generated with the post-processing pipeline.

.. figure:: img/simulation_statistics.png
   :scale: 50 %
   :alt: simulation statistics
   
   Example of a statistics plot generated with the post-processing pipeline.

Outputs
********************
CosolvKit generates topology and positions files that will be used to run the MD simulation, the output format is decided by the field `md_format` in the config file.

Access help message
**********************

.. code-block:: bash

    $ create_cosolvent_system.py --help