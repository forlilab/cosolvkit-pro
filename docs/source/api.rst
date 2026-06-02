:: _api:

cosolvkit API
###############################

The CosolvKit API allows for a more advanced, flexible use of CosolvKit where the user can create their own scripts. 

Config class and config.yaml
###############################

CosolvKit implements a `Config` class to handle the list of setup options.
In the data folder a template of the `config.yaml` file used to setup the system building is provided.
It is also possible to overwrite some of the options from the API:

.. code-block:: python

    from cosolvkit.config import Config
    config = Config.from_config('config.yaml')

    # Modify options before building
    config.padding = 12.0
    config.output_dir = 'my_results'

Creating CosolventMolecules in Python
######################################

It is possible to define cosolvent molecules directly in Python instead of through the YAML config.

.. code-block:: python

    from cosolvkit.cosolvent_system import CosolventMolecule 

    cosolvent_molecules = list()
    cosolvent_molecules.append(CosolventMolecule(name="benzene",
                                                 smiles="C1=CC=CC=C1",
                                                 resname="BEN",
                                                 concentration=0.25))
    cosolvent_molecules.append(CosolventMolecule(name="methanol",
                                                 smiles="CO",
                                                 # MET is reserved for residues
                                                 resname="MNL",
                                                 copies=58))



Building a CosolventSystem
##########################

Thanks to the flexible API CosolvKit allows the user to instantiate custom CosolventSystem classes with user prepared proteins.
Cosolvents can be loaded from a YAML config file or defined directly as a list of dicts:

.. code-block:: python

    from cosolvkit.config import Config
    from cosolvkit.cosolvent_system import CosolventSystem
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    config = Config.from_config('config.yaml')

    cosolvent_system = CosolventSystem(cosolvents=config.cosolvents,
                                       forcefields=config.md_engine,
                                       modeller=modeller)
    cosolvent_system.build()

Building a CosolventMembraneSystem
##################################

A cosolvent system with membrane can be easily built:

.. code-block:: python

    from cosolvkit.config import Config
    from cosolvkit.cosolvent_system import CosolventMembraneSystem
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    config = Config.from_config('config.yaml')

    cosolvent_system = CosolventMembraneSystem(cosolvents=config.cosolvents,
                                               forcefields=config.md_engine,
                                               modeller=modeller,
                                               lipid_type="POPC")

    # Or pass a pre-equilibrated lipid patch
    cosolvent_system = CosolventMembraneSystem(cosolvents=config.cosolvents,
                                               forcefields=config.md_engine,
                                               modeller=modeller,
                                               lipid_patch_path="path/to/the/patch")

    cosolvent_system.build()


Adding repulsive forces in case of aggregation events
#####################################################

Aggregation events can be common for some types of cosolvents, if in doubt, we suggest to run a simulation without custom repulsive forces and inspect the RDF profiles (please refer to the original paper for more details).  
If aggregation is observed, CosolvKit offers the possibility to add a custom repulsive force between specified residues.

.. code-block:: python

    from cosolvkit.config import Config
    from cosolvkit.cosolvent_system import CosolventSystem, CosolventMolecule
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    cosolvent_molecules = [
        CosolventMolecule(name="benzene", smiles="C1=CC=CC=C1",
                          resname="BEN", concentration=0.25)
    ]

    config = Config.from_config('config.yaml')

    cosolvent_system = CosolventSystem(cosolvents=cosolvent_molecules,
                                       forcefields=config.md_engine,
                                       modeller=modeller)
    cosolvent_system.build()
    cosolvent_system.add_repulsive_forces({"BEN_BEN": {"residueA": "BEN", "residueB": "BEN",
                                                        "epsilon": 0.01, "sigma": 4.0}})


Use custom solvent
##################

CosolvKit offers the possibility of using solvents different from water. In case of water the solvation is done by OpenMM, while for custom cosolvents CosolvKit exploits the same method used to place cosolvent molecules to place solvent molecules (if filling the box with solvent can be pretty slow).
This feature of CosolvKit is meant to offer flexibility for different advanced tasks.  

The solvent can be specified as SMILES string and the number of molecules requested can be specified optionally.

.. code-block:: python

    #... Previous code to create cosolvent system
    cosolvent_system.build(solvent_smiles="CO", n_solvent_molecules=350)


Saving topologies and the system
################################

Once the cosolvent system is created and parametrized, it has to be saved for the next steps (likely MD simulation).
Depending on what MD engine was selected the format of the topology files can change.  

.. code-block:: python

    #... Previous code to create and parametrize the cosolvent system
    cosolvent_system.save_topology(topology=cosolvent_system.modeller.topology,
                                   positions=cosolvent_system.modeller.positions,
                                   system=cosolvent_system.system,
                                   # Gather the md_format from the config file
                                   simulation_format=config.md_format,
                                   forcefield=cosolvent_system.forcefield)


Run MD simulations with CosolvKit
#################################

CosolvKit offers a general and standard protocol to run MD simulations that can be used for the majority of the use cases.  
The flags `run_cosolvent_system` and `run_md` in the `Config` class take care of building the cosolvent system and using the standard MD protocol to run a simulation.

.. code-block:: python

    from cosolvkit.simulation import run_simulation

    if config.md_format.upper() != "OPENMM":
            # Change the next two lines depending on the simulation_format you chose
            topo = os.path.join(config.output, "system.prmtop")
            pos = os.path.join(config.output, "system.rst7")
            # This is for openmm
            pdb = None
            system = None
        else:
            topo = None
            pos = None
            # This is for openmm
            pdb = os.path.join(config.output, "system.pdb")
            system = os.path.join(config.output, "system.xml")
        
        if config.md_format.upper() == "OPENMM":
            print(f"Starting MD simulation from the files: {pdb}, {system}")
        else:
            print(f"Starting MD simulation from the files: {topo}, {pos}")
        
        run_simulation(
                        simulation_format = config.md_format,
                        topology = topo,
                        positions = pos,
                        pdb = pdb,
                        system = system,
                        warming_steps = 100000,
                        simulation_steps = 6250000, # 25ns
                        results_path = config.output, # This should be the name of system being simulated
                        seed=None
        )

Post processing analysis
########################

CosolvKit offers a very basic package to analyze the results of the MD simulations.  
In particualr, Radial Distribution Functions (RDFs) of the cosolvent atoms and waters are generated with the respective autocorrelation functions.  
Furthermore, densities of the specified cosolvent molecules are depicted during the simulation and saved as a PyMol session for further analysis (check the pre-print for more examples of the use of cosolvent densities).  

.. code-block:: python

    # The whole analysis module relies on the Report class
    from cosolvkit.analysis import Report

    report = Report(log_file="statistics.csv",
                    traj_file="trajectory.dcd",
                    top_file="system.prmtop",
                    cosolvent_names=["BEN"])
    # Generate RDF and autocorrelation plots
    report.generate_report(out_path="results")

    # Generate density files
    # analysis_selection_string is a string in MDAnalysis format
    # to select specific cosolvents for the densities
    report.generate_density_maps(out_path="densities",
                                 analysis_selection_string="")

    report.generate_pymol_reports(topology="system.prmtop",
                                  trajectory="trajectory.dcd",
                                  density_files=["map_density_BEN.dx"],
                                  # It's possible to specify PyMol selection string to highlight
                                  # specific residues for that particular density
                                  selection_string="",
                                  out_path="results")                

