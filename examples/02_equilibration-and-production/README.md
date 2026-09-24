# 02 — Equilibration and Production (with AutoPath)

CosolvKit builds and parametrizes the cosolvent system but **does not run MD** —
that is delegated to an external engine. This example shows how to take the files
CosolvKit produces and run **equilibration** and then **production** MD with
[AutoPath](https://github.com/forlilab/autopath), reusing the `6E22 + benzene`
system from [`../01_build_cosolvent_system`](../01_build_cosolvent_system).

AutoPath can also build the system itself, but here we **skip its build step** and
feed it the system CosolvKit already prepared.

**Inputs are read from example 01; all outputs of this phase are written here in 02**
(under `./<name>/`, default `./6E22_benzene/`), so the 01 build directory stays clean.

## Prerequisites

- The `autopath` conda/micromamba environment (provides `autopath`, OpenMM, MDAnalysis,
  mdtraj, ...). A CUDA GPU is required.
- Run example 01 first. With the OpenMM engine it writes, into
  `../01_build_cosolvent_system/6E22_benzene/`:
  - `system.pdb` — coordinates + topology
  - `system.xml` — serialized OpenMM `System` (force field already applied)

  These two files are the only inputs the equilibration phase reads (read-only).

> **Note on topology.** The OpenMM engine does not emit a `.prmtop`, so both scripts
> use `system.pdb` as the topology for trajectory analysis (mdtraj / MDAnalysis).

## Phase 1 — Equilibration

`ap_equilibration.py` loads `system.pdb` + `system.xml` from the 01 directory and runs a
JSON-driven, multi-stage minimization → warm-up → restrained-MD protocol with
`autopath.Equilibration`. Results are written into **this** directory (`./6E22_benzene/`).

The protocol [`cosolvent_equilibration.json`](cosolvent_equilibration.json) restrains
**only the protein** (backbone and side chains, with progressively released force
constants). Cosolvents and water are left **unrestrained** so they equilibrate freely
around the protein — the key difference from AutoPath's stock protein–ligand protocol,
which restrains a bound `resname UNK` ligand that a cosolvent box does not have.

```bash
# On a GPU node, in the autopath env:
micromamba activate autopath
python ap_equilibration.py --system-dir ../01_build_cosolvent_system/6E22_benzene
# or submit to SLURM:
sbatch run_equilibration.q
```

`--system-dir` is the read-only input (01); outputs default to `./<name>` here in 02.
Use `--out-dir` to choose a different output location (production must use the same one).

Outputs (in `./6E22_benzene/equilibration/`):

| File | Purpose |
|------|---------|
| `system_equil_6E22_benzene.xml` | equilibrated OpenMM `System` — **input to production** |
| `checkpoint_equil_6E22_benzene.chk` | equilibrated state (positions + velocities + box) — **input to production** |
| `equilibration_6E22_benzene_aligned.dcd` | wrapped/aligned equilibration trajectory |
| `RMSD_6E22_benzene.csv` + plots | protein RMSD over equilibration |

## Phase 2 — Production (vanilla MD)

`ap_production.py` continues from the equilibration checkpoint and runs unbiased
conventional MD (100 ns, 4 fs, 300 K, NPT) with `autopath.VanillaMD`. No restraints
are applied. It reads `system.pdb` (topology) from the 01 directory and the
equilibration handoff (`system_equil_<name>.xml`, `checkpoint_equil_<name>.chk`) from
this directory's `./6E22_benzene/equilibration/`.

```bash
python ap_production.py --system-dir ../01_build_cosolvent_system/6E22_benzene --replica rep1
# or:
sbatch run_production.q
```

Run it once per replica (`--replica rep1`, `rep2`, ...; the checkpoint's velocities
are resampled each time via `restart_velocities=True`).

Outputs (in `./6E22_benzene/MD/`):

| File | Purpose |
|------|---------|
| `MD_6E22_benzene_rep1_aligned.dcd` | wrapped/aligned production trajectory |
| `MD_6E22_benzene_rep1_checkpoint.chk` / `.xml` | final state / system |
| `6E22_benzene_rep1_rmsd.csv` + plots | protein RMSD over production |

The aligned production trajectory + the 01 `system.pdb` are what you feed into CosolvKit's
analysis (`analyze_cosolvent_simulation`) to compute AGFE density maps and hotspots.

The trajectory is superposed onto the backbone of the 01 `system.pdb`, not onto its own
first frame. The build keeps the protein at the input PDB's coordinates, so every system
built from the same PDB, whatever its box, lands in one frame, and density maps from
different replicas or probes can be merged directly. For builds from different input
PDBs, pass a shared structure with `--align-reference` (same backbone atoms).

## Handoff at a glance

```
examples/01_build_cosolvent_system   ->  01_build_cosolvent_system/6E22_benzene/system.pdb + system.xml
        (create_cosolvent_system)                         |  (read-only input)
                                                          v
ap_equilibration.py  (Equilibration)  ->  02_.../6E22_benzene/equilibration/
                                              system_equil_*.xml + checkpoint_equil_*.chk
                                                          |
ap_production.py     (VanillaMD)       ->  02_.../6E22_benzene/MD/MD_*_aligned.dcd
                                                          |
analyze_cosolvent_simulation           ->  AGFE maps + hotspots
```
