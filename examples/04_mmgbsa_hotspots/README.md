# 04 — Per-hotspot MMGBSA via the AutoPath integration

Takes the hotspots from [`../03_analysis`](../03_analysis) and asks a different question
of them: *which probe molecule actually sat in this pocket, and what is its interaction
energy?* CosolvKit records the occupancy, builds a trajectory from the frames where that
molecule was bound, hands it to AutoPath's MMPBSA wrapper, and folds the answer — total
plus per-residue decomposition — back onto the `Hotspot` and its `PocketResidue`s.
The same occupancy also feeds a cheaper **LIE** leg (see below), which runs inline in
about a second per molecule.

This stage is **CPU-only** and runs in the **`autopath`** env, which is where both
CosolvKit and the `autopath` package live.

The examples below invoke the CLI as `python -m cosolvkit.cli.refine_hotspots`, which
works from any install. The shorter `refine_hotspots` console script is equivalent but
only exists after the package has been (re)installed since that entry point was added:

```bash
pip install -e .        # from the repo root, once
refine_hotspots --help  # then the short form works too
```

## Prerequisites

Example 03 must have run, because this reads its hotspot checkpoint:

- `../03_analysis/results/merged/hotspot_checkpoints/` — detected hotspots
- `../03_analysis/results/merged/map_agfe_*.dx` — used to rank binding sites the same
  way `binding_sites.csv` does
- `../03_analysis/analysis.yaml` — reused as-is; it already names the trajectories,
  the **prmtop** topologies and the probe resnames

## Step 1 — Record which probe molecules occupied which hotspot

```bash
micromamba activate autopath

python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results \
                --stride 5 \
                --annotate-only
```

This scans every trajectory in the config and writes, per hotspot, one record per
(source file, probe molecule): the frames that molecule spent inside the blob, plus the
protein residues lining the pocket and which molecule contacted each. It re-saves the
checkpoint in place and writes `results/hotspot_occupancy.csv`.

`--stride 5` samples every 5th frame. Annotation is the slow step — raise the stride
while exploring, drop it to 1 for a final run. Re-running is safe: a second pass
replaces the records rather than doubling them.

Read `results/hotspot_occupancy.csv` before going further. A hotspot with
`n_frames_bound` in the single digits will give an MMGBSA number built on almost no
sampling.

## Step 2 — Generate the MMGBSA jobs

```bash
python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results \
                --stride 5 \
                --target binding_sites --top-n 3 \
                --mode mmgbsa \
                --mmgbsa-n-frames 20
```

For each of the top 3 binding sites this writes `results/bs_<id>/`:

| Path | What it is |
|------|-----------|
| `pose.pdb` | protein + the one bound probe molecule, from its best-occupied frame |
| `manifest.json` | provenance: source file, frame, probe resid, pocket residues |
| `mmgbsa/<PROBE><resid>/frames.dcd` | the selected frames, and only those |
| `mmgbsa/<PROBE><resid>/frames.json` | which source file and original frame each one came from |
| `mmgbsa/<PROBE><resid>/mmgbsa.in` | the MMPBSA input |
| `run_autopath.py` | calls `ProteinLigandAnalyzer.prepare_mmgbsa_batch` |
| `job.slurm` | qfile for the above |

`--target hotspots` refines individual per-probe hotspots instead of cross-probe
pockets; the directories are then named `hs_<PROBE>_<id>`.

## Step 3 — Run it, in two stages

MMGBSA here is **two SLURM jobs, not one** — that is AutoPath's design, not an
oversight.

```bash
# Stage 1: writes the MMPBSA inputs and a second qfile. Fast (seconds).
bash results/submit_all.sh

# Stage 2: the actual MMGBSA. Run once stage 1 has finished, per target.
cd results/bs_1 && ./run_mmgbsa_batch.sh
```

Stage 1 produces `results/bs_1/qfiles_mmgbsa/*.q` and `run_mmgbsa_batch.sh`; nothing is
computed until you submit that. Stage 2 writes `FINAL_RESULTS_mmpbsa.dat` and
`FINAL_DECOMP_mmpbsa.dat` next to `frames.dcd`.

## Step 4 — Collect the results back onto the hotspots

```bash
python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results \
                --collect
```

This parses every finished job, attaches it to the checkpoint, and writes two tables:

| Path | One row per |
|------|-------------|
| `results/mmgbsa_results.csv` | refined molecule: `delta_total`, `std_err`, `n_frames`, per-component energies |
| `results/mmgbsa_decomposition.csv` | residue per result, with **both** the stripped-complex index and the original resid |

Unfinished jobs are skipped with a log line rather than failing the pass, so you can
collect while the rest are still queued. Re-collecting replaces rather than duplicates.

In Python:

```python
from cosolvkit.analysis.sites.detect import HotspotDetector

sites = HotspotDetector.load_checkpoint(
    "../03_analysis/results/merged", ["BEN", "IMI"])

for h in sites["BEN"]:
    for m in h.mmgbsa:
        print(f"site {h.site_id}: {m.probe_resname}{m.probe_resid} "
              f"dG = {m.delta_total:.2f} +/- {m.std_err:.2f} kcal/mol (n={m.n_frames})")
    for pr in h.pocket_residues:
        if pr.mmgbsa_decomposition:
            total = pr.mmgbsa_decomposition["TOTAL"]["average"]
            print(f"   {pr.resname}{pr.resid} chain {pr.chain}: {total:+.3f}")
```

## Key knobs

| Flag | Default | Why you would change it |
|------|---------|-------------------------|
| `--stride` | 1 | Annotation cost. 5–25 while exploring. |
| `--mmgbsa-n-frames` | 20 | Frames per job. More frames, smaller standard error. |
| `--mmgbsa-n-molecules` | 1 | Emit a job per occupying copy to get a spread instead of one number. |
| `--igb` | 5 | GB model. `--radii` follows it automatically (igb 5 → `mbondi2`). |
| `--radii` | from `--igb` | Override only deliberately; a mismatched pair changes the energies without erroring. |
| `--no-decomp` | off | Skip per-residue decomposition — required to keep a transition metal (see below). |
| `--mmgbsa-source` | all | Restrict to one replica instead of pooling. |
| `--lie-n-frames` | 0 = all | Frames per LIE. LIE costs ~1 s, so the default uses every occupied frame. |
| `--lie-cutoff` | 6.0 | Angstrom shell forming the LIE environment. |
| `--lie-exclude-probes` | off | Also drop the other cosolvent copies from the LIE environment. |
| `--submit` | off | `sbatch` stage 1 immediately instead of only writing the qfiles. |

## Things that will bite you

**Some metals and decomposition are mutually exclusive.** Per-residue decomposition forces 
`gbsa=2` (ICOSA), and sander cannot atom-type transition metals — it aborts with
`bad atom type: Mn` and an otherwise empty error. So metals are stripped for
decomposition runs, with a warning naming the distance to the ligand, and the run
**refuses** if one is within 6 Å, where removing it would change the answer. Use
`--no-decomp` to keep the metal instead.

**Two residue numberings.** MMPBSA reports decomposition against the *stripped* complex,
renumbered from 1 after solvent, the other probe copies and any metals are removed.
`PocketResidue.resid` is the original topology numbering. `mmgbsa_decomposition.csv`
carries both plus the residue name, and a name mismatch raises rather than writing a
silently wrong row.

**Frame indices are per-trajectory.** A frame number is meaningless without its file,
which is why `frames.json` records both and why frames pooled from several replicas are
each read from their own trajectory.

## LIE instead — cheaper, and it survives metals

`--mode lie` scores the same bound molecule with Linear Interaction Energy via
`ProteinLigandAnalyzer.compute_LIE` (pytraj) instead of MMPBSA. It runs **inline**: no
qfile, no `submit_all.sh`, no second stage, no `--collect`. One command, and the numbers
are on the hotspot when it returns.

```bash
python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results_lie \
                --stride 5 \
                --target binding_sites --top-n 3 \
                --mode lie
```

Measured on this example: ~1 s per molecule against ~40 s plus a two-stage SLURM
handoff for MMGBSA. Because it is that cheap, `--lie-n-frames` defaults to **every**
frame the molecule occupied the site rather than a 20-frame sample.

Outputs `results_lie/lie_results.csv` (one row per refined molecule) plus, per target,
`<tag>/lie/<PROBE><resid>/{frames.dcd, LIE_results.csv, LIE_components.png}`. On the
objects it lands as `hotspot.lie` (a list of `LieResult`) and `lie_total` in the flat
row.

### What LIE is, and is not here

**It is not a ΔG.** `compute_LIE` returns the raw ligand–environment EELEC and VDW terms
in kcal/mol, with **no α/β coefficients applied** and no explicit desolvation leg. The
field is `total`, not `delta_total`, to keep that straight. Use it to rank, not to
predict affinity.

**Water stays in the environment.** The environment is every residue within
`--lie-cutoff` of the ligand, solvent included — that is how the protocol is normally
used, and the solvent term is what makes the α/β form approximate binding rather than a
bare contact energy. Only counter-ions are excluded by default.

**So LIE and MMGBSA are not interchangeable.** LIE scores a cutoff shell including
water; MMGBSA scores the whole receptor and subtracts a solvation leg. On this example
the same three molecules gave:

| target | molecule | MMGBSA ΔTOTAL | LIE total |
|--------|----------|---------------|-----------|
| bs_16  | IMI 807  | −8.04 | −19.19 ± 3.13 |
| bs_14  | BEN 810  | −6.88 | −10.32 ± 1.36 |
| bs_28  | BEN 806  | −5.91 | −10.09 ± 1.41 |

LIE is roughly twice as negative, and the ranking happened to agree on all three — three
points is weak evidence, but it is the right sign for using LIE as a pre-screen and
MMGBSA on the survivors.

**LIE works where decomposition refuses.** It needs no `ante-MMPBSA`, no sander and no
ICOSA surface area, so the transition-metal problem above simply does not arise: LIE runs
on a metalloprotein with the metal in place.

**Other probe copies are kept by default.** `--lie-exclude-probes` drops them. On both
systems tested this was a *measured no-op* — no second probe was ever inside the 6 Å
shell — but that is a property of how the box was built, not physics: `build_config_BEN.yaml`
applies a BEN–BEN repulsive force at σ = 4 Å (`CosolventSystem.add_repulsive_forces`).
Without that repulsion probes can aggregate, and then the flag matters.

# TODO this is to be fully implemented 

## Steered MD instead
`--mode smd` generates an AutoPath steered-MD job from the same pose — pulling the probe
out of the pocket rather than scoring it in place. That leg is GPU MD and might take longer, so the generated qfile requests a GPU accordingly. `--mode both` does both.

## Where this sits in the workflow

```
01 build              ->  system.prmtop + system.pdb + system.xml
02 equil + production ->  MD/MD_6E22_benzene_rep1_aligned.dcd
03 analysis           ->  results/merged/hotspot_checkpoints/
04 annotate           ->  results/hotspot_occupancy.csv          (which molecule, which frames)
04 generate           ->  results/bs_*/{pose.pdb, mmgbsa/frames.dcd, job.slurm}
04 run (2 stages)     ->  FINAL_RESULTS_mmpbsa.dat, FINAL_DECOMP_mmpbsa.dat
04 collect            ->  results/mmgbsa_{results,decomposition}.csv
                          + hotspot.mmgbsa and pocket_residue.mmgbsa_decomposition
```
