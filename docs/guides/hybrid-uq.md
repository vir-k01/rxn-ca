# Hybrid UQ: phase traces under reaction-energy errors

This guide explains the `rxn_ca.uq` package and the `HybridUQMaker` flow: what they compute, how the code is
laid out, and how to run them. It is written so that someone (or an agent) new to the code can run it and
change it safely.

## The question

Every reaction energy in a library carries an error. Treat the error of reaction *r* as
δ<sub>r</sub> ~ N(0, σ), with σ = 15 meV/atom by default. How much do the simulated phase traces move?

The direct answer is to run hundreds of simulations, each with its library perturbed by one draw of δ. That
costs hundreds of simulations per condition. The hybrid method gets the same distribution for about
`n_baselines + 4 × anchor_reps` simulations (13 with the defaults).

## The method in four steps

1. **Energy samples.** Draw δ for every reaction, for each of `n_samples` samples (500 by default). Re-score
   the library with the real scorer object at every temperature. Any scorer works (TammanScore,
   DiffusionScorer, ...), because the scores are recomputed, not approximated.
2. **Baselines.** Run a few unperturbed simulations with full history. From each, record per sweep how many
   times each reaction was chosen, and the volume change those choices caused (an `EventRecord`).
3. **Cheap ensemble.** For each sample, reweight the recorded choices by how much the sample changes each
   reaction's choice probability, and replay the volume changes. The replay applies mass action (a reactant
   that is more or less abundant than in the baseline gets contacted more or less often). It also applies a
   limiting-reagent step, so a phase is never consumed below zero. This gives a trace per sample in a
   fraction of a second. At δ = 0 it reproduces the baseline exactly.
4. **Anchors and calibration.** The cheap ensemble gets the shape right but not the width. Pick the samples
   at the 2/25/75/98 % quantiles of the target's final cheap fraction (the anchors) and run them for real.
   Then, per phase and per sweep, separately below and above the baseline:

       cal = baseline + s × (cheap − baseline)

   The slope s comes from a zero-intercept least-squares fit of (real − baseline) on (cheap − baseline)
   through the anchors. It is 1 where the anchors show no response, clipped to [0, 3], and smoothed over
   21 sweeps. The map only stretches or shrinks deviations from the baseline: δ = 0 stays on the baseline and
   each sample keeps its rank. Finally the mean, std and 5/95 % quantiles over samples are smoothed in time
   (Savitzky-Golay, 11 sweeps, order 2, shorter than a ramp stage so real temperature-step kinks survive).

   **Gate:** if the target's final cheap fraction has a std below `gate_sd` (0.02), nothing moves enough to
   calibrate. No anchors are run, and the cheap ensemble is reported as is.

### What the calibration is and is not

- It belongs to **one condition** (recipe + library + scorer + energy model). Calibrations borrowed from
  another condition were about 3× worse in testing. Do not reuse anchors across conditions.
- It is for **independent per-reaction errors**. It does not transfer to a coherent shift of one phase's
  energy (every reaction involving that phase moving together). For that, run controlled sweeps instead.
- Only the scored reaction energy is perturbed. Phase energies, and so a DiffusionScorer's chemical-potential
  diagram, are not.

### How well it works

Benchmarked in the high-throughput campaign this code was ported from, against brute force: 696 conditions,
100 real perturbed runs each.

- Trace σ ratio (hybrid / brute): 1.00.
- Mean-trace error 0.005, against a split-half noise floor of 0.003.
- Final-yield Wasserstein-1 distance 0.020, against a floor of 0.019.
- Better than the cheap ensemble alone in 619 of 696 conditions.

The port in this package reproduces that pipeline's outputs: identical summaries, and the cheap ensemble to
float32 rounding, checked on a production condition.

## Code layout

| Path | What it holds |
|---|---|
| `rxn_ca/uq/perturb.py` | `sample_deltas` (seeded δ), `perturbed_scores` (re-scoring), `PerturbedScores` (deltas, temps, scores; npz save/load; `check_library`), `perturbed_library` (sample k applied to a library), `library_reactions` |
| `rxn_ca/uq/events.py` | `EventRecord`, `extract_events` (from one full-history realization) |
| `rxn_ca/uq/reweight.py` | The cheap estimator: `reaction_groups`, `weight_matrix(es)`, `forward_lr_batch`, `forward_lr`, `mass_fractions` |
| `rxn_ca/uq/hybrid.py` | `cheap_ensemble`, `select_anchors`, `observed_trace`, `calibrate_traces`, `summarize`, `HybridResult`, and a serial `run_hybrid_uq` |
| `rxn_ca/utilities/uq.py` | Glue for the jobs: baseline/anchor recipes, `score_energy_samples`, reading `MultiRxnCAMaker` outputs, `cheap_analysis`, `calibration` |
| `rxn_ca/computing/schemas/uq.py` | `HybridUQCheapDoc`, `HybridUQResultDoc` |
| `rxn_ca/computing/jobs.py` | `HybridUQMaker` (the flow) and its three jobs: `score_energy_samples_job`, `hybrid_uq_cheap_job`, `hybrid_uq_calibration_job` |

## Running it with jobflow

### Inputs

- `recipe`: the `ReactionRecipe` to study. Its `num_realizations` is ignored: the flow sets its own.
- `reaction_library`: the scored `ReactionLibrary` the recipe runs with.
- `target`: the phase the anchors are chosen on, usually the recipe's target product.
- Energy-sample scores, in one of two forms:
    - `scores_file`: a `PerturbedScores` npz for this exact library (see below), or
    - `base_reactions`: the rxn_network `ReactionSet` the library was scored from. The flow then scores the
      samples itself, with `recipe.get_score_class()` and `scorer_kwargs`.

### Minimal example

```python
from jobflow import run_locally
from rxn_ca.computing.jobs import HybridUQMaker

flow = HybridUQMaker(n_baselines=5, anchor_reps=2, num_workers=256).make(
    recipe, reaction_library, target="Sr2WO5", scores_file="scores.npz")
run_locally(flow, create_folders=True)      # or submit with jobflow-remote / FireWorks
```

Use `create_folders=True` (or a manager that gives each job its own directory). Each `MultiRxnCAMaker` writes
`result_doc_<i>.json` into its working directory, and the next job reads them back from there.

### The flow

| # | Job | What it does | Writes |
|---|---|---|---|
| 1 | `HybridUQ scores` (only without `scores_file`) | Scores `n_samples` energy samples, one temperature per process | `uq_scores.npz` |
| 2 | `HybridUQ baselines` (`MultiRxnCAMaker`, `MULTI_RECIPE`) | `n_baselines` unperturbed realizations over Ray, full history (`compress=False`, `compute_plots=False`) | `result_doc_0.json`, `multi_rxn_ca_result.json` |
| 3 | `HybridUQ cheap ensemble` | Event records → cheap ensemble → gate → anchor samples and their perturbed libraries | `uq_cheap.npz`; returns `HybridUQCheapDoc` |
| 4 | `HybridUQ anchors` (`MultiRxnCAMaker`, `MULTI_LIB`) | `anchor_reps` realizations of each anchor library, `analysis_only=True`, one observation per sweep | `result_doc_<i>.json` (one per anchor) |
| 5 | `HybridUQ calibration` | Width calibration and smoothed summaries | `hybrid_uq_samples.npz`, `hybrid_uq_result.json`; returns `HybridUQResultDoc` |

If the gate closes, job 4 runs zero realizations and job 5 reports the cheap ensemble.

### Maker parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_samples` | 500 | Energy samples (only when scoring in the flow) |
| `sigma` | 0.015 | Error per reaction energy, eV/atom (only when scoring in the flow) |
| `scorer_kwargs` | `{}` | Extra kwargs for the scorer class (only when scoring in the flow) |
| `n_baselines` | 5 | Unperturbed full-history realizations |
| `anchor_reps` | 2 | Realizations per anchor sample |
| `quantiles` | (2, 25, 75, 98) | Where the anchors go, in % of the target's final cheap fraction |
| `gate_sd` | 0.02 | Skip anchors below this std of the target's final cheap fraction |
| `num_workers`, `memory_per_task_gb`, `cpus_per_task` | None, None, 1 | Passed to both `MultiRxnCAMaker`s (and the scoring pool) |
| `save_samples` | True | Write every sample's traces to `hybrid_uq_samples.npz` |
| `metadata` | None | Stored on the result doc |

### Making a scores file yourself

The scoring needs the unscored reactions at each temperature and the scorer that built the library. The
library itself does not keep those. If your library came from rxn_network, pass `base_reactions` and let the
flow do it. Otherwise build the file directly:

```python
from rxn_ca.uq.perturb import perturbed_scores, sample_deltas, stoich_of

rxns_at = lambda T: [...]          # reactions at T (ComputedReaction), in library id order
scorer_at = lambda T: MyScorer(...)  # the scorer the library was built with, at T
deltas = sample_deltas([stoich_of(r) for r in rxns_at(temps[0])], n_samples=500)
scores = perturbed_scores(rxns_at, scorer_at, temps, deltas)
scores.check_library(library)      # raises if the reaction ids do not line up
scores.save("scores.npz")
```

`sample_deltas` seeds each draw from (sample number, reaction). The same sample number gives the same error
to the same reaction in every library, so samples are paired across scorers and energy models. A reaction
and its reverse get opposite errors.

## Outputs

`HybridUQResultDoc` (in the store and in `hybrid_uq_result.json`):

- `phases`, `temperature` (one entry per trace point: the initial state plus one per sweep).
- `baseline`, `mean`, `std`, `q05`, `q95`: per-phase mass-fraction traces of the solid phases. Phases that
  never reach 0.002 are left out.
- `final_per_sample`: every phase's calibrated final fraction, for every sample.
- `gated`, `anchor_ks`, `anchor_reps`, `anchor_final` (each anchor's real final fractions).
- `stress_mean`, `stress_p95`: how often the limiting-reagent step had to step in (fraction of proposed
  consumption it removed). Near 0 means the cheap model ran without strain. Large values mean the cheap
  ensemble leaned on mass balance; the anchors still correct its width.
- `samples_file`: path of `hybrid_uq_samples.npz`, which holds `cheap` and `cal` (samples × trace points ×
  phases), `baseline`, `stress`, `anchor_ks`, `anchor_traces`, the calibration slopes `s_lo` / `s_hi`, and
  `summary_<stat>`.

## Without jobflow

`rxn_ca.uq.run_hybrid_uq(recipe, library, scores, target)` runs the same steps serially in one process. For
your own parallel setup, call the pieces: `run_baselines` / `extract_events`, `cheap_ensemble`,
`select_anchors`, `run_anchor` (or `perturbed_library` + your own runner + `observed_trace`),
`calibrate_traces`, `summarize`.

## Things that will bite you

- **Baselines need every per-step diff.** No `compress_freq`, no `num_frames`, not analysis-only.
  `extract_events` checks the diff count and raises if any are missing.
- **Full-history plots are slow.** `MultiRxnCAResultDoc` plot data rebuilds every other step: about 25 min per
  realization at 12³. The flow turns it off for the baselines with `compute_plots=False`.
- **Ray worker metadata.** The Ray worker used to walk the whole history to build reaction-choice metadata
  that it then discarded (over 30 min per 12³ baseline). That line was removed on `fix/parallel-analysis-only`,
  which is merged into this branch along with the analysis-only fixes the anchor runs need.
- **Anchors need one observation per sweep.** The flow passes `compress_freq = simulation_size³`.
  `observed_trace` raises if the observation count is not one per sweep plus the initial state.
- **Reaction ids must line up.** Scores are indexed by library reaction id. `PerturbedScores.check_library`
  compares the δ = 0 scores with the library's stored scores and raises on a mismatch. The flow calls it.
- **Heating schedules** with regrind steps are not supported by the event extraction.
- **Not yet tested:** the `base_reactions` scoring path inside the flow, and a full run at production lattice
  size (12³). The flow was tested end to end at 6³, gated and ungated.
