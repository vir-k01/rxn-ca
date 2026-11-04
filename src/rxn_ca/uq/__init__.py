"""Post-hoc uncertainty quantification for rxn-ca phase traces (hybrid method).

The question: if every reaction energy in the library carries an error delta_r ~ N(0, sigma), how much do the
simulated phase traces move? Running hundreds of perturbed simulations per condition is too expensive, so the
hybrid method does this instead:

1. ``perturb``: draw the energy errors and re-score the library with the real scorer for every sample.
2. ``events``: run a few unperturbed (baseline) simulations with full history and record, per sweep, how many
   times each reaction was chosen and what volume change it caused.
3. ``reweight``: the cheap estimator. Reweight the recorded events by how much each sample changes the
   reaction's choice probability, and replay the volume changes (with mass action and a limiting-reagent
   step that never overdraws a phase). This gives a cheap trace for every sample in a fraction of a second.
4. ``hybrid``: pick a few anchor samples at quantiles of the cheap ensemble, run them as real perturbed
   simulations, and use them to correct the width of the cheap ensemble (per phase, per sweep). Then smooth
   the ensemble statistics in time.

The calibration is per condition: it does not transfer between recipes, scorers or energy models.

For jobflow, use rxn_ca.computing.jobs.HybridUQMaker (a Flow that runs the simulations with MultiRxnCAMaker).
"""

from .perturb import (
    DEFAULT_SIGMA,
    PerturbedScores,
    canonical_key,
    perturbed_library,
    perturbed_scores,
    sample_deltas,
)
from .events import EventRecord, extract_events
from .reweight import (
    forward_lr,
    forward_lr_batch,
    mass_fractions,
    reaction_groups,
    solid_reactant_lists,
    weight_matrices,
    weight_matrix,
)
from .hybrid import (
    HybridResult,
    calibrate_traces,
    cheap_ensemble,
    observed_trace,
    run_anchor,
    run_baselines,
    run_hybrid_uq,
    select_anchors,
    summarize,
)

__all__ = [
    "DEFAULT_SIGMA",
    "PerturbedScores",
    "canonical_key",
    "perturbed_library",
    "perturbed_scores",
    "sample_deltas",
    "EventRecord",
    "extract_events",
    "forward_lr",
    "forward_lr_batch",
    "mass_fractions",
    "reaction_groups",
    "solid_reactant_lists",
    "weight_matrices",
    "weight_matrix",
    "HybridResult",
    "calibrate_traces",
    "cheap_ensemble",
    "observed_trace",
    "run_anchor",
    "run_baselines",
    "run_hybrid_uq",
    "select_anchors",
    "summarize",
]
