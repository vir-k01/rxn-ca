"""Hybrid UQ: cheap ensemble + a few real anchor runs that correct its width.

Steps for one condition (recipe + library + energy samples):

1. Baselines. A few unperturbed full-history runs, reduced to EventRecords.
2. Cheap ensemble. Every energy sample through the cheap estimator (reweight.forward_lr_batch), as mass-fraction
   traces averaged over the baselines. bl is the delta = 0 trace.
3. Gate and anchors. If the target's final cheap fraction barely moves (std < gate_sd), there is nothing to
   calibrate. Otherwise take the samples at the 2/25/75/98 % quantiles of the target's final cheap fraction
   and run each as a real perturbed simulation (anchor_reps times).
4. Width calibration, per phase and sweep, separately below and above the baseline:
       cal(t, p) = bl(t, p) + s(t, p) * (cheap(t, p) - bl(t, p))
   s is the zero-intercept least-squares slope of (real - bl) on (cheap - bl) through the anchors, after
   smoothing both over s_win sweeps. s = 1 where the anchors show no response (|cheap - bl| below resp_tol).
   Slopes are clipped to [0, 3], then smoothed over s_win sweeps. Samples are renormalized per sweep.
   The map only stretches or shrinks deviations from the baseline: it keeps delta = 0 on the baseline and
   keeps each sample's rank.
5. Summary. Mean, std and 5/95 % quantiles over samples, smoothed in time with a Savitzky-Golay filter
   (sg_win sweeps, order sg_order). The window is shorter than a ramp stage, so real temperature-step kinks
   survive.

The calibration belongs to one condition. It does not transfer to other recipes, scorers or energy models,
and it does not transfer to coherent shifts of one phase's energy.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from .events import EventRecord, extract_events, solid_phases, sweep_temperatures
from .perturb import PerturbedScores, perturbed_library
from .reweight import (forward_lr_batch, mass_fractions, reaction_groups, solid_reactant_lists,
                       weight_matrices)

QUANTILES = (2, 25, 75, 98)
GATE_SD = 0.02
RESP_TOL = 0.02
S_WIN = 21
SG_WIN, SG_ORD = 11, 2


def _single(recipe: ReactionRecipe, name: str) -> ReactionRecipe:
    return dataclasses.replace(recipe, num_realizations=1, name=name)


def run_baselines(recipe: ReactionRecipe, library: ReactionLibrary, n_reps: int = 5,
                  existing: Sequence[EventRecord] = ()) -> List[EventRecord]:
    """Unperturbed full-history runs, reduced to EventRecords. `existing` records are kept and counted."""
    from ..utilities.single_sim import run_single_sim

    evs = list(existing)
    n_rxn = len(library.get_rxns_at_temp(library.temps[0]).reactions)
    while len(evs) < n_reps:
        doc = run_single_sim(_single(recipe, f"{recipe.name}__baseline{len(evs)}"), reaction_lib=library,
                             phase_set=library.phases)
        evs.append(extract_events(doc.results[0], recipe.heating_schedule, library.phases, n_rxn))
    return evs


def cheap_ensemble(events: Sequence[EventRecord], scores: PerturbedScores, library: ReactionLibrary):
    """Cheap mass-fraction traces for every sample, averaged over the baselines.

    Returns (mf [S, n_sweeps+1, P], bl [n_sweeps+1, P], stress [S], phases).
    """
    phases = list(events[0].phases)
    groups = reaction_groups(library)
    solid = solid_reactant_lists(library)
    mv = library.phases.volumes
    Ab = weight_matrices(scores, groups)
    A0 = {int(T): np.ones((1, scores.s0.shape[1])) for T in scores.temps}
    mf = np.zeros((scores.n_samples,) + events[0].vol.shape)
    bl = np.zeros(events[0].vol.shape)
    stress = np.zeros(scores.n_samples)
    for ev in events:
        if list(ev.phases) != phases:
            raise ValueError("baseline records disagree on phase order")
        v, st = forward_lr_batch(ev, Ab, solid)
        mf += mass_fractions(v, phases, mv)
        stress += st
        v0, _ = forward_lr_batch(ev, A0, solid)
        bl += mass_fractions(v0[0], phases, mv)
    n = len(events)
    return mf / n, bl / n, stress / n, phases


def select_anchors(final_target: np.ndarray, quantiles: Sequence[float] = QUANTILES,
                   gate_sd: float = GATE_SD) -> List[int]:
    """Sample ids at the given quantiles of the target's final cheap fraction; [] if it barely moves."""
    final_target = np.asarray(final_target, dtype=float)
    if final_target.std() < gate_sd:
        return []
    order = np.argsort(final_target)
    n = len(final_target)
    return sorted({int(order[int(q / 100 * (n - 1))]) for q in quantiles})


def run_anchor(recipe: ReactionRecipe, library: ReactionLibrary, scores: PerturbedScores, k: int,
               phases: Optional[Sequence[str]] = None, rep: int = 0) -> np.ndarray:
    """One real simulation with sample k applied. Returns its mass-fraction trace [n_sweeps+1, P].

    Runs analysis_only with one observation per sweep, so it is as light as a normal production run.
    """
    from ..analysis.bulk_reaction_analyzer import BulkReactionAnalyzer
    from ..analysis.phase_volume_observer import VOLUMES
    from ..utilities.single_sim import run_single_sim

    phases = list(phases) if phases is not None else solid_phases(library.phases)
    lib_k = perturbed_library(library, scores, k)
    n_sites = recipe.simulation_size ** 3
    doc = run_single_sim(_single(recipe, f"{recipe.name}__anchor_k{k}_r{rep}"), reaction_lib=lib_k,
                         phase_set=lib_k.phases, analysis_only=True, compress_freq=n_sites)
    obs = doc.observed_results[0]
    steps = BulkReactionAnalyzer([obs], lib_k.phases, recipe.heating_schedule).loaded_step_idxs
    n_expected = len(sweep_temperatures(recipe.heating_schedule)) + 1
    if len(steps) != n_expected:
        raise ValueError(f"anchor run has {len(steps)} observations, expected {n_expected} (one per sweep)")
    vol = np.array([[obs.observation_at(s)[VOLUMES].get(p, 0.0) for p in phases] for s in steps])
    return mass_fractions(vol, phases, lib_k.phases.volumes)


def _smooth_t(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean over axis 0 (time), edge-padded; any trailing shape."""
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), xp], 0), 0)
    return (c[w:] - c[:-w]) / w


def calibrate_traces(mf: np.ndarray, bl: np.ndarray, anchor_ks: Sequence[int],
                     anchor_traces: Sequence[np.ndarray], resp_tol: float = RESP_TOL, s_win: int = S_WIN):
    """Width-only calibration of the cheap ensemble through the anchors.

    mf: (S, T, P) cheap traces; bl: (T, P) delta = 0 trace.
    anchor_traces[i]: real trace (T, P) of sample anchor_ks[i], mean over its repetitions.
    Returns (cal [S, T, P], s_lo [T, P], s_hi [T, P]).
    """
    s_lo = np.ones_like(bl)
    s_hi = np.ones_like(bl)
    if len(anchor_ks):
        ks = list(anchor_ks)
        x = np.stack([_smooth_t(mf[k], s_win) for k in ks]) - _smooth_t(bl, s_win)[None]
        y = np.stack([_smooth_t(np.asarray(a, float), s_win) for a in anchor_traces]) - _smooth_t(bl, s_win)[None]
        for mask, dest in ((x < 0, s_lo), (x > 0, s_hi)):
            xm, ym = np.where(mask, x, 0.0), np.where(mask, y, 0.0)
            den = (xm * xm).sum(0)
            dest[:] = np.where(den > resp_tol ** 2, np.clip((xm * ym).sum(0) / np.maximum(den, 1e-12), 0, 3), 1.0)
        s_lo, s_hi = _smooth_t(s_lo, s_win), _smooth_t(s_hi, s_win)
    dev = mf - bl[None]
    cal = np.clip(bl[None] + np.where(dev < 0, s_lo[None], s_hi[None]) * dev, 0, 1)
    cal /= np.maximum(cal.sum(2, keepdims=True), 1e-12)
    return cal, s_lo, s_hi


def summarize(cal: np.ndarray, sg_win: int = SG_WIN, sg_order: int = SG_ORD) -> Dict[str, np.ndarray]:
    """Mean, std and 5/95 % quantiles over samples (axis 0), smoothed in time. Each is (T, P)."""
    def sg(x):
        return savgol_filter(x, sg_win, sg_order, axis=0, mode="interp")
    return {"mean": np.clip(sg(cal.mean(0)), 0, 1),
            "std": np.clip(sg(cal.std(0)), 0, None),
            "q05": np.clip(sg(np.percentile(cal, 5, 0)), 0, 1),
            "q95": np.clip(sg(np.percentile(cal, 95, 0)), 0, 1)}


@dataclass
class HybridResult:
    phases: List[str]
    temps: np.ndarray            # (n_sweeps,) temperature of each sweep
    cheap: np.ndarray            # (S, T, P) cheap traces
    baseline: np.ndarray         # (T, P) delta = 0 trace
    cal: np.ndarray              # (S, T, P) calibrated traces
    stress: np.ndarray           # (S,) limiting-reagent stress per sample
    anchor_ks: List[int]
    anchor_traces: List[np.ndarray]
    s_lo: np.ndarray
    s_hi: np.ndarray
    summary: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def gated(self):
        return not self.anchor_ks

    def save(self, path, keep_samples: bool = True):
        extra = {f"summary_{k}": v.astype(np.float32) for k, v in self.summary.items()}
        if keep_samples:
            extra.update(cheap=self.cheap.astype(np.float32), cal=self.cal.astype(np.float32))
        np.savez_compressed(path, phases=np.array(self.phases), temps=self.temps,
                            baseline=self.baseline.astype(np.float32), stress=self.stress.astype(np.float32),
                            anchor_ks=np.array(self.anchor_ks, dtype=np.int32),
                            anchor_traces=np.array(self.anchor_traces, dtype=np.float32),
                            s_lo=self.s_lo.astype(np.float32), s_hi=self.s_hi.astype(np.float32), **extra)


def run_hybrid_uq(recipe: ReactionRecipe, library: ReactionLibrary, scores: PerturbedScores, target: str,
                  n_baselines: int = 5, anchor_reps: int = 2, events: Sequence[EventRecord] = (),
                  quantiles: Sequence[float] = QUANTILES, gate_sd: float = GATE_SD) -> HybridResult:
    """The whole hybrid pipeline for one condition, run serially.

    scores: PerturbedScores for this library (e.g. 500 samples). target: the phase the anchors are chosen on.
    events: baseline EventRecords already computed (more are run up to n_baselines).
    For large campaigns, run the pieces (run_baselines, run_anchor) in parallel and call cheap_ensemble,
    select_anchors, calibrate_traces and summarize on the collected outputs.
    """
    scores.check_library(library)
    evs = run_baselines(recipe, library, n_baselines, existing=events)
    mf, bl, stress, phases = cheap_ensemble(evs, scores, library)
    ks = select_anchors(mf[:, -1, phases.index(target)], quantiles, gate_sd)
    traces = [np.mean([run_anchor(recipe, library, scores, k, phases, rep) for rep in range(anchor_reps)], 0)
              for k in ks]
    cal, s_lo, s_hi = calibrate_traces(mf, bl, ks, traces)
    return HybridResult(phases, evs[0].temps, mf, bl, cal, stress, ks, traces, s_lo, s_hi, summarize(cal))
