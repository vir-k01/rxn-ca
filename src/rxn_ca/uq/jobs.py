"""jobflow Maker for the hybrid UQ of one recipe + reaction library.

The job runs every step of rxn_ca.uq.hybrid on one node:
  1. energy samples: scores for n_samples seeded energy-error samples (or precomputed scores)
  2. baselines: n_baselines unperturbed full-history runs, reduced to event records
  3. cheap ensemble for every sample, gate, quantile anchors
  4. anchors: each anchor sample run for real anchor_reps times (analysis-only)
  5. width calibration and smoothed summaries
Steps 1, 2 and 4 run in a fork pool of num_workers processes (one simulation per process).

Outputs: a HybridUQResultDoc (summaries, per-sample finals, anchor info; small enough for the JobStore),
plus hybrid_uq_samples.npz in the launch directory with the full per-sample cheap and calibrated traces.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from jobflow import Maker, job

from ..computing.schemas.base_schema import BaseSchema
from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from .events import EventRecord, extract_events, sweep_temperatures
from .hybrid import (GATE_SD, QUANTILES, HybridResult, _single, calibrate_traces, cheap_ensemble, run_anchor,
                     select_anchors, summarize)
from .perturb import (DEFAULT_SIGMA, PerturbedScores, library_reactions, perturbed_scores, sample_deltas,
                      stoich_of)

# Worker state, set in the parent before the pool forks (so nothing large is pickled per task).
_G: dict = {}


def _baseline_task(i: int) -> EventRecord:
    from ..utilities.single_sim import run_single_sim

    recipe, lib = _G["recipe"], _G["library"]
    n_rxn = len(lib.get_rxns_at_temp(lib.temps[0]).reactions)
    doc = run_single_sim(_single(recipe, f"{recipe.name}__baseline{i}"), reaction_lib=lib, phase_set=lib.phases)
    return extract_events(doc.results[0], recipe.heating_schedule, lib.phases, n_rxn)


def _anchor_task(args) -> np.ndarray:
    k, rep = args
    return run_anchor(_G["recipe"], _G["library"], _G["scores"], k, _G["phases"], rep)


def _score_task(T: int) -> PerturbedScores:
    rset, ps, cls, kw = _G["rxn_set"], _G["library"].phases, _G["score_class"], _G["scorer_kwargs"]
    return perturbed_scores(lambda t: library_reactions(rset, t, ps),
                            lambda t: cls(temp=t, phase_set=ps, **kw), [T], _G["deltas"])


def _map(fn, tasks, n_workers):
    tasks = list(tasks)
    if n_workers == 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("fork").Pool(min(n_workers, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def _per_phase(arr: np.ndarray, phases: Sequence[str], keep: Sequence[int]) -> Dict[str, List[float]]:
    return {phases[i]: np.round(arr[:, i], 5).tolist() for i in keep}


@dataclass
class HybridUQResultDoc(BaseSchema):
    """Hybrid UQ result for one recipe + library.

    Traces are mass fractions of the solid phases, one value per sweep plus the initial state
    (temperature[i] is the temperature of trace point i). Phases that never reach 0.002 are left out of the
    trace fields. final_per_sample holds every phase's calibrated final fraction for every energy sample.
    """
    recipe: ReactionRecipe
    target: str
    n_samples: int
    sigma_eV_per_atom: Optional[float]
    n_baselines: int
    phases: List[str]
    temperature: List[int]
    baseline: Dict[str, List[float]]
    mean: Dict[str, List[float]]
    std: Dict[str, List[float]]
    q05: Dict[str, List[float]]
    q95: Dict[str, List[float]]
    final_per_sample: Dict[str, List[float]]
    gated: bool
    anchor_ks: List[int]
    anchor_reps: int
    anchor_final: Dict[str, List[float]]
    stress_mean: float
    stress_p95: float
    samples_file: Optional[str] = None
    metadata: dict = None
    run_dir: str = None


@dataclass
class HybridUQMaker(Maker):
    """Hybrid UQ of one recipe + reaction library: phase-trace distributions under reaction-energy errors.

    Parameters
    ----------
    n_samples (int): number of energy-error samples, used when scores are computed here. Default 500.
    sigma (float): energy error per reaction, eV/atom, used when scores are computed here. Default 0.015.
    n_baselines (int): unperturbed full-history runs for the cheap estimator. Default 5.
    anchor_reps (int): real runs per anchor sample. Default 2.
    quantiles (tuple): quantiles (%) of the target's final cheap fraction where anchors are placed.
        Default (2, 25, 75, 98).
    gate_sd (float): if the target's final cheap fraction has a std below this, no anchors are run and the
        cheap ensemble is reported as is. Default 0.02.
    scorer_kwargs (dict): extra kwargs for the recipe's scorer class, used when scores are computed here.
    num_workers (int): processes for the simulations and the scoring. Default None = all cores.
    save_samples (bool): write hybrid_uq_samples.npz (all per-sample traces) to the launch directory.
    metadata (dict): stored on the result doc.
    """

    name: str = "HybridUQ Maker"
    n_samples: int = 500
    sigma: float = DEFAULT_SIGMA
    n_baselines: int = 5
    anchor_reps: int = 2
    quantiles: tuple = QUANTILES
    gate_sd: float = GATE_SD
    scorer_kwargs: dict = field(default_factory=dict)
    num_workers: int = None
    save_samples: bool = True
    metadata: dict = field(default=None)

    @job
    def make(self, recipe: ReactionRecipe, reaction_library: ReactionLibrary, target: str,
             scores: Union[PerturbedScores, str, None] = None, base_reactions=None) -> HybridUQResultDoc:
        """
        recipe: the recipe to run (num_realizations is ignored; each run is one realization).
        reaction_library: the scored library the recipe runs with.
        target: phase the anchors are chosen on (usually the recipe's target product).
        scores: precomputed PerturbedScores for this library, or a path to its .npz. If None, scores are
            computed here from base_reactions with the recipe's scorer class.
        base_reactions: the rxn_network ReactionSet the library was scored from (needed when scores is None).
        """
        n_workers = self.num_workers or mp.cpu_count()
        lib = reaction_library
        _G.clear()
        _G.update(recipe=recipe, library=lib)

        # 1. energy samples
        sigma = self.sigma
        if isinstance(scores, str):
            scores = PerturbedScores.load(scores)
            sigma = None
        elif scores is None:
            if base_reactions is None:
                raise ValueError("give either scores or base_reactions")
            temps = sorted(int(t) for t in lib.temps)
            stoichs = [stoich_of(r) for r in library_reactions(base_reactions, temps[0], lib.phases)]
            deltas = sample_deltas(stoichs, self.n_samples, self.sigma)
            _G.update(rxn_set=base_reactions, score_class=recipe.get_score_class(),
                      scorer_kwargs=self.scorer_kwargs, deltas=deltas)
            print(f"scoring {self.n_samples} energy samples at {len(temps)} temperatures")
            parts = _map(_score_task, temps, n_workers)
            scores = PerturbedScores(deltas, np.array(temps, dtype=np.int32),
                                     np.concatenate([p.s for p in parts], 1), np.concatenate([p.s0 for p in parts], 0))
        else:
            sigma = None
        scores.check_library(lib)
        _G["scores"] = scores

        # 2. baselines
        print(f"running {self.n_baselines} baseline simulations (full history)")
        evs = _map(_baseline_task, range(self.n_baselines), n_workers)

        # 3. cheap ensemble, gate, anchors
        mf, bl, stress, phases = cheap_ensemble(evs, scores, lib)
        ti = phases.index(target)
        ks = select_anchors(mf[:, -1, ti], self.quantiles, self.gate_sd)
        print(f"cheap ensemble: {target} final {mf[:, -1, ti].mean():.3f} ± {mf[:, -1, ti].std():.3f}; "
              f"anchors {ks if ks else 'none (gated)'}")

        # 4. anchors
        traces = []
        if ks:
            _G["phases"] = phases
            tasks = [(k, rep) for k in ks for rep in range(self.anchor_reps)]
            runs = _map(_anchor_task, tasks, n_workers)
            traces = [np.mean([tr for (k2, _), tr in zip(tasks, runs) if k2 == k], 0) for k in ks]

        # 5. calibration and summaries
        cal, s_lo, s_hi = calibrate_traces(mf, bl, ks, traces)
        summ = summarize(cal)
        res = HybridResult(phases, evs[0].temps, mf, bl, cal, stress, ks, traces, s_lo, s_hi, summ)
        samples_file = None
        if self.save_samples:
            samples_file = os.path.join(os.getcwd(), "hybrid_uq_samples.npz")
            res.save(samples_file)
        _G.clear()

        keep = [i for i, p in enumerate(phases) if summ["mean"][:, i].max() >= 0.002]
        tseq = sweep_temperatures(recipe.heating_schedule).tolist()
        doc = HybridUQResultDoc(
            recipe=recipe, target=target, n_samples=scores.n_samples, sigma_eV_per_atom=sigma,
            n_baselines=len(evs), phases=phases, temperature=[tseq[0]] + tseq,
            baseline=_per_phase(bl, phases, keep), mean=_per_phase(summ["mean"], phases, keep),
            std=_per_phase(summ["std"], phases, keep), q05=_per_phase(summ["q05"], phases, keep),
            q95=_per_phase(summ["q95"], phases, keep),
            final_per_sample={p: np.round(cal[:, -1, i], 5).tolist() for i, p in enumerate(phases)},
            gated=not ks, anchor_ks=list(ks), anchor_reps=self.anchor_reps if ks else 0,
            anchor_final={p: [round(float(t[-1, i]), 5) for t in traces] for i, p in enumerate(phases)},
            stress_mean=float(stress.mean()), stress_p95=float(np.percentile(stress, 95)),
            samples_file=samples_file, metadata=self.metadata, run_dir=os.getcwd())
        doc.to_file("hybrid_uq_result.json")
        return doc
