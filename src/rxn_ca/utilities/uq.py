"""Helpers for the hybrid UQ jobs in rxn_ca.computing.jobs (HybridUQMaker).

The method itself lives in rxn_ca.uq. These functions connect it to the jobflow flow: they build the recipes
for the baseline and anchor runs, compute the energy-sample scores, and turn the MultiRxnCAMaker outputs into
the cheap-ensemble and calibration documents.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..computing.schemas.ca_result_schema import RxnCAResultDoc
from ..computing.schemas.uq import HybridUQCheapDoc, HybridUQResultDoc
from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from ..uq.events import extract_events, sweep_temperatures
from ..uq.hybrid import (GATE_SD, QUANTILES, HybridResult, calibrate_traces, cheap_ensemble, observed_trace,
                         select_anchors, summarize)
from ..uq.perturb import (DEFAULT_SIGMA, PerturbedScores, library_reactions, perturbed_library,
                          perturbed_scores, sample_phase_deltas)

# Worker state for the scoring pool, set in the parent before it forks.
_G: dict = {}


def baseline_recipe(recipe: ReactionRecipe, n_baselines: int) -> ReactionRecipe:
    """The recipe for the unperturbed full-history runs: n_baselines realizations."""
    return dataclasses.replace(recipe, num_realizations=n_baselines, name=f"{recipe.name}__uq_baseline")


def anchor_recipe(recipe: ReactionRecipe, anchor_reps: int) -> ReactionRecipe:
    """The recipe for the anchor runs: anchor_reps realizations per anchor library."""
    return dataclasses.replace(recipe, num_realizations=anchor_reps, name=f"{recipe.name}__uq_anchor")


def n_sites(recipe: ReactionRecipe) -> int:
    return recipe.simulation_size ** 3


def _score_task(T: int) -> PerturbedScores:
    rset, ps, cls, kw = _G["rxn_set"], _G["phase_set"], _G["score_class"], _G["scorer_kwargs"]
    return perturbed_scores(lambda t: library_reactions(rset, t, ps),
                            lambda t: cls(temp=t, phase_set=ps, **kw), [T], _G["deltas"])


def score_energy_samples(library: ReactionLibrary, base_reactions, score_class, out_file: str,
                         n_samples: int = 500, sigma: Union[float, Dict[str, float]] = DEFAULT_SIGMA,
                         scorer_kwargs: Optional[dict] = None, num_workers: Optional[int] = None) -> str:
    """Score every library reaction under n_samples seeded energy-error samples; one temperature per process.

    The errors are per phase (sample_phase_deltas): sigma in eV/atom, one float for every solid phase or a
    dict phase -> sigma. base_reactions: the rxn_network ReactionSet the library was scored from.
    score_class: its scorer class. Writes a PerturbedScores npz to out_file, checks it against the library,
    and returns out_file.
    """
    temps = sorted(int(t) for t in library.temps)
    deltas, sig = sample_phase_deltas(library_reactions(base_reactions, temps[0], library.phases), n_samples,
                                      sigma, gas_phases=library.phases.gas_phases)
    _G.clear()
    _G.update(rxn_set=base_reactions, phase_set=library.phases, score_class=score_class,
              scorer_kwargs=scorer_kwargs or {}, deltas=deltas)
    n_workers = min(num_workers or mp.cpu_count(), len(temps))
    if n_workers > 1:
        with mp.get_context("fork").Pool(n_workers) as pool:
            parts = pool.map(_score_task, temps, chunksize=1)
    else:
        parts = [_score_task(T) for T in temps]
    _G.clear()
    scores = PerturbedScores(deltas, np.array(temps, dtype=np.int32),
                             np.concatenate([p.s for p in parts], 1), np.concatenate([p.s0 for p in parts], 0),
                             phase_sigma=sig)
    scores.check_library(library)
    scores.save(out_file)
    return out_file


def result_docs_of(multi_doc) -> List[RxnCAResultDoc]:
    """The per-recipe result docs of a MultiRxnCAResultDoc: from the store if saved there, else from its
    run directory (MultiRxnCAMaker writes result_doc_<i>.json there)."""
    if multi_doc.result_docs:
        return list(multi_doc.result_docs)
    return [RxnCAResultDoc.from_file(os.path.join(multi_doc.run_dir, f"result_doc_{i}.json"))
            for i in range(len(multi_doc.recipes))]


def cheap_analysis(baseline_multi_doc, library: ReactionLibrary, scores_file: str, target: str,
                   out_file: str, quantiles: Sequence[float] = QUANTILES,
                   gate_sd: float = GATE_SD) -> HybridUQCheapDoc:
    """Baseline runs -> event records -> cheap ensemble -> gate -> anchor samples and their libraries."""
    scores = PerturbedScores.load(scores_file)
    scores.check_library(library)
    docs = result_docs_of(baseline_multi_doc)
    n_rxn = len(library.get_rxns_at_temp(library.temps[0]).reactions)
    evs = [extract_events(res, doc.recipe.heating_schedule, library.phases, n_rxn)
           for doc in docs for res in doc.results]
    if not evs:
        raise ValueError("no full-history baseline realizations found")
    mf, bl, stress, phases = cheap_ensemble(evs, scores, library)
    ti = phases.index(target)
    ks = select_anchors(mf[:, -1, ti], quantiles, gate_sd)
    np.savez_compressed(out_file, cheap=mf.astype(np.float32), baseline=bl.astype(np.float32),
                        stress=stress.astype(np.float32), phases=np.array(phases), temps=evs[0].temps,
                        scores_file=scores_file)
    return HybridUQCheapDoc(
        target=target, n_samples=scores.n_samples, n_baselines=len(evs), phases=phases,
        cheap_file=os.path.abspath(out_file), anchor_ks=ks,
        anchor_libraries=[perturbed_library(library, scores, k) for k in ks], gated=not ks,
        cheap_final_target_mean=float(mf[:, -1, ti].mean()), cheap_final_target_std=float(mf[:, -1, ti].std()),
        stress_mean=float(stress.mean()), stress_p95=float(np.percentile(stress, 95)),
        phase_sigma=scores.phase_sigma, run_dir=os.getcwd())


def _per_phase(arr: np.ndarray, phases: Sequence[str], keep: Sequence[int]) -> Dict[str, List[float]]:
    return {phases[i]: np.round(arr[:, i], 5).tolist() for i in keep}


def calibration(cheap_doc: HybridUQCheapDoc, anchor_multi_doc, recipe: ReactionRecipe,
                samples_file: Optional[str] = None, metadata: Optional[dict] = None) -> HybridUQResultDoc:
    """Anchor runs -> per-anchor mean traces -> width calibration of the cheap ensemble -> summaries."""
    z = np.load(cheap_doc.cheap_file, allow_pickle=True)
    mf, bl, stress = z["cheap"].astype(float), z["baseline"].astype(float), z["stress"].astype(float)
    phases = list(cheap_doc.phases)
    traces, n_reps = [], 0
    if cheap_doc.anchor_ks:
        docs = result_docs_of(anchor_multi_doc)
        if len(docs) != len(cheap_doc.anchor_ks):
            raise ValueError(f"{len(docs)} anchor result docs for {len(cheap_doc.anchor_ks)} anchors")
        for doc in docs:                     # same order as anchor_ks / anchor_libraries
            reps = [observed_trace(o, doc.recipe.heating_schedule, doc.phases, phases)
                    for o in doc.observed_results]
            if not reps:
                raise ValueError("an anchor has no analysis-only realizations")
            n_reps = len(reps)
            traces.append(np.mean(reps, 0))
    cal, s_lo, s_hi = calibrate_traces(mf, bl, cheap_doc.anchor_ks, traces)
    summ = summarize(cal)
    if samples_file:
        HybridResult(phases, z["temps"], mf, bl, cal, stress, list(cheap_doc.anchor_ks), traces, s_lo, s_hi,
                     summ).save(samples_file)
    keep = [i for i, p in enumerate(phases) if summ["mean"][:, i].max() >= 0.002]
    tseq = sweep_temperatures(recipe.heating_schedule).tolist()
    return HybridUQResultDoc(
        recipe=recipe, target=cheap_doc.target, n_samples=cheap_doc.n_samples, n_baselines=cheap_doc.n_baselines,
        phases=phases, temperature=[tseq[0]] + tseq,
        baseline=_per_phase(bl, phases, keep), mean=_per_phase(summ["mean"], phases, keep),
        std=_per_phase(summ["std"], phases, keep), q05=_per_phase(summ["q05"], phases, keep),
        q95=_per_phase(summ["q95"], phases, keep),
        final_per_sample={p: np.round(cal[:, -1, i], 5).tolist() for i, p in enumerate(phases)},
        gated=cheap_doc.gated, anchor_ks=list(cheap_doc.anchor_ks), anchor_reps=n_reps,
        anchor_final={p: [round(float(t[-1, i]), 5) for t in traces] for i, p in enumerate(phases)},
        stress_mean=float(stress.mean()), stress_p95=float(np.percentile(stress, 95)),
        samples_file=os.path.abspath(samples_file) if samples_file else None, phase_sigma=cheap_doc.phase_sigma,
        metadata=metadata, run_dir=os.getcwd())
