"""The cheap estimator: replay a baseline run's events under perturbed scores.

Event model (from rxn_ca.core): a site visit picks one interaction (a reactant set) with probability
proportional to its top score / d^3, or the no-op with weight `inertia`; then one reaction within the set,
proportional to its score. Changing the scores from s(0) to s(delta) changes the odds of "reaction r chosen"
by the event weight

    A_r = [max_g s(delta) / max_g s(0)] * [s_r(delta) / s_r(0)] * [sum_g s(0) / sum_g s(delta)]

(g = r's reactant set). The normalization across interactions is not recorded, so each sweep is normalized
against its recorded no-op count instead (mean-field):

    u_r(t) = N_r(t) * A_r * avail_r(t),   Ntilde_r(t) = (N_none(t) + sum_r N_r(t)) * u_r / (N_none(t) + sum_r u_r)

avail_r(t) = prod over r's solid reactants p of min((V_p(t) + eps) / (V0_p(t) + eps), cap) is the mass-action
term: a reactant that is more (less) abundant than in the baseline is contacted more (less) often. V0 is the
baseline's own replayed volume, so delta = 0 reproduces the baseline exactly.

Volumes advance by the baseline's mean volume change per choice: V(t+1) = V(t) + sum_r Ntilde_r(t) dvol_r(t) / N_r(t).
If a sweep's reweighted choices would consume more of a phase than exists, every reaction consuming it is
scaled down so the phase lands at zero (limiting reagent). No volume is ever deleted. The fraction of proposed
consumption removed this way is returned as `stress` (0 = the closure never had to step in).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
from pymatgen.core import Composition

from ..reactions import ReactionLibrary

EPS = 1e-3      # regularizes the availability ratio when a baseline phase volume goes to zero
CAP = 30.0      # upper bound on a single reactant's availability ratio


def _reactions_in_id_order(library: ReactionLibrary):
    rset = library.get_rxns_at_temp(library.temps[0])
    return [rset.get_rxn_by_id(r) for r in range(len(rset.reactions))]


def reaction_groups(library: ReactionLibrary) -> List[np.ndarray]:
    """Reaction ids grouped by reactant set (the interaction a site visit chooses between)."""
    groups: Dict[frozenset, list] = {}
    for i, rx in enumerate(_reactions_in_id_order(library)):
        groups.setdefault(frozenset(rx.reactants), []).append(i)
    return [np.array(ids) for ids in groups.values()]


def solid_reactant_lists(library: ReactionLibrary) -> List[List[str]]:
    gases = set(library.phases.gas_phases)
    return [[p for p in rx.reactants if p not in gases] for rx in _reactions_in_id_order(library)]


def weight_matrix(s0: np.ndarray, s_new: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """Event weights A_r at one temperature, from baseline scores s0 and perturbed scores s_new (both (R,))."""
    a = np.ones_like(s0, dtype=float)
    for ids in groups:
        top = s_new[ids].max() / max(s0[ids].max(), 1e-300)
        norm = s0[ids].sum() / max(s_new[ids].sum(), 1e-300)
        a[ids] = top * norm * (s_new[ids] / np.maximum(s0[ids], 1e-300))
        a[ids[s0[ids] <= 1e-300]] = 0.0      # never chosen in the baseline, so no recorded counts anyway
    return a


def weight_matrices(scores, groups: Sequence[np.ndarray], ks=None) -> Dict[int, np.ndarray]:
    """T -> (S, R) event weights for samples ks (default all) of a PerturbedScores."""
    ks = range(scores.n_samples) if ks is None else ks
    return {int(T): np.array([weight_matrix(scores.s0[it], scores.s[k, it], groups) for k in ks])
            for it, T in enumerate(scores.temps)}


def _baseline_recurrence(counts, per_choice, vol0):
    n_sweeps = counts.shape[0]
    V0 = np.zeros((n_sweeps + 1, vol0.shape[0]))
    V0[0] = vol0
    for t in range(n_sweeps):
        V0[t + 1] = np.maximum(V0[t] + (counts[t][:, None] * per_choice[t]).sum(0), 0.0)
    return V0


def forward_lr_batch(ev, A_batch: Mapping[int, np.ndarray], solid_reactants: Sequence[Sequence[str]],
                     cap: float = CAP, eps: float = EPS):
    """Replay one baseline realization under S samples of event weights at once.

    ev: EventRecord (or a dict with counts, none, dvol, vol, temps, phases).
    A_batch: T -> (S, R) event weights (weight_matrices).
    Returns (vol [S, n_sweeps+1, P], stress [S]).
    """
    get = (lambda k: ev[k]) if isinstance(ev, dict) else (lambda k: getattr(ev, k))
    counts, none, dvol, vol0, temps = get("counts"), get("none"), get("dvol"), get("vol"), get("temps")
    pidx = {str(p): i for i, p in enumerate(get("phases"))}
    n_sweeps, n_rxn = counts.shape
    S = next(iter(A_batch.values())).shape[0]
    P = vol0.shape[1]
    per_choice = np.zeros_like(dvol, dtype=float)
    nz = counts > 0
    per_choice[nz] = dvol[nz] / counts[nz][:, None]
    neg = np.minimum(per_choice, 0.0)
    # reactant index table, padded with a dummy column P whose ratio is fixed at 1
    rmax = max(1, max(len(sr) for sr in solid_reactants))
    ridx = np.full((n_rxn, rmax), P, dtype=int)
    for r, sr in enumerate(solid_reactants):
        for m, p in enumerate(sr):
            ridx[r, m] = pidx[p]
    V0 = _baseline_recurrence(counts, per_choice, vol0[0].astype(float))
    V = np.repeat(V0[0][None], S, axis=0)
    out = np.zeros((S, n_sweeps + 1, P))
    out[:, 0] = V
    removed = np.zeros(S)
    proposed = np.zeros(S)
    for t in range(n_sweeps):
        a = A_batch[int(temps[t])].copy()
        ratio = np.minimum((V + eps) / (V0[t] + eps), cap)                  # (S, P)
        ratio = np.concatenate([ratio, np.ones((S, 1))], axis=1)
        av = np.maximum(ratio[:, ridx].prod(2), 0.0)                        # (S, R)
        live = counts[t] > 0
        a = np.where(live[None], a * av, a)
        u = counts[t][None] * a
        z = none[t] + u.sum(1, keepdims=True)
        ntilde = (none[t] + counts[t].sum()) * u / z                       # (S, R)
        nt0 = ntilde.copy()
        cons_mask = neg[t] < 0                                              # (R, P)
        for _ in range(3):                                                  # limiting reagent, with knock-ons
            cons = -np.einsum("sr,rp->sp", ntilde, neg[t])
            over = cons > V * (1 + 1e-9) + 1e-9
            if not over.any():
                break
            scale_p = np.where(over, V / np.maximum(cons, 1e-300), 1.0)    # (S, P)
            s_r = np.where(cons_mask[None], scale_p[:, None, :], np.inf).min(2)
            s_r = np.where(np.isfinite(s_r), s_r, 1.0)
            ntilde = ntilde * np.where(ntilde > 0, s_r, 1.0)
        c0 = -np.einsum("sr,rp->s", nt0, neg[t])
        proposed += c0
        removed += c0 + np.einsum("sr,rp->s", ntilde, neg[t])
        V = np.maximum(V + np.einsum("sr,rp->sp", ntilde, per_choice[t]), 0.0)
        out[:, t + 1] = V
    return out, removed / np.maximum(proposed, 1e-300)


def forward_lr(ev, A: Mapping[int, np.ndarray], solid_reactants, cap: float = CAP, eps: float = EPS):
    """forward_lr_batch for one sample. A: T -> (R,). Returns (vol [n_sweeps+1, P], stress)."""
    vol, st = forward_lr_batch(ev, {T: np.asarray(a)[None] for T, a in A.items()}, solid_reactants, cap, eps)
    return vol[0], float(st[0])


def mass_fractions(vol: np.ndarray, phases: Sequence[str], molar_vols: Mapping[str, float]) -> np.ndarray:
    """Solid phase volumes (..., P) -> mass fractions (..., P)."""
    w = np.array([Composition(p).weight / molar_vols[p] for p in phases])
    m = vol * w
    return m / np.maximum(m.sum(-1, keepdims=True), 1e-12)
