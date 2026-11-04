"""Energy perturbations: draw per-reaction energy errors and re-score a library with them.

Each sample k gives every reaction r an error delta_r(k) ~ N(0, sigma) in eV/atom of reaction. The error is
the same at every temperature, and a reaction and its reverse get opposite errors. The draw is seeded from
(k, reaction), so the same sample number gives the same error in every energy model, scorer and recipe
(paired samples).

Scores are recomputed with the real scorer objects, so any scorer works (TammanScore, DiffusionScorer, ...).
Only the scored reaction energy is shifted; phase energies (and so a DiffusionScorer's chemical potential
diagram) are not.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np

from ..reactions import ReactionLibrary, ScoredReaction, ScoredReactionSet

DEFAULT_SIGMA = 0.015   # eV/atom, per reaction energy


def canonical_key(stoich: Mapping[str, float]):
    """(key, sign). key is the same for a reaction and its reverse; sign is +1 for the given orientation.

    stoich: formula -> coefficient, negative for reactants (e.g. {"CaCO3": -1, "CaO": 1, "CO2": 1}).
    """
    k = tuple(sorted((f, round(v, 6)) for f, v in stoich.items()))
    kr = tuple(sorted((f, round(-v, 6)) for f, v in stoich.items()))
    return (k, 1.0) if k <= kr else (kr, -1.0)


def _draw(sample: int, key, sigma: float) -> float:
    seed = [sample, zlib.crc32(repr(key).encode())]
    return float(np.random.default_rng(seed).normal(0.0, sigma))


def sample_deltas(stoichs: Sequence[Mapping[str, float]], n_samples: int, sigma: float = DEFAULT_SIGMA,
                  first_sample: int = 0) -> np.ndarray:
    """Seeded energy errors, shape (n_samples, n_reactions), eV/atom.

    stoichs: one formula -> coefficient map per library reaction id (reactants negative).
    """
    keys = [canonical_key(s) for s in stoichs]
    return np.array([[sgn * _draw(k, key, sigma) for key, sgn in keys]
                     for k in range(first_sample, first_sample + n_samples)])


def stoich_of(rxn) -> Dict[str, float]:
    """formula -> coefficient (reactants negative) of a pymatgen / rxn_network reaction."""
    return {c.reduced_formula: float(v) for c, v in zip(rxn.compositions, rxn.coefficients)}


@dataclass
class PerturbedScores:
    """Scores of every library reaction under every energy sample.

    deltas: (S, R) energy errors, eV/atom
    temps:  (nT,) temperatures, K
    s:      (S, nT, R) perturbed scores
    s0:     (nT, R) unperturbed scores (must equal the library's competitiveness)
    """
    deltas: np.ndarray
    temps: np.ndarray
    s: np.ndarray
    s0: np.ndarray

    def save(self, path):
        np.savez_compressed(path, deltas=self.deltas, temps=self.temps, s=self.s, s0=self.s0)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["deltas"], z["temps"], z["s"], z["s0"])

    @property
    def n_samples(self):
        return self.s.shape[0]

    def subset(self, ks) -> "PerturbedScores":
        ks = np.asarray(ks, dtype=int)
        return PerturbedScores(self.deltas[ks], self.temps, self.s[ks], self.s0)

    def check_library(self, library: ReactionLibrary, rtol: float = 1e-6):
        """Raise if s0 does not match the library's stored competitiveness (reaction ids misaligned)."""
        for i, T in enumerate(self.temps):
            rs = library.get_rxns_at_temp(int(T))
            sl = np.array([rs.get_rxn_by_id(r).competitiveness for r in range(self.s0.shape[1])])
            dev = np.abs(sl - self.s0[i]).max() / max(np.abs(sl).max(), 1e-300)
            if dev > rtol:
                raise ValueError(f"scores at delta = 0 differ from the library at {T} K (rel. dev {dev:.2e}); "
                                 "are the reactions in library id order?")


def perturbed_scores(reactions_at: Callable[[int], List], scorer_at: Callable[[int], object],
                     temps: Sequence[int], deltas: np.ndarray) -> PerturbedScores:
    """Score every library reaction at every temperature under every energy sample.

    reactions_at(T): the reactions (rxn_network ComputedReaction, with energy_per_atom at T) in library id
        order. These are the objects the library was scored from.
    scorer_at(T): the scorer used to build the library at T (anything with .score(rxn)).
    deltas: (S, R) energy errors from sample_deltas.
    """
    deltas = np.asarray(deltas, dtype=float)
    S, R = deltas.shape
    temps = np.asarray(sorted(int(t) for t in temps), dtype=np.int32)
    s = np.full((S, len(temps), R), np.nan)
    s0 = np.full((len(temps), R), np.nan)
    for it, T in enumerate(temps):
        base = list(reactions_at(int(T)))
        if len(base) != R:
            raise ValueError(f"{len(base)} reactions at {T} K, deltas have {R}")
        sc = scorer_at(int(T))
        s0[it] = [sc.score(r) for r in base]
        for i, r in enumerate(base):
            e0 = r.energy_per_atom
            for k in range(S):
                rk = copy.copy(r)
                rk.__dict__["energy_per_atom"] = e0 + deltas[k, i]   # overrides the cached_property
                s[k, it, i] = sc.score(rk)
    return PerturbedScores(deltas, temps, s, s0)


def library_reactions(rxn_set, temp: int, phase_set) -> List:
    """The reactions of an rxn_network ReactionSet at `temp`, in library id order.

    Mirrors rxn_ca.reactions.score_rxns: reactions with only gas reactants are not in the library.
    """
    rset = rxn_set.set_new_temperature(temp)
    out = []
    for rxn in rset.get_rxns():
        if any(c.reduced_formula not in phase_set.gas_phases for c in rxn.reactants):
            out.append(rxn)
    return out


def perturbed_library(library: ReactionLibrary, scores: PerturbedScores, k: int) -> ReactionLibrary:
    """Copy of the library with sample k applied: energies shifted by delta_k and scores replaced."""
    tix = {int(T): i for i, T in enumerate(scores.temps)}
    out = ReactionLibrary(library.phases)
    for T, rset in library.lib.items():
        it = tix[int(T)]
        rxns = []
        for r in range(len(rset.reactions)):
            rx = rset.get_rxn_by_id(r)
            e = None if rx.energy_per_atom is None else rx.energy_per_atom + scores.deltas[k, r]
            rxns.append(ScoredReaction(dict(rx._reactants), dict(rx._products), float(scores.s[k, it, r]),
                                       energy_per_atom=e))
        out.add_rxns_at_temp(ScoredReactionSet(rxns, phase_set=library.phases), T)
    return out
