"""Energy perturbations: draw energy errors and re-score a library with them.

Two error models, both giving per-reaction errors delta_r (eV/atom of reaction), the same at every
temperature, opposite for a reaction and its reverse:

- per phase (the default, sample_phase_deltas): every phase p gets its own error d_p ~ N(0, sigma_p^2)
  (eV/atom), independent across phases. A reaction's error follows exactly from its stoichiometry,
  delta_r = sum_c nu_c n_c d_c / N_r (nu_c: coefficient, n_c: atoms per formula, N_r: atoms in the reaction,
  as rxn_network normalizes energy_per_atom), so reactions that share a phase get correlated errors.
  sigma is one float for every solid phase, or a dict phase -> sigma (unnamed phases get 0). rho in [0, 1]
  sets how correlated the phase errors are (0: independent; 1 with equal sigmas: one shared offset, which
  cancels in balanced reactions).
- per reaction (sample_deltas): every reaction gets an independent error ~ N(0, sigma^2).

Draws are seeded from (sample, phase) or (sample, reaction), so the same sample number gives the same error
in every energy model, scorer and recipe (paired samples).

Scores are recomputed with the real scorer objects, so any scorer works (TammanScore, DiffusionScorer, ...).
Only the scored reaction energy is shifted; phase energies (and so a DiffusionScorer's chemical potential
diagram) are not.
"""
from __future__ import annotations

import copy
import json
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..reactions import ReactionLibrary, ScoredReaction, ScoredReactionSet

DEFAULT_SIGMA = 0.015   # eV/atom


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


def phase_sigmas(phases: Sequence[str], sigma: Union[float, Mapping[str, float]] = DEFAULT_SIGMA,
                 gas_phases: Sequence[str] = ()) -> Dict[str, float]:
    """sigma per phase (eV/atom). A float applies to every phase not in gas_phases (gases get 0). A dict gives
    each named phase its sigma; phases it does not name get 0 (no error)."""
    if isinstance(sigma, Mapping):
        out = {p: float(sigma.get(p, 0.0)) for p in phases}
    else:
        gases = set(gas_phases)
        out = {p: (0.0 if p in gases else float(sigma)) for p in phases}
    if any(v < 0 for v in out.values()):
        raise ValueError("sigma must be >= 0")
    return out


def sample_phase_deltas(reactions: Sequence, n_samples: int,
                        sigma: Union[float, Mapping[str, float]] = DEFAULT_SIGMA, gas_phases: Sequence[str] = (),
                        rho: float = 0.0, first_sample: int = 0):
    """Per-reaction errors from per-phase errors. Returns (deltas (n_samples, R), sigmas per phase).

    reactions: rxn_network ComputedReactions in library id order (only compositions, coefficients and
        num_atoms are used, so any temperature works).
    sigma: eV/atom; one float for all non-gas phases, or a dict phase -> sigma (see phase_sigmas).
    rho: correlation between the phase errors, in [0, 1]. Each sample draws one shared z and one z_p per
        phase, and d_p = sigma_p (sqrt(rho) z + sqrt(1 - rho) z_p). rho = 0: independent phases. rho = 1 with
        equal sigmas: every phase is off by the same amount, which cancels in every balanced reaction without
        unperturbed phases (gases).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    phases = sorted({c.reduced_formula for r in reactions for c in r.compositions})
    sig = phase_sigmas(phases, sigma, gas_phases)
    pix = {p: i for i, p in enumerate(phases)}
    s_vec = np.array([sig[p] for p in phases])
    ks = range(first_sample, first_sample + n_samples)
    # z_p(k) seeded from (k, phase), z(k) from (k, shared key): the same draws in every library (paired samples)
    zp = np.array([[_draw(k, p, 1.0) for p in phases] for k in ks])
    z = np.array([_draw(k, "__shared_phase_error__", 1.0) for k in ks])[:, None]
    d = s_vec[None] * (np.sqrt(rho) * z + np.sqrt(1.0 - rho) * zp)
    # delta_r = sum_c nu_c n_c d_c / N_r
    M = np.zeros((len(phases), len(reactions)))
    for j, r in enumerate(reactions):
        for c, nu in zip(r.compositions, r.coefficients):
            M[pix[c.reduced_formula], j] += float(nu) * c.num_atoms / r.num_atoms
    return d @ M, sig


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
    phase_sigma: sigma per phase (eV/atom) when the deltas came from sample_phase_deltas, else None
    phase_correlation: rho used by sample_phase_deltas, else None
    """
    deltas: np.ndarray
    temps: np.ndarray
    s: np.ndarray
    s0: np.ndarray
    phase_sigma: Optional[Dict[str, float]] = None
    phase_correlation: Optional[float] = None

    def save(self, path):
        extra = {}
        if self.phase_sigma is not None:
            extra["phase_sigma"] = np.array(json.dumps(self.phase_sigma))
        if self.phase_correlation is not None:
            extra["phase_correlation"] = np.array(self.phase_correlation)
        np.savez_compressed(path, deltas=self.deltas, temps=self.temps, s=self.s, s0=self.s0, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        sig = json.loads(str(z["phase_sigma"])) if "phase_sigma" in z.files else None
        rho = float(z["phase_correlation"]) if "phase_correlation" in z.files else None
        return cls(z["deltas"], z["temps"], z["s"], z["s0"], sig, rho)

    @property
    def n_samples(self):
        return self.s.shape[0]

    def subset(self, ks) -> "PerturbedScores":
        ks = np.asarray(ks, dtype=int)
        return PerturbedScores(self.deltas[ks], self.temps, self.s[ks], self.s0, self.phase_sigma,
                               self.phase_correlation)

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
