"""Event records from an unperturbed (baseline) simulation.

The cheap estimator needs, per sweep, how often each reaction was chosen and the volume change those choices
caused. This module extracts that from one full-history realization (run_single_sim with analysis_only=False,
no compress_freq, so every per-site diff is kept).

A sweep is one visit per site (n_sites diffs). Between heating stages the runner inserts one joiner diff
whose REACTION_CHOSEN is stale; those are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from pylattica.core.constants import GENERAL, SITES
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY

from ..core.constants import REACTION_CHOSEN, VOLUME
from ..core.heating import HeatingSchedule, HeatingStep
from ..phases.solid_phase_set import SolidPhaseSet


@dataclass
class EventRecord:
    """Per-sweep event aggregates of one baseline realization.

    counts[t, r]   number of times reaction r was chosen in sweep t
    none[t]        site visits in sweep t with no reaction (free space, swaps, inertia no-ops)
    dvol[t, r, p]  volume change of solid phase p caused by sweep t's choices of reaction r
    vol[t, p]      solid phase volumes before sweep t (vol[0] = initial state; n_sweeps + 1 rows)
    temps[t]       temperature of sweep t, K
    phases         solid phases (column order of dvol / vol)
    """
    counts: np.ndarray
    none: np.ndarray
    dvol: np.ndarray
    vol: np.ndarray
    temps: np.ndarray
    phases: List[str] = field(default_factory=list)

    def as_arrays(self):
        return dict(counts=self.counts, none=self.none, dvol=self.dvol, vol=self.vol, temps=self.temps)

    def save(self, path):
        np.savez_compressed(path, phases=np.array(self.phases), **self.as_arrays())

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(z["counts"], z["none"], z["dvol"], z["vol"], z["temps"], [str(p) for p in z["phases"]])

    @property
    def n_sweeps(self):
        return self.counts.shape[0]


def solid_phases(phase_set: SolidPhaseSet) -> List[str]:
    return sorted(p for p in phase_set.phases if p not in phase_set.gas_phases and p != SolidPhaseSet.FREE_SPACE)


def sweep_temperatures(schedule: HeatingSchedule) -> np.ndarray:
    steps = schedule.steps
    if any(not isinstance(s, HeatingStep) for s in steps):
        raise ValueError("event extraction supports heating steps only (no regrind steps)")
    return np.concatenate([[s.temperature] * s.duration for s in steps]).astype(np.int32)


def extract_events(result, schedule: HeatingSchedule, phase_set: SolidPhaseSet, n_reactions: int) -> EventRecord:
    """Build the EventRecord of one full-history realization.

    result: one realization, e.g. ``doc.results[0]`` of a run_single_sim(..., analysis_only=False) doc.
    schedule: the recipe's heating schedule. phase_set: the library's phase set.
    n_reactions: number of reactions in the library (ids 0 .. n_reactions-1).
    """
    phases = solid_phases(phase_set)
    pidx = {p: i for i, p in enumerate(phases)}
    temps = sweep_temperatures(schedule)
    durations = [s.duration for s in schedule.steps]
    n_sweeps = len(temps)

    first = result.first_step
    site_ids = list(first.site_ids())
    n_sites = len(site_ids)
    counts = np.zeros((n_sweeps, n_reactions), dtype=np.int32)
    none = np.zeros(n_sweeps, dtype=np.int32)
    dvol = np.zeros((n_sweeps, n_reactions, len(phases)), dtype=np.float32)
    vol = np.zeros((n_sweeps + 1, len(phases)), dtype=np.float32)

    occ, v = {}, {}
    for sid in site_ids:
        st = first.get_site_state(sid)
        occ[sid], v[sid] = st[DISCRETE_OCCUPANCY], st[VOLUME]
        if occ[sid] in pidx:
            vol[0, pidx[occ[sid]]] += v[sid]

    diffs = result.get_diffs()
    expected = n_sweeps * n_sites + len(durations) - 1
    if len(diffs) != expected:
        raise ValueError(f"{len(diffs)} diffs, expected {expected}: the result must keep every per-site diff "
                         "(analysis_only=False, no compress_freq / num_frames)")
    event_diffs, ptr = [], 0
    for si, dur in enumerate(durations):
        if si > 0:
            ptr += 1                                   # stage joiner diff
        event_diffs.extend(diffs[ptr:ptr + dur * n_sites])
        ptr += dur * n_sites

    for i, diff in enumerate(event_diffs):
        t = i // n_sites
        rid = (diff.get(GENERAL) or {}).get(REACTION_CHOSEN)
        if rid is None:
            none[t] += 1
        else:
            counts[t, rid] += 1
        for sid, upd in (diff.get(SITES) or {}).items():
            po, pn = occ[sid], upd.get(DISCRETE_OCCUPANCY, occ[sid])
            vo, vn = v[sid], upd.get(VOLUME, v[sid])
            if rid is not None:
                if po in pidx:
                    dvol[t, rid, pidx[po]] -= vo
                if pn in pidx:
                    dvol[t, rid, pidx[pn]] += vn
            occ[sid], v[sid] = pn, vn
        if (i + 1) % n_sites == 0:
            tot = np.zeros(len(phases), dtype=np.float32)
            for sid in occ:
                if occ[sid] in pidx:
                    tot[pidx[occ[sid]]] += v[sid]
            vol[t + 1] = tot
    return EventRecord(counts, none, dvol, vol, temps, phases)
