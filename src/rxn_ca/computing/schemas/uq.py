from dataclasses import dataclass
from typing import Dict, List, Optional

from ...core.recipe import ReactionRecipe
from ...reactions import ReactionLibrary
from .base_schema import BaseSchema


@dataclass
class HybridUQCheapDoc(BaseSchema):
    """Output of the cheap-ensemble analysis job of the hybrid UQ flow.

    The per-sample cheap traces are in cheap_file (npz: cheap [S, T, P], baseline [T, P], stress [S], phases).
    anchor_libraries[i] is the library with energy sample anchor_ks[i] applied; it is what the anchor runs use.
    gated is True when the target's final cheap fraction barely moves: then there are no anchors.
    """
    target: str
    n_samples: int
    n_baselines: int
    phases: List[str]
    cheap_file: str
    anchor_ks: List[int]
    anchor_libraries: List[ReactionLibrary]
    gated: bool
    cheap_final_target_mean: float
    cheap_final_target_std: float
    stress_mean: float
    stress_p95: float
    run_dir: str = None


@dataclass
class HybridUQResultDoc(BaseSchema):
    """Hybrid UQ result for one recipe + library.

    Traces are mass fractions of the solid phases, one value per sweep plus the initial state
    (temperature[i] is the temperature of trace point i). Phases that never reach 0.002 are left out of the
    trace fields. final_per_sample holds every phase's calibrated final fraction for every energy sample.
    The full per-sample traces are in samples_file (npz).
    """
    recipe: ReactionRecipe
    target: str
    n_samples: int
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
