from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from ..computing.schemas.multi_ca_result_schema import MultiRxnCAResultDoc
from ..computing.schemas.ca_result_schema import RxnCAResultDoc
from ..computing.schemas.uq import HybridUQCheapDoc, HybridUQResultDoc
from ..utilities.parallel_sim import run_multi_recipe_parallel_ray
from ..utilities.uq import (anchor_recipe, baseline_recipe, calibration, cheap_analysis, n_sites,
                            score_energy_samples)
from ..uq.hybrid import GATE_SD, QUANTILES
from ..uq.perturb import DEFAULT_SIGMA
from pylattica.core import Simulation
from jobflow import job, Flow, Maker
from dataclasses import dataclass, field
from typing import Dict, List, Union
import os
from enum import Enum

_JOBSTORE_OBJECTS = [ReactionLibrary, RxnCAResultDoc] 
#have to include RxnCAResultDoc because it contains lists of ReactionResult objects, 
#and jobflow DOES NOT automatically recursively search inside each list item.
#solution is to use a MultiRxnCAResultDoc as the output schema.

class MultiRxnType(Enum):
    MULTI_LIB = "multi_lib"
    MULTI_RECIPE = "multi_recipe"
    MULTI_LIB_AND_RECIPE = "multi_lib_and_recipe"


@dataclass
class MultiRxnCAMaker(Maker):
    """
    Maker for running multiple recipes in parallel using Ray. Can run one of the following:
    - MULTI_LIB: Run multiple recipes in parallel using one reaction library.
    - MULTI_RECIPE: Run multiple recipes in parallel using multiple reaction libraries.
    - MULTI_LIB_AND_RECIPE: Run multiple recipes in parallel using multiple reaction libraries and multiple initial simulations.
    Default is MULTI_LIB_AND_RECIPE. 

    Parameters
    ----------
    multi_rxn_type (Enum): The type of multi-recipe run to perform.
        Instance of rxn_ca.computing.jobs.MultiRxnType.
    metadata (dict): Metadata to save with the result doc.
        Default is None. Helps with tagging the result doc.
    save_results_to_store (bool): Whether to save the result docs to the store.
        Default is False. The result docs are saved to the launch directory by default, 
        and not the JobStore becuase these files can be extremely large. 
    num_workers (int): Number of workers to use for the parallel run.
        Default is None, which uses all available cores on the current node.
    memory_per_task_gb (float): Memory per task in GB.
        Default is None, which uses the default memory per task.
    cpus_per_task (int): Number of CPUs per task.
        Default is 1. Any more does not usually improve performance.
        Consider increasing if you are running into memory issues.
    compress (bool): Whether to compress the result docs.
        Default is True. Compresing the output result docs is highly recommended.
    reaction_plotter_kwargs (dict): Keyword arguments for the reaction plotter.
        Default is {"include_heating_trace": True}. 
        See rxn_ca.analysis.visualization.reaction_plotter for more details.
    compute_plots (bool): Whether to compute the plot data and convergence flags for the result doc.
        Default is True. Set False for full-history runs, where loading every step for the plots is slow.
    """
    
    name: str = "MultiRxnCA Maker"
    multi_rxn_type: MultiRxnType = field(default=MultiRxnType.MULTI_LIB_AND_RECIPE)
    metadata: dict = field(default=None)
    save_results_to_store: bool = field(default=False)
    num_workers: int = None
    memory_per_task_gb: float = field(default=None)
    cpus_per_task: int = field(default=1)
    compress: bool = field(default=True)
    reaction_plotter_kwargs: dict = field(default_factory=lambda: {"include_heating_trace": True})
    compute_plots: bool = field(default=True)
    
    @job(data=_JOBSTORE_OBJECTS)
    def make(self, recipes: List[ReactionRecipe] | ReactionRecipe, 
             reaction_libraries: List[ReactionLibrary] | ReactionLibrary, 
             initial_simulations: List[Simulation] | Simulation | str = None,
             **kwargs) -> MultiRxnCAResultDoc:
        
        if self.multi_rxn_type == MultiRxnType.MULTI_LIB:
            if isinstance(recipes, list) and len(recipes) != 1:
                raise ValueError("MultiRxnType.MULTI_LIB requires exactly one recipe")
            
            print(f"Running {len(reaction_libraries)} library realizations for given recipe")
            if isinstance(recipes, ReactionRecipe):
                recipes = [recipes] * len(reaction_libraries)
            else:
                recipes = [recipes[0]] * len(reaction_libraries)
            
        if self.multi_rxn_type == MultiRxnType.MULTI_RECIPE:
            if isinstance(reaction_libraries, list) and len(reaction_libraries) != 1:
                raise ValueError("MultiRxnType.MULTI_RECIPE requires exactly one reaction library")
            
            print(f"Running {len(recipes)} recipe realizations with given reaction library")
            if isinstance(reaction_libraries, ReactionLibrary):
                reaction_libraries = [reaction_libraries] * len(recipes)
            else:
                reaction_libraries = [reaction_libraries[0]] * len(recipes)
            
        if self.multi_rxn_type == MultiRxnType.MULTI_LIB_AND_RECIPE:
            if len(recipes) != len(reaction_libraries):
                raise ValueError("MultiRxnType.MULTI_LIB_AND_RECIPE requires the same number of recipes and reaction libraries")
            
        if initial_simulations is None:
            initial_simulations = [None] * len(recipes)
        if initial_simulations:
            if not isinstance(initial_simulations, list):
                if isinstance(initial_simulations, Simulation):
                    initial_simulations = [initial_simulations] * len(recipes)
                elif isinstance(initial_simulations, str):
                    initial_simulations = [Simulation.from_file(initial_simulations)] * len(recipes)
                else:
                    raise ValueError("If initial_simulations is provided, it must be a list of Simulation objects, \
                        a single Simulation object, or a string path to a Simulation object")
            elif len(initial_simulations) != len(recipes):
                raise ValueError("If initial_simulations is provided, it must be the same length as the number of recipes/libraries")
            
        result_docs = run_multi_recipe_parallel_ray(recipes, 
                                                    reaction_libraries, 
                                                    initial_simulations, 
                                                    num_workers=self.num_workers, 
                                                    memory_per_task_gb=self.memory_per_task_gb, 
                                                    cpus_per_task=self.cpus_per_task, 
                                                    compress=self.compress,
                                                    **kwargs)
        
        print("============ SAVING RESULT DOCS =============")
        
        for i,result_doc in enumerate(result_docs):
            print(f"Saving result doc {i} to file...")
            result_doc.to_file(f"result_doc_{i}.json")

        print("============ CREATING MULTI RXN CA RESULT DOCUMENT =============")
        multi_rxn_ca_result_doc = MultiRxnCAResultDoc.from_multiple_jobs(result_docs, 
                                                                        metadata=self.metadata, 
                                                                        save_results_to_store=self.save_results_to_store,
                                                                        reaction_plotter_kwargs=self.reaction_plotter_kwargs,
                                                                        compute_plots=self.compute_plots,
                                                                        run_dir=os.getcwd())
        multi_rxn_ca_result_doc.to_file("multi_rxn_ca_result.json")
        return multi_rxn_ca_result_doc

# ---------------------------------------------------------------------------------------------------------
# Hybrid UQ: phase-trace distributions under reaction-energy errors (method in rxn_ca.uq)
# ---------------------------------------------------------------------------------------------------------

UQ_SCORES_FILE = "uq_scores.npz"
UQ_CHEAP_FILE = "uq_cheap.npz"
UQ_SAMPLES_FILE = "hybrid_uq_samples.npz"
UQ_RESULT_FILE = "hybrid_uq_result.json"


@job
def score_energy_samples_job(reaction_library: ReactionLibrary, base_reactions, recipe: ReactionRecipe,
                             n_samples: int = 500, sigma: Union[float, Dict[str, float]] = DEFAULT_SIGMA,
                             phase_correlation: float = 0.0, scorer_kwargs: dict = None,
                             num_workers: int = None) -> str:
    """Score the library under n_samples per-phase energy-error samples. Returns the path of the scores npz."""
    return score_energy_samples(reaction_library, base_reactions, recipe.get_score_class(),
                                os.path.abspath(UQ_SCORES_FILE), n_samples=n_samples, sigma=sigma,
                                rho=phase_correlation,
                                scorer_kwargs=scorer_kwargs, num_workers=num_workers)


@job(data=[ReactionLibrary])
def hybrid_uq_cheap_job(baseline_doc: MultiRxnCAResultDoc, reaction_library: ReactionLibrary, scores_file: str,
                        target: str, quantiles: tuple = QUANTILES, gate_sd: float = GATE_SD) -> HybridUQCheapDoc:
    """Cheap ensemble from the baseline runs. Picks the anchor samples and builds their libraries."""
    return cheap_analysis(baseline_doc, reaction_library, scores_file, target, os.path.abspath(UQ_CHEAP_FILE),
                          quantiles=quantiles, gate_sd=gate_sd)


@job
def hybrid_uq_calibration_job(cheap_doc: HybridUQCheapDoc, anchor_doc: MultiRxnCAResultDoc,
                              recipe: ReactionRecipe, save_samples: bool = True,
                              metadata: dict = None) -> HybridUQResultDoc:
    """Width calibration of the cheap ensemble through the anchor runs, and the smoothed summaries."""
    doc = calibration(cheap_doc, anchor_doc, recipe,
                      samples_file=os.path.abspath(UQ_SAMPLES_FILE) if save_samples else None, metadata=metadata)
    doc.to_file(UQ_RESULT_FILE)
    return doc


@dataclass
class HybridUQMaker(Maker):
    """
    Hybrid UQ of one recipe + reaction library: the spread of the phase traces when every reaction energy
    carries an error ~ N(0, sigma). The method is in rxn_ca.uq. The calibration belongs to this one condition.

    The flow:
    1. score_energy_samples_job (only without a scores file): scores under n_samples energy samples.
    2. MultiRxnCAMaker: n_baselines unperturbed realizations with full history.
    3. hybrid_uq_cheap_job: cheap ensemble; anchor samples at the quantiles of the target's final fraction
       (none if it barely moves) and their perturbed libraries.
    4. MultiRxnCAMaker: anchor_reps analysis-only realizations of each anchor library.
    5. hybrid_uq_calibration_job: width calibration and summaries -> HybridUQResultDoc.

    Parameters
    ----------
    n_samples (int): energy samples, when scoring in the flow. Default 500.
    sigma (float or dict): energy error per phase in eV/atom, when scoring in the flow. A float applies to every
        solid phase (gases get 0); a dict {phase: sigma} sets the named phases, the rest get 0. Errors are
        independent between phases unless phase_correlation > 0; each reaction's error follows from its
        stoichiometry. Default 0.015.
    phase_correlation (float): correlation between phase errors, 0 to 1, when scoring in the flow. 0: independent;
        1 with equal sigmas: every phase off by the same amount, which cancels in balanced reactions. Default 0.
    scorer_kwargs (dict): extra kwargs for the recipe's scorer class, when scoring in the flow.
    n_baselines (int): unperturbed realizations. Default 5.
    anchor_reps (int): realizations per anchor sample. Default 2.
    quantiles (tuple): quantiles (%) of the target's final cheap fraction where anchors go. Default (2, 25, 75, 98).
    gate_sd (float): no anchors if the target's final cheap fraction has a std below this. Default 0.02.
    num_workers (int): cores for the simulations and the scoring. Default None (all cores on the node).
    memory_per_task_gb (float), cpus_per_task (int): passed to both MultiRxnCAMakers.
    save_samples (bool): write every sample's traces to hybrid_uq_samples.npz. Default True.
    metadata (dict): stored on the result doc.
    """

    name: str = "HybridUQ"
    n_samples: int = 500
    sigma: Union[float, Dict[str, float]] = DEFAULT_SIGMA
    phase_correlation: float = 0.0
    scorer_kwargs: dict = field(default_factory=dict)
    n_baselines: int = 5
    anchor_reps: int = 2
    quantiles: tuple = QUANTILES
    gate_sd: float = GATE_SD
    num_workers: int = None
    memory_per_task_gb: float = None
    cpus_per_task: int = 1
    save_samples: bool = True
    metadata: dict = None

    def make(self, recipe: ReactionRecipe, reaction_library: ReactionLibrary, target: str,
             scores_file: str = None, base_reactions=None) -> Flow:
        """
        recipe, reaction_library: the condition to run.
        target: the phase the anchors are chosen on (usually the recipe's target product).
        scores_file: PerturbedScores npz for this library. Without it, the flow scores the library itself
            from base_reactions (the rxn_network ReactionSet the library was scored from).
        """
        if scores_file is None and base_reactions is None:
            raise ValueError("give either scores_file or base_reactions")
        jobs = []
        if scores_file is None:
            scores = score_energy_samples_job(reaction_library, base_reactions, recipe, n_samples=self.n_samples,
                                              sigma=self.sigma, phase_correlation=self.phase_correlation,
                                              scorer_kwargs=self.scorer_kwargs,
                                              num_workers=self.num_workers)
            jobs.append(self._named(scores, "scores"))
            scores_file = scores.output

        # The cheap estimator needs every per-step diff: no compression, no plots.
        baselines = self._multi("baselines", MultiRxnType.MULTI_RECIPE, compress=False, compute_plots=False).make(
            [baseline_recipe(recipe, self.n_baselines)], reaction_library)
        cheap = self._named(hybrid_uq_cheap_job(baselines.output, reaction_library, scores_file, target,
                                                quantiles=self.quantiles, gate_sd=self.gate_sd), "cheap ensemble")
        # Anchors only need the phase volumes, observed once per sweep.
        anchors = self._multi("anchors", MultiRxnType.MULTI_LIB).make(
            anchor_recipe(recipe, self.anchor_reps), cheap.output.anchor_libraries,
            analysis_only=True, compress_freq=n_sites(recipe))
        calib = self._named(hybrid_uq_calibration_job(cheap.output, anchors.output, recipe,
                                                      save_samples=self.save_samples, metadata=self.metadata),
                            "calibration")
        jobs += [baselines, cheap, anchors, calib]
        return Flow(jobs, output=calib.output, name=self.name)

    def _multi(self, label: str, multi_rxn_type: MultiRxnType, **kwargs) -> MultiRxnCAMaker:
        return MultiRxnCAMaker(name=f"{self.name} {label}", multi_rxn_type=multi_rxn_type,
                               num_workers=self.num_workers, memory_per_task_gb=self.memory_per_task_gb,
                               cpus_per_task=self.cpus_per_task, **kwargs)

    def _named(self, j, label: str):
        j.name = f"{self.name} {label}"
        return j
