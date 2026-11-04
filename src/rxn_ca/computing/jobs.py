from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from ..computing.schemas.multi_ca_result_schema import MultiRxnCAResultDoc
from ..computing.schemas.ca_result_schema import RxnCAResultDoc
from ..utilities.parallel_sim import run_multi_recipe_parallel_ray
from pylattica.core import Simulation
from jobflow import job, Flow, Maker
from dataclasses import dataclass, field
from typing import List
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

@job
def score_energy_samples_job(reaction_library: ReactionLibrary, base_reactions, recipe: ReactionRecipe,
                             n_samples: int = 500, sigma: float = 0.015, scorer_kwargs: dict = None,
                             num_workers: int = None) -> str:
    """Score the library under n_samples seeded energy-error samples. Returns the path of the scores npz."""
    from ..utilities.uq import score_energy_samples
    return score_energy_samples(reaction_library, base_reactions, recipe.get_score_class(),
                                os.path.join(os.getcwd(), "uq_scores.npz"), n_samples=n_samples, sigma=sigma,
                                scorer_kwargs=scorer_kwargs, num_workers=num_workers)


@job(data=[ReactionLibrary])
def hybrid_uq_cheap_job(baseline_doc: MultiRxnCAResultDoc, reaction_library: ReactionLibrary, scores_file: str,
                        target: str, quantiles: tuple = (2, 25, 75, 98), gate_sd: float = 0.02):
    """Cheap ensemble from the baseline runs; picks the anchor samples and builds their libraries."""
    from ..utilities.uq import cheap_analysis
    return cheap_analysis(baseline_doc, reaction_library, scores_file, target,
                          os.path.join(os.getcwd(), "uq_cheap.npz"), quantiles=quantiles, gate_sd=gate_sd)


@job
def hybrid_uq_calibration_job(cheap_doc, anchor_doc: MultiRxnCAResultDoc, recipe: ReactionRecipe,
                              save_samples: bool = True, metadata: dict = None):
    """Width calibration of the cheap ensemble through the anchor runs; smoothed summaries."""
    from ..utilities.uq import calibration
    doc = calibration(cheap_doc, anchor_doc, recipe,
                      samples_file=os.path.join(os.getcwd(), "hybrid_uq_samples.npz") if save_samples else None,
                      metadata=metadata)
    doc.to_file("hybrid_uq_result.json")
    return doc


@dataclass
class HybridUQMaker(Maker):
    """
    Flow for the hybrid UQ of one recipe + reaction library: how the phase traces spread when every reaction
    energy carries an error ~ N(0, sigma). Method in rxn_ca.uq; the calibration belongs to this one condition.

    Jobs, in order:
    1. (only if no scores file is given) score_energy_samples_job: scores under n_samples energy samples.
    2. MultiRxnCAMaker (Ray): n_baselines unperturbed full-history realizations.
    3. hybrid_uq_cheap_job: cheap ensemble for every sample; picks anchor samples at the quantiles of the
       target's final cheap fraction (none if it barely moves) and builds their perturbed libraries.
    4. MultiRxnCAMaker (Ray): anchor_reps analysis-only realizations of each anchor library.
    5. hybrid_uq_calibration_job: width calibration and smoothed summaries -> HybridUQResultDoc.

    Parameters
    ----------
    n_samples (int): energy samples when scoring here. Default 500.
    sigma (float): energy error per reaction (eV/atom) when scoring here. Default 0.015.
    n_baselines (int): unperturbed full-history realizations. Default 5.
    anchor_reps (int): realizations per anchor sample. Default 2.
    quantiles (tuple): quantiles (%) where anchors are placed. Default (2, 25, 75, 98).
    gate_sd (float): no anchors if the target's final cheap fraction has a std below this. Default 0.02.
    scorer_kwargs (dict): extra kwargs for the recipe's scorer class when scoring here.
    num_workers, memory_per_task_gb, cpus_per_task: passed to both MultiRxnCAMakers (and the scoring job).
    save_samples (bool): write hybrid_uq_samples.npz (every sample's traces) in the calibration job's directory.
    metadata (dict): stored on the result doc.
    """

    name: str = "HybridUQ"
    n_samples: int = 500
    sigma: float = 0.015
    n_baselines: int = 5
    anchor_reps: int = 2
    quantiles: tuple = (2, 25, 75, 98)
    gate_sd: float = 0.02
    scorer_kwargs: dict = field(default_factory=dict)
    num_workers: int = None
    memory_per_task_gb: float = None
    cpus_per_task: int = 1
    save_samples: bool = True
    metadata: dict = None

    def make(self, recipe: ReactionRecipe, reaction_library: ReactionLibrary, target: str,
             scores_file: str = None, base_reactions=None) -> Flow:
        """
        recipe: the recipe to run. reaction_library: the scored library it runs with.
        target: the phase the anchors are chosen on (usually the recipe's target product).
        scores_file: a PerturbedScores npz for this library (rxn_ca.uq.perturbed_scores). If None, the scores
            are computed in the flow from base_reactions with the recipe's scorer class.
        base_reactions: the rxn_network ReactionSet the library was scored from (needed without scores_file).
        """
        from ..utilities.uq import anchor_recipe, baseline_recipe, n_sites

        jobs = []
        ray_kw = dict(num_workers=self.num_workers, memory_per_task_gb=self.memory_per_task_gb,
                      cpus_per_task=self.cpus_per_task)
        if scores_file is None:
            if base_reactions is None:
                raise ValueError("give either scores_file or base_reactions")
            score = score_energy_samples_job(reaction_library, base_reactions, recipe, n_samples=self.n_samples,
                                             sigma=self.sigma, scorer_kwargs=self.scorer_kwargs,
                                             num_workers=self.num_workers)
            score.name = f"{self.name} scores"
            jobs.append(score)
            scores_file = score.output

        # full per-step history is needed for the event records: no compression, no plots
        baselines = MultiRxnCAMaker(name=f"{self.name} baselines", multi_rxn_type=MultiRxnType.MULTI_RECIPE,
                                    compress=False, compute_plots=False, **ray_kw).make(
            [baseline_recipe(recipe, self.n_baselines)], reaction_library)
        cheap = hybrid_uq_cheap_job(baselines.output, reaction_library, scores_file, target,
                                    quantiles=self.quantiles, gate_sd=self.gate_sd)
        cheap.name = f"{self.name} cheap ensemble"
        # one observation per sweep, analysis-only
        anchors = MultiRxnCAMaker(name=f"{self.name} anchors", multi_rxn_type=MultiRxnType.MULTI_LIB,
                                  **ray_kw).make(
            anchor_recipe(recipe, self.anchor_reps), cheap.output.anchor_libraries,
            analysis_only=True, compress_freq=n_sites(recipe))
        calib = hybrid_uq_calibration_job(cheap.output, anchors.output, recipe, save_samples=self.save_samples,
                                          metadata=self.metadata)
        calib.name = f"{self.name} calibration"
        jobs += [baselines, cheap, anchors, calib]
        return Flow(jobs, output=calib.output, name=self.name)
