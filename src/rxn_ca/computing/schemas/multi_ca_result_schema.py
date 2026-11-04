from typing import List, Optional

from ...core.recipe import ReactionRecipe
from ...computing.schemas.ca_result_schema import RxnCAResultDoc
from ...analysis.visualization.reaction_plotter import ReactionPlotter
from ...analysis.bulk_reaction_analyzer import BulkReactionAnalyzer
from pylattica.core import Simulation
from ...utilities.viz import get_plotted_data

from .base_schema import BaseSchema
from dataclasses import dataclass

@dataclass
class MultiRxnCAResultDoc(BaseSchema):
    recipes: List[ReactionRecipe]
    result_docs: Optional[List[RxnCAResultDoc]]
    mass_fraction_plots: Optional[List[dict]]
    molar_fraction_plots: Optional[List[dict]]
    molar_amount_plots: Optional[List[dict]]
    phase_volume_plots: Optional[List[dict]]
    phase_mass_plots: Optional[List[dict]]
    elemental_amount_plots: Optional[List[dict]]
    elemental_fraction_plots: Optional[List[dict]]
    final_simulations: Optional[List[Simulation]]
    have_simulations_converged: Optional[List[bool]]
    metadata: dict = None
    run_dir: str = None
    
    @classmethod
    def from_multiple_jobs(cls, result_docs : List[RxnCAResultDoc], 
                           metadata : Optional[dict] = None, 
                           save_results_to_store : bool = False,
                           run_dir: str = None,
                           **kwargs):

        reaction_plotter_kwargs = kwargs.get("reaction_plotter_kwargs", {})
        recipes = [rd.recipe for rd in result_docs]
        plots = {k: None for k in ("mass_fraction_plots", "molar_fraction_plots", "molar_amount_plots",
                                   "phase_volume_plots", "phase_mass_plots", "elemental_amount_plots",
                                   "elemental_fraction_plots")}
        have_simulations_converged = None
        # compute_plots=False skips the plot data and convergence flags (both load every analyzed step,
        # which is slow for full-history results)
        if kwargs.get("compute_plots", True):
            analyzers = [BulkReactionAnalyzer.from_result_doc(rd) for rd in result_docs]
            plotter_objects = [ReactionPlotter(analyzer, **reaction_plotter_kwargs) for analyzer in analyzers]

            plots["mass_fraction_plots"] = [get_plotted_data(p.plot_mass_fractions()) for p in plotter_objects]
            plots["molar_fraction_plots"] = [get_plotted_data(p.plot_molar_phase_fractions()) for p in plotter_objects]
            plots["molar_amount_plots"] = [get_plotted_data(p.plot_molar_phase_amounts()) for p in plotter_objects]
            plots["phase_volume_plots"] = [get_plotted_data(p.plot_phase_volumes()) for p in plotter_objects]
            plots["phase_mass_plots"] = [get_plotted_data(p.plot_phase_masses()) for p in plotter_objects]
            plots["elemental_amount_plots"] = [get_plotted_data(p.plot_elemental_amounts()) for p in plotter_objects]
            plots["elemental_fraction_plots"] = [get_plotted_data(p.plot_elemental_fractions()) for p in plotter_objects]

            have_simulations_converged = [analyzer.has_simulation_converged() for analyzer in analyzers]
        
        final_simulations = [rd.final_simulation for rd in result_docs]
        
        return cls(recipes=recipes,
                   result_docs=result_docs if save_results_to_store else None,
                   **plots,
                   final_simulations=final_simulations,
                   have_simulations_converged=have_simulations_converged,
                   metadata=metadata,
                   run_dir=run_dir)