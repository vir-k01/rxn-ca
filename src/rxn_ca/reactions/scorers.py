import math
from tqdm import tqdm
from abc import ABC, abstractmethod
from .scored_reaction import ScoredReaction

import numpy as np
import pandas as pd
import warnings
import pkg_resources

from ..phases.solid_phase_set import SolidPhaseSet
from pymatgen.core import Composition
from typing import List

from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram
from rxn_network.entries.entry_set import GibbsEntrySet

from ..phases.gasses import DEFAULT_GASES

KB = 8.6173303e-5 # Boltzmann constant in eV/K
Na = 6.02214076e23 # Avogadro's number

def softplus(x):
    return 1/3 * math.log(1 + math.exp(3*x))

def tamman_score_exp(t_tm_ratio):
    return math.exp(4.82*(t_tm_ratio) - 3.21)

def tamman_score_softplus(t_tm_ratio):
    return math.log(1 + math.exp(14 * (t_tm_ratio - 0.8)))

def huttig_score_exp(t_tm_ratio):
    return math.exp(2.41*(t_tm_ratio) - 0.8)

def huttig_score_softplus(t_tm_ratio):
    return 0.25 * math.log(1 + math.exp(30 * (t_tm_ratio - 0.33)))

def erf(x):
    return 0.5 * (1 + math.erf(-35 * (x + 0.03)))

def tamman_erf_score(tm_ratio, delta_g):
    return tamman_score_softplus(tm_ratio) * erf(delta_g)

def huttig_erf_score(tm_ratio, delta_g):
    return huttig_score_softplus(tm_ratio) * erf(delta_g)
         
class BasicScore(ABC):

    def __init__(self, phase_set: SolidPhaseSet, temp: int = None):
        self.phases = phase_set
        self.temp = temp

    @abstractmethod
    def score(self, rxn: ComputedReaction):
        pass

class TammanHuttigScoreExponential(BasicScore):
    # https://en.wikipedia.org/wiki/Tammann_and_H%C3%BCttig_temperatures


    def score(self, rxn: ComputedReaction):
        phases = [c.reduced_formula for c in rxn.reactants]
        non_gasses = [p for p in phases if p not in DEFAULT_GASES]
        mps = [self.phases.get_melting_point(p) for p in non_gasses]
        min_mp = min(mps)

        # Softplus adjustment
        # delta_g_adjustment = softplus(-rxn.energy_per_atom)
        delta_g_adjustment = softplus(-(2*rxn.energy_per_atom + 0.8))
        

        if len(non_gasses) < len(phases):
            # Huttig
            return huttig_score_exp(self.temp / min_mp) * delta_g_adjustment
        else:
            # Tamman
            return tamman_score_exp(self.temp / min_mp) * delta_g_adjustment

class TammanHuttigScoreSoftplus(BasicScore):
    # https://en.wikipedia.org/wiki/Tammann_and_H%C3%BCttig_temperatures


    def score(self, rxn: ComputedReaction):
        phases = [c.reduced_formula for c in rxn.reactants]
        non_gasses = [p for p in phases if p not in DEFAULT_GASES]
        mps = [self.phases.get_melting_point(p) for p in non_gasses]
        min_mp = min(mps)

        # Softplus adjustment
        delta_g_adjustment = softplus(-rxn.energy_per_atom)
        delta_g_adjustment = softplus(-(2*rxn.energy_per_atom + 0.8))

        if len(non_gasses) < len(phases):
            # Huttig
            return huttig_score_softplus(self.temp / min_mp) * delta_g_adjustment
        else:
            # Tamman
            return tamman_score_softplus(self.temp / min_mp) * delta_g_adjustment


class TammanHuttigScoreErf(BasicScore):
    # https://en.wikipedia.org/wiki/Tammann_and_H%C3%BCttig_temperatures

    def score(self, rxn: ComputedReaction):
        phases = [c.reduced_formula for c in rxn.reactants]
        non_gasses = [p for p in phases if p not in self.phases.gas_phases]
        mps = [self.phases.get_melting_point(p) for p in non_gasses]
        min_mp = min(mps)

        # Softplus adjustment
        delta_g_adjustment = erf(rxn.energy_per_atom)

        if len(non_gasses) == 1:
            # Huttig
            return huttig_score_softplus(self.temp / min_mp) * delta_g_adjustment
        else:
            # Tamman
            return tamman_score_softplus(self.temp / min_mp) * delta_g_adjustment

class GibbsErfScore(BasicScore):
    # https://en.wikipedia.org/wiki/Tammann_and_H%C3%BCttig_temperatures

    def score(self, rxn: ComputedReaction):
        return erf(rxn.energy_per_atom)

class TammanScore(BasicScore):
    # https://en.wikipedia.org/wiki/Tammann_and_H%C3%BCttig_temperatures

    def score(self, rxn: ComputedReaction):
        phases = [c.reduced_formula for c in rxn.reactants]
        non_gasses = [p for p in phases if p not in self.phases.gas_phases]
        mps = [self.phases.get_melting_point(p) for p in non_gasses]
        min_mp = min(mps)

        # Softplus adjustment
        delta_g_adjustment = erf(rxn.energy_per_atom)
        return tamman_score_softplus(self.temp / min_mp) * delta_g_adjustment  

class ConstantScore(BasicScore):

    def score(self, _):
        return 1.0

class GibbsErfScore(BasicScore):
        
    def score(self, rxn: ComputedReaction):
        return erf(rxn.energy_per_atom)
    

class TammanTightLinear(BasicScore):

    def score(self, rxn: ComputedReaction):
        phases = [c.reduced_formula for c in rxn.reactants]
        non_gasses = [p for p in phases if p not in self.phases.gas_phases]
        mps = [self.phases.get_melting_point(p) for p in non_gasses]
        min_mp = min(mps)

        def _score(x):
            return 1/2*(1 + math.erf(20*(x -0.6))) * (1/0.6*x)

        # Softplus adjustment
        delta_g_adjustment = erf(rxn.energy_per_atom)
        return _score(self.temp / min_mp) * delta_g_adjustment  

class DiffusionScorer(BasicScore):
    """
    A scorer that uses the diffusion coefficients of the species in the reaction to score the reaction.
    Args:
        phase_set (SolidPhaseSet): The phase set to use for phase information.
        chem_pot_diagram (ChemicalPotentialDiagram): The chemical potential diagram to use for chemical potential information.
        precursor_size (float): The size of the precursor in meters. Default is 1e-7 (0.1 micro-m).
        scale_factor (float): The scale factor to use for the scoring. Default is 5e13. This is used to scale the score to a reasonable range.
                            Modify this to ensure no math range errors. Might have to adjust based on the system.
        chemsys (list[str] | str): The chemical system to use for the diffusion coefficients. Default is None, which uses the chemical system of the chemical potential diagram.
        temp (int): The temperature to use for the scoring.
        self_diffusion (bool): Whether to use self-diffusion coefficients instead of the full transport tensor. Default is False.
    """
    # https://arxiv.org/abs/2501.08560 (Section 4.5)

    def __init__(self, 
                 phase_set: SolidPhaseSet, 
                 entry_set: GibbsEntrySet, 
                 temp : int,
                 precursor_size : float = 1e-8, 
                 scale_factor : float = 5e12, 
                 chemsys : list[str] | str = None, 
                 self_diffusion=False,
                 tamman_factor : float = 1.0,
                 mode : str = 'min'
                 ):
        
        self.phase_set = phase_set
        self.temp = temp
        self.chem_pot_diagram = ChemicalPotentialDiagram(entry_set.get_entries_with_new_temperature(self.temp))
        self.max_flux = 1e15
        self.min_flux = 1e10
        self.chemsys = chemsys if chemsys else self.chem_pot_diagram.chemical_system
        self.precursor_size = precursor_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.tamman_factor = tamman_factor
        self.diff_df = load_diffusivities(chemsys = self.chemsys, self_diffusion=self_diffusion)   # Note: temps in Kelvin
    
    def score(self, reaction: ComputedReaction):
        # This scorer assumes binary reactions (A + B -> C + D + E...), with any number of gas phases, and atmost two solid reactants
        
        dG = reaction.energy_per_atom
        
        if dG > 1.0:
            return 1e-5 # Edge case for reactions with very high energy barriers.
        
        if len(reaction.reactants) > 1:
            reactants_dict = {c.reduced_formula : v for c, v in reaction.reactant_coeffs.items()}
            products_dict = {c.reduced_formula : v for c, v in reaction.product_coeffs.items()}
        else: # Edge case for single reactant, i.e., a decomposition reaction. This is equivalent to the reverse of a formation reaction with the negative of the free energy change
            reactants_dict = {c.reduced_formula : -v for c, v in reaction.product_coeffs.items()}
            products_dict = {c.reduced_formula : -v for c, v in reaction.reactant_coeffs.items()}
        
        species_dict = {k:v for k,v in enumerate(self.chemsys.split('-'))}
        
        if len(self.chem_pot_diagram.chemical_system.split('-')) > 3:
            species = self.chem_pot_diagram.chemical_system.split('-')
            if 'C' in species:
                del_index = species.index('C')
            mu = np.delete(mu_distance(list(reactants_dict.keys()), self.chem_pot_diagram, self.mode), del_index)
        else:
            mu = mu_distance(list(reactants_dict.keys()), self.chem_pot_diagram, self.mode)
        
        if self.chem_pot_diagram.chemical_system != self.chemsys:  # Reorder mu to match the chemical system, i.e., order in which the L_ij values are stored (need better way of handling this)
            returned_order = self.chem_pot_diagram.chemical_system.split('-')
            indices = [returned_order.index(label) for label in self.chemsys.split('-')]
            mu = mu[indices]
        
        all_fluxes = []

        
        mol_normalizer = sum(products_dict.values())
        for p, coeff in products_dict.items():
            el_fraction = {el: 0 for el in self.chem_pot_diagram.chemical_system.split('-')}
            for sp in Composition(p).elements:
                el_fraction.update({sp.symbol: Composition(p).get_atomic_fraction(sp)})
            #if len(el_fraction) < 3:
            #    el_fraction = [el_fraction[0], 0, 0] # Edge case for single element products
            
            vol = self.phase_set.get_vol(p)

            if p not in DEFAULT_GASES:
                melt_pt = self.phase_set.get_melting_point(p)
                if self.temp > melt_pt:
                    fluxes = [self.max_flux, self.max_flux, self.max_flux] # Set fluxes to max if above melting point
                    all_fluxes.append(np.sum([np.abs(fluxes[i]/el_fraction[species_dict[i]]*vol) for i in range(len(fluxes)) if el_fraction[species_dict[i]] > 0]))
            
            fluxes = get_fluxes_across_interface(
                Composition(p).reduced_formula,
                self.diff_df,
                self.temp,
                mu,
            )
            K_d = np.sum([np.abs(fluxes[i]/el_fraction[species_dict[i]]*vol) for i in range(len(fluxes)) if el_fraction[species_dict[i]] > 0])
            all_fluxes.append(K_d * coeff/mol_normalizer) # TODO: check if indices match between Lij data and pmg structure species
        
        selected_flux = np.sum(all_fluxes) # Select the highest flux across all products, assuming the fastest step "pulls" the reaction forward
        phases = [c.reduced_formula for c in reaction.reactants]
        non_gasses = [p for p in phases if p not in DEFAULT_GASES]
        mps = [self.phase_set.get_melting_point(p) for p in non_gasses]
        min_mp = np.min(mps)        
        
        system_factor = 1/self.precursor_size**2/self.scale_factor/Na/KB/self.temp
        
        onset_factor = tamman_score_softplus(self.temp/min_mp*self.tamman_factor)
        #if dG > 1.0:
        #    return onset_factor*softplus(1e-5) # Edge case for reactions with very high energy barriers.
        
        selectivity_factor = selected_flux*system_factor
        selectivity_factor = softplus(selectivity_factor) if selectivity_factor < 2 else selectivity_factor
        score = onset_factor*selectivity_factor*erf(dG)
        return score if score < 20 else 20 # Cap score at 20, anything higher doesnt really help with selectivity/rate

def mu_distance(phases : list, chempot : ChemicalPotentialDiagram, mode : str ='min'):
    
    if len(phases) > 2:
        phases = [p for p in phases if p not in DEFAULT_GASES]
        if len(phases) > 2:
            raise ValueError("Only works for reactions between two solids or a solid and a gas")
    
     # Get the domains for the two phases
    domains = []
    for phase in phases:
        if phase in list(chempot.domains.keys()):
            domains.append(chempot.domains[phase])
        else:
            domains.append(chempot.metastable_domains[phase])

    if mode == 'max':
        max_distance = 0
        ind_1, ind_2 = 0, 0
        for node1 in range(len(domains[0])):
            for node2 in range(len(domains[1])):
                distance = np.linalg.norm(domains[0][node1] - domains[1][node2])
                if distance > max_distance:
                    max_distance = distance
                    ind_1, ind_2 = node1, node2
        return domains[0][ind_1] - domains[1][ind_2] 
    
    if mode == 'min':
        min_distance = 1000
        ind_1, ind_2 = 0, 0
        for node1 in range(len(domains[0])):
            for node2 in range(len(domains[1])):
                distance = np.linalg.norm(domains[0][node1] - domains[1][node2])
                if distance < min_distance:
                    min_distance = distance
                    ind_1, ind_2 = node1, node2
        return domains[0][ind_1] - domains[1][ind_2]
    
    if mode == 'mean':
        return np.mean(domains[0], axis=0) - np.mean(domains[1], axis=0)

def get_fluxes_across_interface(product : str, L_data : pd.DataFrame, temperature : int, mu : np.array):
    for t in list(set(L_data['Temperature'])):
        if isinstance(t, str):
            t = unstringify_temp(t)
    temps = list(set(L_data['Temperature']))
    diffs = np.abs(np.array(temps) - temperature)
    closest_temp_idx = np.argmin(diffs)
    closest_temp = temps[closest_temp_idx]
    formula_data =  L_data[(L_data['Formula'] == product) & (L_data['Temperature'] == closest_temp)]
    # Extract L_ij values
    
    L_ij_values = np.eye(len(mu))*1e13
    
    for i in range(len(mu)):
        for j in range(i, len(mu)):
            try:
                L_ij_values[i][j] = formula_data[f'L{i}{j}'].values[0]
                L_ij_values[j][i] = formula_data[f'L{i}{j}'].values[0]
            except IndexError as e:
                warnings.warn(f'No L_ij data for {product} at {temperature}K, setting to k_B*T')
                return np.eye(len(mu))*KB*temperature*1e14*len(mu)
    return np.dot(L_ij_values, mu)

def load_diffusivities(chemsys : list = None, self_diffusion : bool = False):
    if chemsys in ['Ba-Ti-O', 'Ba-Ti-O-C']:
        if self_diffusion:
            try:
                diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "batio_self_coeffs.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Transport data is not available yet. Please contact the authors if you would like to use this feature!")
        else:
            try:
                diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "batio_transport_coeffs.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Transport data is not available yet. Please contact the authors if you would like to use this feature!")
            
    if chemsys in ['Fe-Bi-O', 'Bi-Fe-O', 'Bi-O-Fe']:
            if self_diffusion:
                diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "bifeo_self_coeffs.csv")
            else:
                diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "bifeo_transport_coeffs.csv")

    if chemsys in ["Li-Ti-O", "Li-O-Ti", "Ti-Li-O"]:
        if self_diffusion:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "litio_self_coeffs.csv")
        else:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "litio_transport_coeffs.csv")
        
    if chemsys in ["Ba-Fe-O", "Fe-Ba-O", "Ba-O-Fe"]:
        if self_diffusion:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "Ba-Fe-O-self_coeffs.csv")
        else:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "Ba-Fe-O-transport.csv")
    
    if chemsys in ["Y-Fe-O", "Fe-Y-O", "Y-O-Fe"]:
        if self_diffusion:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "Y-Fe-O-self_coeffs.csv")
        else:
            diff_csv_path = pkg_resources.resource_filename("rxn_ca.phases", "Y-Fe-O-transport.csv")
    df = pd.read_csv(diff_csv_path)
    if df.isna().any().any():
        df.fillna(0, inplace=True)
    return df

def stringify_temp(temp):
    return f"{temp}.0K"

def unstringify_temp(temp_str):
    return int(temp_str.split('.')[0])

def score_rxns(reactions: ReactionSet, scorer: BasicScore, phase_set: SolidPhaseSet = None):
    scored_reactions = []

    for rxn in tqdm(reactions.get_rxns(), desc=f"Scoring reactions... at temp {scorer.temp}"):
        reactants = [r.reduced_formula for r in rxn.reactants]
        non_gases = [r for r in reactants if r not in phase_set.gas_phases]
        if len(non_gases) > 0:
            scored_rxn = ScoredReaction.from_rxn_network(scorer.score(rxn), rxn, phase_set.volumes)
            scored_reactions.append(scored_rxn)

    return scored_reactions