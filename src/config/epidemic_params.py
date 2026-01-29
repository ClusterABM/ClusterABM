"""
Epidemic parameters calibrated for R₀ = 2.5 (COVID-19-like pathogen).

This centralizes ALL transmission parameters to ensure consistency
across State Agents, Entity Agents, and Cluster Teams.
"""

import numpy as np
from typing import Dict, Tuple


class EpidemicParams:
    """
    SEIRD model parameters calibrated for realistic epidemic dynamics.
    
    TARGET: R₀ = 2.5 (basic reproductive number without interventions)
    """
    
    # ═══════════════════════════════════════════════════════════════
    # REPRODUCTIVE NUMBER
    # ═══════════════════════════════════════════════════════════════
    R0_TARGET = 2.5
    
    # ═══════════════════════════════════════════════════════════════
    # TIMING PARAMETERS (days)
    # ═══════════════════════════════════════════════════════════════
    LATENT_PERIOD_MEAN = 3.0       # E→I: Incubation period
    INFECTIOUS_PERIOD_MEAN = 10.0   # I→R: Time to recovery
    IMMUNITY_DURATION = 180.0       # R→S: Immunity duration (hard gate)
    
    # ═══════════════════════════════════════════════════════════════
    # TRANSMISSION PARAMETERS (calibrated for R₀ = 2.5)
    # ═══════════════════════════════════════════════════════════════
    
    # Household transmission intensity (exponential SAR model)
    # With 1 infected household member, P(exposure) = 1 - exp(-BETA_HOUSEHOLD)
    BETA_HOUSEHOLD = 0.18  # ⭐ BOOSTED from 0.50 → 80% daily with 1 infected
    
    # Non-household contact transmission (per infected contact)
    BETA_CONTACT = 0.035  # ⭐ BOOSTED from 0.15 → 8% per contact per day
    
    # Community baseline (when no infected contacts)
    BETA_COMMUNITY = 0.005  # 0.5% baseline community transmission
    
    # Occupation multipliers
    OCCUPATION_RISK = {
        'healthcare_worker': 2.0,  # 2x exposure
        'retail_worker': 1.5,
        'teacher': 1.8,
        'office_worker': 1.0,
        'student': 1.2,
        'retired': 0.8
    }
    
    # ═══════════════════════════════════════════════════════════════
    # VACCINATION EFFECTIVENESS
    # ═══════════════════════════════════════════════════════════════
    VACCINE_EFFICACY = {
        0: {'exposure': 1.0, 'progression': 1.0, 'recovery': 1.0, 'death': 1.0},  # Unvaccinated
        1: {'exposure': 0.50, 'progression': 0.85, 'recovery': 1.2, 'death': 0.3},  # Partial
        2: {'exposure': 0.30, 'progression': 0.70, 'recovery': 1.4, 'death': 0.1}   # Full
    }
    
    # ═══════════════════════════════════════════════════════════════
    # AGE-STRATIFIED IFR (Cumulative Infection Fatality Rate)
    # ═══════════════════════════════════════════════════════════════
    IFR_BY_AGE = {
        (0, 20): 0.00002,    # 0.002%
        (20, 50): 0.0005,    # 0.05%
        (50, 70): 0.005,     # 0.5%
        (70, 120): 0.07      # 7%
    }
    
    @classmethod
    def get_ifr(cls, age: int) -> float:
        """Get cumulative IFR for age."""
        for (min_age, max_age), ifr in cls.IFR_BY_AGE.items():
            if min_age <= age < max_age:
                return ifr
        return 0.005
    
    # ═══════════════════════════════════════════════════════════════
    # BASELINE TRANSITION RATES (for State Agent guidance)
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def get_baseline_rate(cls, transition: str) -> float:
        """
        Get baseline daily transition rate (before adjustments).
        
        These guide State Agent LLM prompts.
        """
        rates = {
            'S->E': cls.BETA_CONTACT,  # Per infected contact
            'E->I': 1.0 / cls.LATENT_PERIOD_MEAN,  # ~33% per day
            'I->R': 1.0 / cls.INFECTIOUS_PERIOD_MEAN,  # ~10% per day
            'I->D': 0.005,  # Base death rate (age-adjusted)
            'R->S': 1.0 / cls.IMMUNITY_DURATION  # ~0.5% per day
        }
        return rates.get(transition, 0.0)
    
    # ═══════════════════════════════════════════════════════════════
    # R₀ CALCULATION
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def calculate_R0_from_network(
        cls,
        avg_household_size: float,
        avg_contacts: float
    ) -> float:
        """
        Calculate expected R₀ from network structure.
        
        R₀ = (household transmissions) + (community transmissions)
        
        Args:
            avg_household_size: Average household size
            avg_contacts: Average daily contacts outside household
            
        Returns:
            Expected R₀
        """
        # ⭐ Calculate household infection probability from BETA_HOUSEHOLD
        household_infection_prob = 1.0 - np.exp(-cls.BETA_HOUSEHOLD)
        
        # Household contribution
        household_R0 = (avg_household_size - 1) * household_infection_prob
        
        # Community contribution
        # R₀_community = β × c × D
        community_R0 = cls.BETA_CONTACT * avg_contacts * cls.INFECTIOUS_PERIOD_MEAN
        
        total_R0 = household_R0 + community_R0
        
        return total_R0
    
    @classmethod
    def calibrate_and_report(
        cls,
        avg_household_size: float,
        avg_contacts: float
    ):
        """Calculate and report expected R₀ given network structure."""
        R0 = cls.calculate_R0_from_network(avg_household_size, avg_contacts)  # ⭐ Removed 3rd parameter
        
        # ⭐ Calculate household infection prob for display
        household_infection_prob = 1.0 - np.exp(-cls.BETA_HOUSEHOLD)
        
        print(f"\n{'='*70}")
        print(f"R₀ CALIBRATION REPORT")
        print(f"{'='*70}")
        print(f"Target R₀: {cls.R0_TARGET}")
        print(f"\nNetwork Structure:")
        print(f"  Average household size: {avg_household_size:.1f}")
        print(f"  Average contacts (non-household): {avg_contacts:.1f}")
        print(f"\nTransmission Parameters:")
        print(f"  Household β: {cls.BETA_HOUSEHOLD:.2f} → ~{household_infection_prob*100:.0f}% daily with 1 infected")
        print(f"  Contact β: {cls.BETA_CONTACT:.3f} → {cls.BETA_CONTACT*100:.1f}% per contact")
        print(f"  Infectious period: {cls.INFECTIOUS_PERIOD_MEAN:.0f} days")
        print(f"\nCalculated R₀: {R0:.2f}")
        
        if abs(R0 - cls.R0_TARGET) > 0.3:
            print(f"\n⚠️  WARNING: R₀ mismatch!")
            if R0 < cls.R0_TARGET:
                print(f"   Epidemic will fizzle out (R₀ < {cls.R0_TARGET})")
                print(f"   → Increase BETA_HOUSEHOLD or BETA_CONTACT in epidemic_params.py")
            else:
                print(f"   Epidemic will explode (R₀ > {cls.R0_TARGET})")
                print(f"   → Decrease BETA_HOUSEHOLD or BETA_CONTACT in epidemic_params.py")
        else:
            print(f"\n✓ R₀ is within target range ({cls.R0_TARGET} ± 0.3)")
        
        print(f"{'='*70}\n")
        
        return R0


class R0Tracker:
    """
    Track effective reproductive number (R_eff) during simulation.
    
    R_eff = R₀ × (fraction susceptible)
    
    Also tracks actual secondary infections.
    """
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.infection_tree = {}  # {agent_id: [list of agents they infected]}
        self.infection_source = {}  # {agent_id: source_agent_id}
        self.generation_times = []  # List of (infector_infection_time, infectee_infection_time)
        
    def record_infection(
        self,
        source_agent_id: int,
        infected_agent_id: int,
        timestep: int
    ):
        """Record a transmission event."""
        if source_agent_id not in self.infection_tree:
            self.infection_tree[source_agent_id] = []
        
        self.infection_tree[source_agent_id].append(infected_agent_id)
        self.infection_source[infected_agent_id] = source_agent_id
    
    def calculate_R_eff(self, current_susceptible: int) -> float:
        """
        Calculate current effective reproductive number.
        
        R_eff = R₀ × (S/N)
        """
        fraction_susceptible = current_susceptible / self.population_size
        R_eff = EpidemicParams.R0_TARGET * fraction_susceptible
        return R_eff
    
    def get_secondary_infections(self) -> Dict:
        """
        Calculate observed secondary infections.
        
        Returns distribution of secondary infections per index case.
        """
        if not self.infection_tree:
            return {'mean': 0.0, 'std': 0.0, 'max': 0}
        
        secondary_counts = [len(infections) for infections in self.infection_tree.values()]
        
        return {
            'mean': np.mean(secondary_counts),
            'std': np.std(secondary_counts),
            'max': max(secondary_counts),
            'n_index_cases': len(secondary_counts)
        }
