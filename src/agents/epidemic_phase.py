"""
Global epidemic phase tracker for temporal awareness.
All agents can query this to understand epidemic dynamics.
"""

import numpy as np
from typing import Dict, List


class EpidemicPhaseTracker:
    """
    Track global epidemic phase and provide context to agents.
    
    Phases:
    - PRE_EPIDEMIC: Few infections, sporadic transmission
    - EXPONENTIAL_GROWTH: R_eff > 1, infections doubling
    - PEAK: Infections plateauing
    - DECLINE: R_eff < 1, infections falling
    - ENDEMIC: Low baseline transmission
    """
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.history = []  # [(timestep, state_counts)]
        self.current_phase = "PRE_EPIDEMIC"
        
    def update(self, timestep: int, state_counts: Dict[str, int]):
        """Update with current state counts."""
        self.history.append((timestep, state_counts.copy()))
        
        # Keep last 14 days only
        if len(self.history) > 14:
            self.history = self.history[-14:]
        
        # Detect phase
        self.current_phase = self._detect_phase()
    
    def _detect_phase(self) -> str:
        """Detect current epidemic phase."""
        if len(self.history) < 3:
            return "PRE_EPIDEMIC"
        
        recent = self.history[-3:]
        
        # Get active infections (E + I)
        active_infections = [
            s['E'] + s['I'] 
            for _, s in recent
        ]
        
        current_active = active_infections[-1]
        prev_active = active_infections[0]
        
        prevalence = current_active / self.population_size
        
        # Phase detection logic
        if prevalence < 0.01:  # <1% infected
            return "PRE_EPIDEMIC"
        
        elif prevalence < 0.05:  # 1-5% infected
            # Check if growing
            if current_active > prev_active * 1.5:  # 50% growth in 3 days
                return "EXPONENTIAL_GROWTH"
            else:
                return "PRE_EPIDEMIC"
        
        elif prevalence < 0.20:  # 5-20% infected
            # Check growth rate
            if current_active > prev_active * 1.2:  # Still growing
                return "EXPONENTIAL_GROWTH"
            elif current_active < prev_active * 0.8:  # Declining
                return "DECLINE"
            else:
                return "PEAK"
        
        elif prevalence < 0.40:  # 20-40% infected
            # Likely at peak or declining
            if current_active > prev_active * 1.1:  # Slow growth
                return "PEAK"
            else:
                return "DECLINE"
        
        else:  # >40% infected
            # Past peak
            return "DECLINE"
    
    def get_phase_multipliers(self) -> Dict[str, float]:
        """
        Get transmission multipliers based on epidemic phase.
        
        Returns:
            Dict with multipliers for each transition
        """
        multipliers = {
            "PRE_EPIDEMIC": {
                'S->E': 1.0,   # Normal
                'E->I': 1.0,
                'I->R': 1.0,
            },
            "EXPONENTIAL_GROWTH": {
                'S->E': 3.0,   # ⭐ 3x transmission during growth!
                'E->I': 1.2,   # Faster progression
                'I->R': 0.9,   # Slightly slower recovery
            },
            "PEAK": {
                'S->E': 2.5,   # Still high transmission
                'E->I': 1.1,
                'I->R': 1.0,
            },
            "DECLINE": {
                'S->E': 1.5,   # Declining but still elevated
                'E->I': 1.0,
                'I->R': 1.2,   # Faster recovery
            },
            "ENDEMIC": {
                'S->E': 0.8,   # Low baseline
                'E->I': 1.0,
                'I->R': 1.1,
            }
        }
        
        return multipliers.get(self.current_phase, multipliers["PRE_EPIDEMIC"])
    
    def get_context_string(self) -> str:
        """Get human-readable context for LLM prompts."""
        if len(self.history) == 0:
            return "Epidemic phase: Unknown (no data yet)"
        
        recent = self.history[-1]
        timestep, state_counts = recent
        
        active = state_counts['E'] + state_counts['I']
        prevalence = active / self.population_size
        
        phase_descriptions = {
            "PRE_EPIDEMIC": f"Early phase - sporadic infections ({prevalence:.1%} prevalence)",
            "EXPONENTIAL_GROWTH": f"⚠️ EXPONENTIAL GROWTH - rapid spread ({prevalence:.1%} prevalence, R_eff > 1)",
            "PEAK": f"Peak phase - infections plateauing ({prevalence:.1%} prevalence)",
            "DECLINE": f"Declining phase - epidemic subsiding ({prevalence:.1%} prevalence, R_eff < 1)",
            "ENDEMIC": f"Endemic phase - stable low transmission ({prevalence:.1%} prevalence)"
        }
        
        # Check growth rate
        if len(self.history) >= 3:
            prev = self.history[-3][1]
            prev_active = prev['E'] + prev['I']
            growth_rate = (active - prev_active) / max(prev_active, 1)
            
            return f"{phase_descriptions[self.current_phase]} | 3-day growth: {growth_rate:+.1%}"
        
        return phase_descriptions[self.current_phase]
