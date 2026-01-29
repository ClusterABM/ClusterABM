"""
State Agent: LLM-based expert for specific epidemic state (S/E/I/R/D).
Has memory, uses tools, reasons about cluster-level transitions.

UPDATED FOR SEIRD MODEL + SINGAPORE COVID-19:
- S (Susceptible): Predicts S→E (with dormitory/quarantine awareness)
- E (Exposed): Predicts E→I
- I (Infected): Predicts I→R AND I→D (Singapore healthcare adjustment)
- R (Recovered): Predicts R→S
- D (Dead): Observes network mourning effects

SINGAPORE-SPECIFIC FEATURES:
- ✓ Dormitory cluster detection (2.5x transmission multiplier)
- ✓ Quarantine enforcement (80% reduction when active)
- ✓ Healthcare quality factor (0.3x global IFR)
- ✓ Cluster type awareness (household/workplace/dormitory)
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os
import signal
import numpy as np
# Import the rate limiter from entity_agent
from src.agents.entity_agent import _llm_rate_limiter

from src.agents.memory_stream import MemoryStream
from src.agents.epidemic_tools import EpidemicToolRegistry
from src.agents.epidemic_phase import EpidemicPhaseTracker
from src.config.epidemic_params import EpidemicParams

def _timeout_handler(signum, frame):
    raise TimeoutError("StateAgent LLM call timed out")

import signal
from contextlib import contextmanager

@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("LLM call timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class StateAgent:
    """
    LLM-based State Agent specializing in one epidemic state.
    
    Each cluster has 5 State Agents (S, E, I, R, D).
    They are full LLM agents with memory and tools.
    
    SPECIAL: I State Agent handles BOTH I→R (recovery) AND I→D (death).
    
    NOTE ON COMPETING RISKS:
    - State Agent returns separate p(I→R) and p(I→D) 
    - Entity agents normalize these as competing risks
    - This prevents recovery/death double-counting
    
    SINGAPORE ENHANCEMENTS:
    - Dormitory cluster detection (high-density transmission)
    - Quarantine order enforcement (reduces transmission)
    - Healthcare quality adjustment (lower mortality)
    """
    
    def __init__(
        self,
        cluster_id: int,
        state: str,  # 'S', 'E', 'I', 'R', or 'D'
        cluster_agents: List,
        tool_registry: EpidemicToolRegistry,
        llm_client: Optional[OpenAI] = None,
        phase_tracker: Optional[EpidemicPhaseTracker] = None
    ):
        """
        Initialize State Agent.
        
        Args:
            cluster_id: Cluster identifier
            state: State specialty ('S', 'E', 'I', 'R', or 'D')
            cluster_agents: List of EntityAgent objects in cluster
            tool_registry: Tool registry
            llm_client: OpenAI client
            phase_tracker: Epidemic phase tracker
        """
        self.cluster_id = cluster_id
        self.state = state
        self.cluster_agents = cluster_agents
        self.tools = tool_registry
        if isinstance(llm_client, tuple):
            self.llm_client = llm_client[0]
        else:
            self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.phase_tracker = phase_tracker
        
        # Memory system
        agent_name = f"StateAgent_{state}_Cluster{cluster_id}"
        self.memory = MemoryStream(
            agent_id=f"{cluster_id}_{state}",
            agent_name=agent_name,
            llm_client=self.llm_client
        )
        
        # Initialize with specialty knowledge
        self._initialize_agent()
    
    def _compress_text(self, text: str, max_len: int = 200) -> str:
        """Compress text for memory storage."""
        return text.replace("\n", " ")[:max_len]
    
    def _initialize_agent(self):
        """Initialize with background."""
        specialty_knowledge = {
            'S': "I am an expert on susceptible populations and infection risk assessment. I analyze transmission dynamics including Singapore's dormitory outbreaks and quarantine measures. I predict S→E transitions (exposure).",
            'E': "I am an expert on exposed populations and disease progression. I analyze incubation periods and predict E→I transitions (symptom onset).",
            'I': "I am an expert on infected populations, recovery, and mortality. I analyze recovery timelines (I→R) and death risk (I→D) accounting for Singapore's excellent healthcare system.",
            'R': "I am an expert on recovered populations and immunity. I analyze reinfection risk and predict R→S transitions.",
            'D': "I am an expert on deceased individuals and their network impact. I monitor psychological effects on surviving neighbors and mourning periods."
        }
        
        background = f"""I am State Agent {self.state} for Cluster {self.cluster_id}.
{specialty_knowledge.get(self.state, 'Expert on epidemic dynamics.')}
I monitor {len([a for a in self.cluster_agents if a.state == self.state])} agents currently in {self.state} state."""
        
        self.memory.add(background, importance=8.0, memory_type="reflection")
    
    def _is_dormitory_cluster(self) -> bool:
        """
        Detect if this is a dormitory cluster (Singapore-specific).
        
        Dormitory clusters have:
        - High proportion of manual_workers
        - Agents with 'dorm' in cluster field
        - Male-dominated demographics
        - Young adult age distribution
        
        Returns:
            True if dormitory cluster detected
        """
        if not self.cluster_agents:
            return False
        
        # Check if agents have dormitory cluster assignment
        dorm_agents = sum(1 for a in self.cluster_agents 
                         if 'dorm' in str(a.profile.get('cluster', '')).lower())
        
        if dorm_agents > len(self.cluster_agents) * 0.3:  # >30% in dormitory
            return True
        
        # Check occupation (manual workers)
        manual_workers = sum(1 for a in self.cluster_agents 
                            if a.profile.get('occupation') == 'manual_worker')
        
        if manual_workers > len(self.cluster_agents) * 0.5:  # >50% manual workers
            return True
        
        return False
    
    def _get_quarantine_reduction(self) -> float:
        """
        Calculate transmission reduction from quarantine enforcement (Singapore-specific).
        
        Singapore had:
        - Strict Quarantine Orders (QO) for contacts
        - Electronic GPS monitoring
        - High compliance rates (~95%)
        - Severe penalties for violations
        
        Returns:
            Multiplier (< 1.0 means reduced transmission)
        """
        if not self.cluster_agents:
            return 1.0
        
        # Count agents under quarantine
        quarantined = sum(1 for a in self.cluster_agents 
                         if a.profile.get('quarantine_status', False))
        
        if quarantined == 0:
            return 1.0
        
        quarantine_rate = quarantined / len(self.cluster_agents)
        
        # Singapore QO compliance: ~95% effective
        # Reduces transmission by 80% for quarantined agents
        reduction = 1.0 - (quarantine_rate * 0.80)
        
        return max(0.1, reduction)  # Minimum 10% (some transmission still possible)
    
    def _get_cluster_age_death_factor(self) -> float:
        """
        Calculate age-weighted death risk factor for cluster.
        
        Returns weighted average IFR multiplier based on age distribution
        of infected agents in the cluster.
        
        Returns:
            Multiplier for baseline death rate (1.0 = average, >1.0 = higher risk)
        """
        if self.state != 'I' or not self.cluster_agents:
            return 1.0
        
        infected_agents = [a for a in self.cluster_agents if a.state == 'I']
        if not infected_agents:
            return 1.0
        
        total_factor = 0.0
        for agent in infected_agents:
            age = agent.profile.get('age', 50)
            
            # Age-stratified IFR multipliers (relative to baseline)
            # Based on Singapore COVID-19 data
            if age < 20:
                factor = 0.01  # Very low (CFR ~0%)
            elif age < 50:
                factor = 0.25  # Low (CFR ~0.01-0.05%)
            elif age < 70:
                factor = 2.5   # Moderate (CFR ~0.5-2%)
            else:
                factor = 35.0  # High (CFR ~7-12% for 70+)
            
            # Adjust for comorbidities
            comorbidities = agent.profile.get('comorbidity_count', 0)
            if comorbidities >= 2:
                factor *= 1.8  # 80% increase for 2+ comorbidities
            elif comorbidities == 1:
                factor *= 1.3  # 30% increase for 1 comorbidity
            
            # Adjust for vaccination
            vacc_status = agent.profile.get('vaccination_status', 0)
            if vacc_status >= 2:
                factor *= 0.1  # 90% protection (fully vaccinated)
            elif vacc_status == 1:
                factor *= 0.3  # 70% protection (partially vaccinated)
            
            total_factor += factor
        
        return total_factor / len(infected_agents)
    
    def analyze_cluster_state(self, timestep: int) -> Dict:
        """
        Analyze current cluster state distribution and patterns.
        
        Args:
            timestep: Current timestep
            
        Returns:
            Analysis dict with observations
        """
        # Get state distribution (SEIRD - 5 states)
        state_dist = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.cluster_agents:
            state_dist[agent.state] += 1
        
        # Get agents in my specialty state
        my_agents = [a for a in self.cluster_agents if a.state == self.state]
        
        # Analyze patterns
        analysis = {
            'total_in_state': len(my_agents),
            'cluster_size': len(self.cluster_agents),
            'state_distribution': state_dist,
            'observations': []
        }
        
        if self.state == 'S':
            # Analyze susceptible agents
            infected_contacts = []
            for agent in my_agents:
                infected_neighbors = [n for n in agent.neighbors if n.state in ['E', 'I']]
                if infected_neighbors:
                    infected_contacts.append(len(infected_neighbors))
            
            if infected_contacts:
                avg_infected_contacts = sum(infected_contacts) / len(infected_contacts)
                analysis['observations'].append(
                    f"{len(infected_contacts)} susceptible agents have exposed/infected contacts (avg: {avg_infected_contacts:.1f})"
                )
                analysis['mean_infected_contacts'] = avg_infected_contacts
            else:
                analysis['mean_infected_contacts'] = 0.0
            
            # Check vaccination rates
            vacc_dist = {0: 0, 1: 0, 2: 0}
            for agent in my_agents:
                vacc_dist[agent.profile['vaccination_status']] += 1
            
            analysis['observations'].append(
                f"Vaccination: {vacc_dist[2]} fully, {vacc_dist[1]} partial, {vacc_dist[0]} unvaccinated"
            )
            
            # Singapore-specific: Check quarantine status
            quarantined = sum(1 for a in my_agents if a.profile.get('quarantine_status', False))
            if quarantined > 0:
                analysis['observations'].append(
                    f"{quarantined} susceptible agents under Quarantine Order (QO)"
                )
        
        elif self.state == 'E':
            # Analyze exposed agents
            days_exposed = [agent.days_in_state for agent in my_agents]
            if days_exposed:
                analysis['observations'].append(
                    f"Exposed agents: days since exposure range {min(days_exposed)}-{max(days_exposed)}, avg {sum(days_exposed)/len(days_exposed):.1f}"
                )
                analysis['avg_days_exposed'] = sum(days_exposed) / len(days_exposed)
            else:
                analysis['avg_days_exposed'] = 0.0
            
            # Check who's likely to develop symptoms soon (days 2-5)
            likely_symptomatic = [a for a in my_agents if 2 <= a.days_in_state <= 5]
            if likely_symptomatic:
                analysis['observations'].append(
                    f"{len(likely_symptomatic)} agents in peak incubation (days 2-5, symptoms likely soon)"
                )
        
        elif self.state == 'I':
            # Analyze infected agents - UPDATED WITH DEATH RISK ANALYSIS
            days_infected = [agent.days_in_state for agent in my_agents]
            if days_infected:
                analysis['observations'].append(
                    f"Infected agents: days infected range {min(days_infected)}-{max(days_infected)}, avg {sum(days_infected)/len(days_infected):.1f}"
                )
                analysis['avg_days_infected'] = sum(days_infected) / len(days_infected)
            else:
                analysis['avg_days_infected'] = 0.0
            
            # Check who's overdue for recovery
            overdue = [a for a in my_agents if a.days_in_state >= 10]
            if overdue:
                analysis['observations'].append(
                    f"{len(overdue)} agents overdue for recovery (>10 days infected)"
                )
            
            # Check elderly at HIGH RISK of death
            elderly_infected = [a for a in my_agents if a.profile['age'] >= 70]
            if elderly_infected:
                avg_age_elderly = sum(a.profile['age'] for a in elderly_infected) / len(elderly_infected)
                analysis['observations'].append(
                    f"{len(elderly_infected)} elderly infected (age 70+, avg {avg_age_elderly:.0f}y, Singapore IFR ~2%)"
                )
            
            # Check middle-aged at moderate risk
            middle_aged = [a for a in my_agents if 50 <= a.profile['age'] < 70]
            if middle_aged:
                analysis['observations'].append(
                    f"{len(middle_aged)} middle-aged infected (age 50-69, Singapore IFR ~0.2%)"
                )
            
            # Check peak death risk period (days 7-14)
            peak_death_risk = [a for a in my_agents if 7 <= a.days_in_state <= 14]
            if peak_death_risk:
                analysis['observations'].append(
                    f"{len(peak_death_risk)} in peak death risk period (days 7-14)"
                )
            
            # Check comorbidity burden
            high_comorbidity = [a for a in my_agents if a.profile.get('comorbidity_count', 0) >= 2]
            if high_comorbidity:
                analysis['observations'].append(
                    f"{len(high_comorbidity)} with 2+ comorbidities (higher death risk)"
                )
        
        elif self.state == 'R':
            # Analyze recovered agents
            days_recovered = [agent.days_in_state for agent in my_agents]
            if days_recovered:
                analysis['observations'].append(
                    f"Recovered agents: days since recovery range {min(days_recovered)}-{max(days_recovered)}"
                )
            
            # Check immunity waning
            waning = [a for a in my_agents if a.days_in_state >= 180]
            if waning:
                analysis['observations'].append(
                    f"{len(waning)} agents with waning immunity (>6 months recovered)"
                )
        
        elif self.state == 'D':
            # Analyze dead agents (for network effects)
            if my_agents:
                days_since_death = [agent.days_since_death for agent in my_agents if hasattr(agent, 'days_since_death')]
                if days_since_death:
                    analysis['observations'].append(
                        f"Deceased agents: {len(days_since_death)} deaths, days since death range {min(days_since_death)}-{max(days_since_death)}"
                    )
                
                # Count agents in mourning period (removed after 7-10 days)
                in_mourning = [a for a in my_agents if hasattr(a, 'days_since_death') and a.days_since_death <= 10]
                if in_mourning:
                    analysis['observations'].append(
                        f"{len(in_mourning)} recently deceased still in network (affecting neighbors psychologically)"
                    )
                
                # Age distribution of deceased
                ages_deceased = [a.profile['age'] for a in my_agents]
                if ages_deceased:
                    avg_age = sum(ages_deceased) / len(ages_deceased)
                    analysis['observations'].append(
                        f"Average age of deceased: {avg_age:.1f} years"
                    )
        
        # Store in memory
        obs_text = f"Timestep {timestep}: {analysis['total_in_state']}/{analysis['cluster_size']} agents in {self.state} state. " + " ".join(analysis['observations'])
        
        self.memory.add(self._compress_text(obs_text), importance=6.0, memory_type="state_summary")
        
        return analysis
    
    def predict_transition_probability(
        self,
        timestep: int,
        neighbor_cluster_context: str,
        transition_type: Optional[str] = None
    ) -> Dict:
        """
        Predict cluster-level transition probability using tools and reasoning.
        
        UPDATED: I State Agent can predict BOTH I→R (recovery) AND I→D (death).
        
        NOTE: For competing risks (I→R vs I→D), normalization happens at entity level.
        State Agent returns raw hazard probabilities.
        
        Args:
            timestep: Current timestep
            neighbor_cluster_context: Context from neighboring clusters
            transition_type: Specific transition to predict (e.g., 'I->R' or 'I->D')
                           If None, uses default for this state
            
        Returns:
            Dict with probability and reasoning
        """
        # Analyze current state
        analysis = self.analyze_cluster_state(timestep)
        
        # Dead agents don't transition
        if self.state == 'D':
            return {
                'probability': 0.0,
                'reasoning': f"{analysis['total_in_state']} deceased agents in cluster (network mourning period)",
                'confidence': 1.0
            }
        
        if analysis['total_in_state'] == 0:
            return {
                'probability': 0.0,
                'reasoning': f"No agents in {self.state} state.",
                'confidence': 1.0
            }
        
        # Determine which transition to predict
        if transition_type is None:
            # Default transitions
            default_transitions = {
                'S': 'S->E',
                'E': 'E->I',
                'I': 'I->R',  # Default for I is recovery
                'R': 'R->S'
            }
            transition_type = default_transitions.get(self.state, 'unknown')
        
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(
            f"cluster transition {self.state} {transition_type} probability patterns",
            n=5
        )
        memory_context = "\n".join([f"- {m.description[:150]}" for m in relevant_memories]) if relevant_memories else "No relevant history."

        # Query transmission rules via RAG
        if transition_type == 'S->E':
            query = f"Singapore COVID-19 transmission probability dormitory cluster quarantine measures with {analysis['observations']}"
        elif transition_type == 'E->I':
            query = f"incubation period and symptom onset for exposed population Singapore"
        elif transition_type == 'I->R':
            query = f"Singapore COVID-19 recovery probability infected population healthcare"
        elif transition_type == 'I->D':
            query = f"Singapore infection fatality rate IFR age-stratified death risk COVID-19 healthcare system"
        else:  # R->S
            query = f"daily reinfection probability for recovered population Singapore"
        
        knowledge = self.tools.execute_tool(
            "query_transmission_rules",
            agent=None,
            query=query
        )
        
        # Build prompt - different for each transition type
        if transition_type == 'I->D':
            prompt = self._build_death_prediction_prompt(analysis, neighbor_cluster_context, memory_context, knowledge, timestep)
        else:
            prompt = self._build_standard_prediction_prompt(analysis, neighbor_cluster_context, memory_context, knowledge, timestep, transition_type)
        
        try:
            with time_limit(30):
                _llm_rate_limiter.wait_if_needed()
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert epidemiologist analyzing Singapore COVID-19 data predicting {self._get_state_name()} population transitions. Use realistic transmission rates based on Singapore's epidemiological data."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.4,
                    max_tokens=600
                )
            
            content = response.choices[0].message.content
            
            reasoning = ""
            probability = 0.05
            confidence = "MODERATE"
            
            if "REASONING:" in content:
                reasoning = content.split("REASONING:")[1].split("PROBABILITY:")[0].strip()
            
            # ═══════════════════════════════════════════════════════════════
            # FIXED: Robust percentage parsing
            # ═══════════════════════════════════════════════════════════════
            if "PROBABILITY:" in content:
                prob_text = content.split("PROBABILITY:")[1].split("CONFIDENCE:")[0].strip().split()[0]
                try:
                    # Remove percentage symbol if present
                    if '%' in prob_text:
                        prob_text = prob_text.replace('%', '')
                        probability = float(prob_text) / 100.0
                    else:
                        probability = float(prob_text)
                    
                    # If looks like percentage (>1.0 but not given as %), convert
                    if probability > 1.0 and '%' not in content:
                        probability /= 100.0
                    
                    # Clamp to reasonable range
                    probability = min(max(probability, 0.0001), 0.50)
                except:
                    fallback_probs = {'S->E': 0.03, 'E->I': 0.25, 'I->R': 0.12, 'I->D': 0.005, 'R->S': 0.005}
                    probability = fallback_probs.get(transition_type, 0.05)
            # ═══════════════════════════════════════════════════════════════
            
            if "CONFIDENCE:" in content:
                confidence = content.split("CONFIDENCE:")[1].strip().split()[0]
            
            # ═══════════════════════════════════════════════════════════════
            # FIXED: Time-shaped hazards + Singapore-specific adjustments
            # ═══════════════════════════════════════════════════════════════
            
            if transition_type == 'S->E':
                # ═══════════════════════════════════════════════════════════════
                # SINGAPORE-SPECIFIC S→E CALIBRATION
                # ═══════════════════════════════════════════════════════════════
                
                infected_count = analysis['state_distribution']['E'] + analysis['state_distribution']['I']
                cluster_size = analysis['cluster_size']
                
                if infected_count > 0:
                    # Scale by infection prevalence
                    prevalence = infected_count / cluster_size
                    
                    # Network topology (infected degree)
                    mean_infected_contacts = analysis.get('mean_infected_contacts', 0.0)
                    contact_scale = min(1.0, mean_infected_contacts / 3.0)
                    
                    # ⭐ SINGAPORE-SPECIFIC: Detect dormitory clusters
                    is_dormitory = self._is_dormitory_cluster()
                    dormitory_mult = 1.0
                    
                    if is_dormitory and prevalence > 0.05:
                        # Dormitory outbreak: 20-40% attack rates (SAR)
                        # High-density, shared facilities → 2.5x community transmission
                        dormitory_mult = 2.5
                        reasoning += f" [DORMITORY CLUSTER: {dormitory_mult:.1f}x transmission]"
                    
                    # ⭐ SINGAPORE-SPECIFIC: Quarantine enforcement
                    quarantine_mult = self._get_quarantine_reduction()
                    if quarantine_mult < 1.0:
                        reasoning += f" [Quarantine enforcement: {(1-quarantine_mult)*100:.0f}% reduction]"
                    
                    # ⭐ EPIDEMIC PHASE MULTIPLIER
                    phase_mult = 1.0
                    if self.phase_tracker:
                        phase_mults = self.phase_tracker.get_phase_multipliers()
                        phase_mult = phase_mults.get('S->E', 1.0)
                        
                        reasoning += f" [Phase: {self.phase_tracker.current_phase}, mult={phase_mult:.1f}x]"
                    
                    # Baseline floor (BOOSTED)
                    if prevalence < 0.10:
                        min_prob = 0.020  # 2%
                    elif prevalence < 0.30:
                        min_prob = 0.035  # 3.5%
                    else:
                        min_prob = 0.050  # 5%
                    
                    # Apply ALL multipliers: network + dormitory + quarantine + phase
                    adjusted_min = min_prob * max(0.3, contact_scale) * dormitory_mult * quarantine_mult * phase_mult
                    
                    # ⭐ CRITICAL: Also apply all multipliers to LLM prediction
                    probability = probability * dormitory_mult * quarantine_mult * phase_mult
                    
                    if probability < adjusted_min:
                        probability = adjusted_min
                        reasoning += f" [Calibrated: {adjusted_min:.1%} (prev={prevalence:.0%}, contacts={mean_infected_contacts:.1f})]"
            
            elif transition_type == 'E->I':
                # ⭐ REALISTIC: 3-5 day incubation (mean 4 days)
                avg_days = analysis.get('avg_days_exposed', 0.0)
                
                # NO symptoms before day 2
                if avg_days < 2:
                    hazard_floor = 0.0  # Too early
                elif avg_days < 3:
                    hazard_floor = 0.05  # Day 2: 5%
                elif avg_days < 4:
                    hazard_floor = 0.15  # Day 3: 15%
                elif avg_days < 5:
                    hazard_floor = 0.25  # Day 4: 25% (peak)
                elif avg_days < 6:
                    hazard_floor = 0.30  # Day 5: 30%
                else:
                    hazard_floor = 0.35  # Day 6+: 35%
                
                hazard_floor = min(hazard_floor, 0.40)
                
                if probability < hazard_floor:
                    probability = hazard_floor
                    reasoning += f" [Incubation hazard: {hazard_floor:.1%} for avg day {avg_days:.1f}]"
            
            elif transition_type == 'I->R':
                # ⭐ REALISTIC: Recovery takes 7-14 days minimum
                avg_days = analysis.get('avg_days_infected', 0.0)
                
                # NO recovery before day 7
                if avg_days < 7:
                    hazard_floor = 0.0
                elif avg_days < 10:
                    # Days 7-9: 5% → 10%
                    hazard_floor = 0.05 + 0.017 * (avg_days - 7)
                elif avg_days < 14:
                    # Days 10-13: 10% → 20%
                    hazard_floor = 0.10 + 0.025 * (avg_days - 10)
                else:
                    # Day 14+: 20% → 30%
                    hazard_floor = 0.20 + 0.02 * min(avg_days - 14, 5)
                
                hazard_floor = min(hazard_floor, 0.35)
                
                if probability < hazard_floor:
                    probability = hazard_floor
                    reasoning += f" [Recovery hazard: {hazard_floor:.1%} for avg day {avg_days:.1f}]"
            
            elif transition_type == 'I->D':
                # ⭐ SINGAPORE-SPECIFIC: Excellent healthcare system
                # Singapore CFR: ~0.05% (vs global ~1-2%)
                # Due to: World-class ICU capacity, early intervention, contact tracing
                
                avg_days = analysis.get('avg_days_infected', 0.0)
                
                print(f"      I→D: avg_days={avg_days:.1f}, cluster_prob={probability:.3f}")
                
                # NO death before day 7
                if avg_days < 7:
                    probability = 0.0  # ⭐ FORCE to zero
                    reasoning += f" [Too early for death: day {avg_days:.1f}]"
                elif 7 <= avg_days <= 14:
                    # Peak death risk days 7-14
                    # ⭐ SINGAPORE REDUCTION: 0.3x global IFR (excellent healthcare)
                    singapore_healthcare_factor = 0.3
                    
                    # Get age-weighted death factor for cluster
                    age_weighted_factor = self._get_cluster_age_death_factor()
                    
                    # Base hazard with Singapore adjustment
                    hazard_floor = 0.002 * singapore_healthcare_factor * age_weighted_factor * (avg_days - 7)
                    
                    if probability < hazard_floor:
                        probability = hazard_floor
                        reasoning += f" [Singapore death hazard: {hazard_floor:.1%} (healthcare={singapore_healthcare_factor}, age_factor={age_weighted_factor:.1f})]"
                
                # NOTE: Competing risk normalization (I→R vs I→D) happens at entity level
            
            # ═══════════════════════════════════════════════════════════════
            
            # Store in memory
            self.memory.add(
                f"T{timestep} {transition_type}: p={probability:.3f} | {self._compress_text(reasoning)}",
                importance=5.0,
                memory_type="decision"
            )
            
            confidence_map = {'LOW': 0.6, 'MODERATE': 0.8, 'HIGH': 0.95}
            confidence_score = confidence_map.get(confidence, 0.8)
            
            return {
                'probability': probability,
                'reasoning': reasoning,
                'confidence': confidence_score,
                'transition_type': transition_type
            }
        
        except Exception as e:
            print(f"StateAgent {self.state} {transition_type} prediction failed: {e}")
            fallback_probs = {'S->E': 0.03, 'E->I': 0.25, 'I->R': 0.12, 'I->D': 0.005, 'R->S': 0.005}
            return {
                'probability': fallback_probs.get(transition_type, 0.05),
                'reasoning': "Fallback estimate due to error.",
                'confidence': 0.5,
                'transition_type': transition_type
            }
        finally:
            signal.alarm(0)

    def _build_death_prediction_prompt(
        self,
        analysis: Dict,
        neighbor_context: str,
        memory_context: str,
        knowledge: str,
        timestep: int
    ) -> str:
        """Build prompt specifically for I→D (death) prediction with Singapore context."""
        
        # Extract death-relevant observations
        avg_days = "unknown"
        elderly_count = 0
        peak_risk_count = 0
        high_comorbidity_count = 0
        
        for obs in analysis['observations']:
            if 'days infected' in obs.lower():
                avg_days = obs
            if 'elderly infected' in obs.lower():
                try:
                    elderly_count = int(obs.split()[0])
                except:
                    pass
            if 'peak death risk' in obs.lower():
                try:
                    peak_risk_count = int(obs.split()[0])
                except:
                    pass
            if 'comorbidities' in obs.lower():
                try:
                    high_comorbidity_count = int(obs.split()[0])
                except:
                    pass
        
        return f"""You are predicting the DAILY DEATH PROBABILITY (I→D transition) for infected agents in your Singapore COVID-19 cluster.

CURRENT CLUSTER STATE:
- Total agents: {analysis['cluster_size']}
- Infected agents: {analysis['total_in_state']}
- State distribution: S={analysis['state_distribution']['S']}, E={analysis['state_distribution']['E']}, I={analysis['state_distribution']['I']}, R={analysis['state_distribution']['R']}, D={analysis['state_distribution']['D']}
- Key observations: {' '.join(analysis['observations'])}

DEATH RISK FACTORS IN YOUR CLUSTER:
- Elderly infected (70+): {elderly_count} agents (Singapore IFR ~2%)
- In peak death risk period (days 7-14): {peak_risk_count} agents
- High comorbidity burden (2+): {high_comorbidity_count} agents

INFECTION TIMELINE:
{avg_days}

NEIGHBORING CLUSTERS:
{neighbor_context}

RELEVANT HISTORY:
{memory_context}

SINGAPORE COVID-19 CONTEXT:
{knowledge}

SINGAPORE AGE-STRATIFIED INFECTION FATALITY RATE (IFR) - CUMULATIVE:
Singapore's excellent healthcare system results in ~70% LOWER mortality than global average.

- Age <20: 0.001% (1 per 100,000) - effectively zero
- Age 20-49: 0.02% (20 per 100,000) - very rare
- Age 50-69: 0.2% (200 per 100,000) - occasional
- Age 70+: 2% (2,000 per 100,000) - moderate (vs 7% globally)

SINGAPORE HEALTHCARE FACTORS:
- World-class ICU capacity (maintained throughout pandemic)
- Early intervention and aggressive treatment
- Comprehensive contact tracing (reduces severe cases)
- High vaccination rates (2020 data: none yet, but good preparedness)

DAILY DEATH RISK BY TIMELINE:
- Days 0-6: Very low (0.05x baseline IFR converted to daily)
- Days 7-14: PEAK RISK (0.2x baseline IFR per day) - Singapore factor
- Days 15+: Declining (0.03x baseline IFR per day)

RISK MULTIPLIERS:
- Each comorbidity: +80% death risk
- Unvaccinated: Baseline risk
- Partially vaccinated: -70% death risk
- Fully vaccinated: -90% death risk

CALCULATION EXAMPLE (SINGAPORE):
- If cluster has 1 elderly agent (70+, Singapore IFR 2%) at day 10 (peak):
  * Daily risk ≈ 0.02 × 0.2 = 0.004 (0.4% daily)
- If cluster has young agents (<30, Singapore IFR 0.02%):
  * Daily risk ≈ 0.0002 × 0.2 = 0.00004 (0.004% daily)

REALISTIC DAILY DEATH PROBABILITY RANGES (SINGAPORE):
- All young, vaccinated: 0.0001-0.001% (extremely rare)
- Mixed ages, some elderly: 0.01-0.1% (very occasional deaths)
- Many elderly, unvaccinated, peak period: 0.2-1% (higher but still controlled)

IMPORTANT: Singapore's mortality is MUCH LOWER than global averages due to excellent healthcare.
Most infected agents recover! Only high-risk groups have significant death risk.

TASK: Estimate the cluster-level DAILY death probability (I→D) for Singapore.

Consider:
1. Age distribution of infected agents (CRITICAL)
2. Days infected (peak risk days 7-14)
3. Comorbidity burden
4. Singapore healthcare system quality (0.3x global IFR)

Think step-by-step:
1. What's the age profile? (If mostly young → extremely low risk)
2. What's the timeline? (If early days → zero risk)
3. Are there elderly in peak risk period? (If yes → moderate risk, but lower than global)
4. Calculate weighted average with Singapore healthcare factor

Provide your answer in this EXACT format:
REASONING: <3-4 sentences analyzing death risk factors quantitatively>
PROBABILITY: <cluster-level DAILY death probability as decimal, e.g., 0.002 for 0.2%>
CONFIDENCE: <confidence in this estimate: LOW/MODERATE/HIGH>

BE REALISTIC: Singapore had very few deaths in early 2020. Most clusters will have ZERO deaths unless many elderly/high-risk."""
    
    def _build_standard_prediction_prompt(
        self,
        analysis: Dict,
        neighbor_context: str,
        memory_context: str,
        knowledge: str,
        timestep: int,
        transition_type: str
    ) -> str:
        """Build prompt for standard transitions (S→E, E→I, I→R, R→S) with Singapore context."""

        # ⭐ Get epidemic phase context
        phase_context = ""
        if self.phase_tracker:
            phase_context = self.phase_tracker.get_context_string()
        else:
            phase_context = "Epidemic phase: Unknown"
        
        prompt = f"""You are State Agent {self.state} for Cluster {self.cluster_id}, predicting DAILY transition probabilities for Singapore COVID-19.

CALIBRATED BASELINE RATES (Singapore R₀ = {EpidemicParams.R0_TARGET}):
- S→E: {EpidemicParams.get_baseline_rate('S->E')*100:.1f}% per infected contact
- E→I: {EpidemicParams.get_baseline_rate('E->I')*100:.1f}% per day (mean: {EpidemicParams.LATENT_PERIOD_MEAN} days)
- I→R: {EpidemicParams.get_baseline_rate('I->R')*100:.1f}% per day (mean: {EpidemicParams.INFECTIOUS_PERIOD_MEAN} days)

Use these as BASELINE - adjust up/down based on cluster conditions.

CURRENT CLUSTER STATE:
- Total agents: {analysis['cluster_size']}
- Agents in {self.state} state: {analysis['total_in_state']}
- State distribution: S={analysis['state_distribution']['S']}, E={analysis['state_distribution']['E']}, I={analysis['state_distribution']['I']}, R={analysis['state_distribution']['R']}, D={analysis['state_distribution']['D']}
- Observations: {' '.join(analysis['observations'])}

NEIGHBORING CLUSTERS:
{neighbor_context}

RELEVANT HISTORY:
{memory_context}

SINGAPORE CONTEXT:
{knowledge}

REALISTIC DAILY PROBABILITY RANGES (use as calibration):
"""
        
        # Add state-specific calibration guidance
        if transition_type == 'S->E':
            infected_count = analysis['state_distribution']['E'] + analysis['state_distribution']['I']
            total_agents = analysis['cluster_size']
            infected_proportion = infected_count / total_agents if total_agents > 0 else 0
            mean_contacts = analysis.get('mean_infected_contacts', 0.0)
            
            # ⭐ Add phase-specific guidance
            phase_guidance = ""
            if self.phase_tracker:
                phase = self.phase_tracker.current_phase
                if phase == "PRE_EPIDEMIC":
                    phase_guidance = "\n🕐 PRE-EPIDEMIC: Low transmission, use LOWER end of ranges"
                elif phase == "EXPONENTIAL_GROWTH":
                    phase_guidance = "\n🚨 EXPONENTIAL GROWTH: High transmission! Use UPPER end of ranges or above!"
                elif phase == "PEAK":
                    phase_guidance = "\n📈 PEAK: Very high transmission, use UPPER end of ranges"
                elif phase == "DECLINE":
                    phase_guidance = "\n📉 DECLINE: Transmission declining, use MIDDLE ranges"
            
            prompt += f"""
S→E DAILY EXPOSURE PROBABILITY RANGES (SINGAPORE):
- No infected/exposed in cluster: {EpidemicParams.BETA_COMMUNITY*100:.1f}% (baseline community)
- Some infected in network: {EpidemicParams.BETA_CONTACT*100:.1f}% per infected contact
- Household infected (1 member): ~{(1-np.exp(-EpidemicParams.BETA_HOUSEHOLD))*100:.0f}% daily
- DORMITORY CLUSTER: 8-25% daily! ⚠️ (high density, shared facilities)
- Quarantine Order (QO) active: DIVIDE by 5 (80% reduction)
- EXPONENTIAL GROWTH PHASE: 8-20%!
{phase_guidance if self.phase_tracker else ''}

YOUR CLUSTER CONTEXT:
- Infected/Exposed proportion: {infected_proportion:.1%}
- Mean infected contacts: {mean_contacts:.1f}

Typical range: 2-20% (depends on cluster type and epidemic phase!)
"""
        
        elif transition_type == 'E->I':
            avg_days = analysis.get('avg_days_exposed', 0.0)
            
            prompt += f"""
E→I DAILY SYMPTOM ONSET PROBABILITY RANGES (SINGAPORE):
- Day 0-1 exposed: 5-10% (very early, unlikely)
- Day 2-3 exposed: 20-35% (typical incubation midpoint)
- Day 4-5 exposed: 35-50% (peak symptom onset)
- Day 6+ exposed: 50-70% (late, most will have progressed)

ADJUSTMENTS:
- High vaccination coverage (>60%): MULTIPLY by 0.7x (milder/no symptoms)
- Young population (<18): MULTIPLY by 0.9x
- Elderly (>60): MULTIPLY by 1.2x
- Comorbidities: MULTIPLY by 1.3x

YOUR CLUSTER SITUATION:
Average days exposed: {avg_days:.1f}

Typical range: 20-50% daily
"""
        
        elif transition_type == 'I->R':
            avg_days = analysis.get('avg_days_infected', 0.0)
            
            prompt += f"""
I→R DAILY RECOVERY PROBABILITY RANGES (SINGAPORE):
Singapore's excellent healthcare accelerates recovery by ~30% vs global average.

- Days 1-3 infected: 2-5% (too early for recovery)
- Days 4-7 infected: 8-15% (typical recovery period starts)
- Days 8-10 infected: 15-25% (peak recovery window)
- Days 11+ infected: 25-40% (overdue, most will recover)

SINGAPORE ADJUSTMENTS:
- Excellent healthcare: Base rates × 1.3
- High vaccination coverage (>60%): MULTIPLY by 1.5x (faster recovery)
- Young population (<18): MULTIPLY by 1.3x
- Elderly population (>60): MULTIPLY by 0.7x
- High comorbidities: MULTIPLY by 0.6-0.8x

NOTE: Death risk (I→D) is handled separately. This is RECOVERY only.

YOUR CLUSTER SITUATION:
Average days infected: {avg_days:.1f}

Typical range for your situation: 10-25% daily (Singapore)
"""
        
        else:  # R->S
            prompt += f"""
R→S DAILY REINFECTION PROBABILITY RANGES (SINGAPORE):
Singapore's comprehensive surveillance and contact tracing reduces reinfection risk.

- Recently recovered (<90 days): 0.03-0.1% (strong immunity)
- Medium term (90-180 days): 0.1-0.5% (waning immunity)
- Long term (>180 days): 0.5-2% (weak immunity)
- Unvaccinated recovered: Upper end of ranges
- Vaccinated recovered: Lower end of ranges

Typical range: 0.05-0.5% daily (Singapore)
"""
        
        prompt += f"""

TASK: Predict the cluster-level DAILY probability of {transition_type} transition for Singapore.

Think step-by-step:
1. What is the baseline rate given the cluster's situation?
2. What specific Singapore factors apply (dormitory, quarantine, healthcare)?
3. Look at the typical ranges above - where does this cluster fall?
4. Calculate your final estimate

Be REALISTIC - epidemics do spread! Use the Singapore-calibrated ranges provided.

Provide your answer in this EXACT format:
REASONING: <3-4 sentences analyzing the situation with quantitative thinking>
PROBABILITY: <cluster-level DAILY transition probability as decimal, e.g., 0.05 for 5%>
CONFIDENCE: <confidence in this estimate: LOW/MODERATE/HIGH>"""
        
        return prompt

    def _get_transition_name(self) -> str:
        """Get transition name."""
        transitions = {
            'S': 'S→E (exposure)',
            'E': 'E→I (symptom onset)',
            'I': 'I→R (recovery) / I→D (death)',
            'R': 'R→S (reinfection)',
            'D': 'None (deceased)'
        }
        return transitions.get(self.state, '?')

    def _get_state_name(self) -> str:
        """Get readable state name."""
        names = {
            'S': 'susceptible',
            'E': 'exposed',
            'I': 'infected',
            'R': 'recovered',
            'D': 'deceased'
        }
        return names.get(self.state, self.state)

    def __repr__(self):
        return f"StateAgent({self.state}, Cluster {self.cluster_id})"
