"""
Meta-Agent: LLM-based coordinator for cluster-level reasoning.
Has memory, uses tools, coordinates State Agents.

UPDATED FOR PARALLEL SEIRD PREDICTION + SINGAPORE COVID-19:
- Manages 5 State Agents (S, E, I, R, D)
- ALL State Agents predict EVERY timestep (parallel hazards)
- Returns probabilities for ALL transitions: S→E, E→I, I→R, I→D, R→S
- Meta-Agent is now OBSERVER/COORDINATOR, not traffic cop
- State Agent I predicts BOTH I→R (recovery) AND I→D (death)

SINGAPORE-SPECIFIC MONITORING:
- ✓ Detects dormitory clusters (high-density outbreaks)
- ✓ Tracks quarantine compliance
- ✓ Monitors imported cases
- ✓ Observes cluster type patterns
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os
import signal

from src.agents.entity_agent import _llm_rate_limiter
from src.agents.memory_stream import MemoryStream
from src.agents.epidemic_tools import EpidemicToolRegistry
from src.agents.state_agent import StateAgent


class MetaAgent:
    """
    LLM-based Meta-Agent that coordinates cluster-level reasoning.
    
    Each cluster has 1 Meta-Agent that:
    - Monitors cluster state (SEIRD)
    - Coordinates ALL State Agents in parallel (no gating)
    - Returns probabilities for ALL transitions simultaneously
    - Has memory and uses tools
    
    NEW: Meta-Agent is an OBSERVER/COORDINATOR, not a traffic cop.
    All SEIRD transitions occur in parallel each timestep.
    
    SINGAPORE ENHANCEMENTS:
    - Dormitory cluster detection and monitoring
    - Quarantine Order (QO) compliance tracking
    - Imported case surveillance
    - Cluster type pattern recognition
    """
    
    def __init__(
        self,
        cluster_id: int,
        cluster_agents: List,
        tool_registry: EpidemicToolRegistry,
        llm_client: Optional[OpenAI] = None
    ):
        """
        Initialize Meta-Agent.
        
        Args:
            cluster_id: Cluster identifier
            cluster_agents: List of EntityAgent objects
            tool_registry: Tool registry
            llm_client: OpenAI client
        """
        self.cluster_id = cluster_id
        self.cluster_agents = cluster_agents
        self.tools = tool_registry
        self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Memory system
        self.memory = MemoryStream(
            agent_id=f"meta_{cluster_id}",
            agent_name=f"MetaAgent_Cluster{cluster_id}",
            llm_client=self.llm_client
        )
        
        # State Agents (will be set externally) - SEIRD: 5 agents
        self.state_agents: Dict[str, StateAgent] = {}
        
        # Initialize
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize with background."""
        background = f"""I am the Meta-Agent coordinating Cluster {self.cluster_id} for Singapore COVID-19 simulation.
I manage {len(self.cluster_agents)} entity agents and coordinate 5 State Agents (S, E, I, R, D).
My role is to monitor cluster dynamics and coordinate parallel predictions from ALL State Agents.
SEIRD model: Susceptible → Exposed → Infected → Recovered or Dead, with reinfection possible.
All transitions occur simultaneously each timestep (parallel hazards model).
I monitor Singapore-specific patterns: dormitory clusters, quarantine compliance, imported cases."""
        
        self.memory.add(background, importance=8.0, memory_type="reflection")
    
    def set_state_agents(self, state_agents: Dict[str, StateAgent]):
        """Set State Agent references."""
        self.state_agents = state_agents
    
    def _detect_cluster_type(self) -> str:
        """
        Detect Singapore-specific cluster type.
        
        Returns:
            Cluster type: 'dormitory', 'household', 'workplace', 'community', 'imported'
        """
        if not self.cluster_agents:
            return 'community'
        
        # Count cluster characteristics
        dorm_count = sum(1 for a in self.cluster_agents 
                        if 'dorm' in str(a.profile.get('cluster', '')).lower())
        
        manual_workers = sum(1 for a in self.cluster_agents 
                            if a.profile.get('occupation') == 'manual_worker')
        
        imported = sum(1 for a in self.cluster_agents 
                      if a.profile.get('is_imported', False))
        
        total = len(self.cluster_agents)
        
        # Classify
        if dorm_count > total * 0.3 or manual_workers > total * 0.5:
            return 'dormitory'
        elif imported > total * 0.5:
            return 'imported'
        
        # Check household concentration
        households = {}
        for agent in self.cluster_agents:
            hh = agent.profile.get('household_id', -1)
            if hh >= 0:
                households[hh] = households.get(hh, 0) + 1
        
        if households and max(households.values()) > total * 0.7:
            return 'household'
        
        # Check occupation concentration
        occupations = {}
        for agent in self.cluster_agents:
            occ = agent.profile.get('occupation', 'unknown')
            occupations[occ] = occupations.get(occ, 0) + 1
        
        if occupations and max(occupations.values()) > total * 0.6:
            return 'workplace'
        
        return 'community'
    
    def monitor_cluster(self, timestep: int) -> Dict:
        """
        Monitor cluster state and dynamics (SEIRD) with Singapore-specific patterns.
        
        Args:
            timestep: Current timestep
            
        Returns:
            Monitoring summary with Singapore-specific observations
        """
        # Get state distribution (SEIRD - 5 states)
        state_dist = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.cluster_agents:
            state_dist[agent.state] += 1
        
        # Calculate proportions
        total = len(self.cluster_agents)
        proportions = {k: v/total for k, v in state_dist.items()}
        
        # Detect cluster type
        cluster_type = self._detect_cluster_type()
        
        observations = []
        observations.append(
            f"Current distribution: S={state_dist['S']} ({proportions['S']:.0%}), "
            f"E={state_dist['E']} ({proportions['E']:.0%}), "
            f"I={state_dist['I']} ({proportions['I']:.0%}), "
            f"R={state_dist['R']} ({proportions['R']:.0%}), "
            f"D={state_dist['D']} ({proportions['D']:.0%})"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # SINGAPORE-SPECIFIC OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════
        
        # Cluster type detection
        if cluster_type == 'dormitory':
            observations.append(f"🏢 DORMITORY CLUSTER detected (high-density migrant worker housing)")
            if proportions['I'] > 0.15:
                observations.append("⚠️ DORMITORY OUTBREAK: High attack rate expected (20-40% SAR)")
        elif cluster_type == 'imported':
            observations.append(f"✈️ IMPORTED CASE cluster (travel-related infections)")
        elif cluster_type == 'household':
            observations.append(f"🏠 HOUSEHOLD cluster (family transmission)")
        elif cluster_type == 'workplace':
            observations.append(f"🏭 WORKPLACE cluster (occupational transmission)")
        else:
            observations.append(f"🌍 COMMUNITY cluster (mixed transmission)")
        
        # Quarantine surveillance (Singapore-specific)
        quarantined_count = sum(1 for a in self.cluster_agents 
                               if a.profile.get('quarantine_status', False))
        if quarantined_count > 0:
            quarantine_rate = quarantined_count / total
            observations.append(
                f"🔒 {quarantined_count} agents under Quarantine Order (QO) "
                f"({quarantine_rate:.0%} - transmission reduced 80%)"
            )
        
        # Imported case tracking
        imported_count = sum(1 for a in self.cluster_agents 
                            if a.profile.get('is_imported', False))
        if imported_count > 0:
            observations.append(f"✈️ {imported_count} imported cases in cluster")
        
        # ═══════════════════════════════════════════════════════════════
        # STANDARD EPIDEMIC OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════
        
        # Check for outbreaks
        if proportions['I'] > 0.3:
            observations.append("⚠️ OUTBREAK: >30% of cluster infected")
        elif proportions['I'] > 0.15:
            observations.append("⚠️ Elevated infection rate in cluster")
        
        # Check exposure wave
        if proportions['E'] > 0.2:
            observations.append("⚠️ Large exposed population - expect symptom wave soon")
        
        # Check recovery progress
        if state_dist['I'] > 0 and proportions['R'] > proportions['I']:
            observations.append("✓ Recovery outpacing new infections")
        
        # Check deaths (with Singapore context)
        if state_dist['D'] > 0:
            avg_age_deceased = sum(a.profile['age'] for a in self.cluster_agents if a.state == 'D') / state_dist['D']
            observations.append(
                f"⚠️ {state_dist['D']} death(s) in cluster (avg age: {avg_age_deceased:.0f}y)"
            )
        
        # Check high-risk infected (elderly)
        elderly_infected = sum(1 for a in self.cluster_agents 
                              if a.state == 'I' and a.profile['age'] >= 70)
        if elderly_infected > 0:
            observations.append(
                f"⚠️ {elderly_infected} elderly infected (Singapore IFR ~2% for 70+)"
            )
        
        # Vaccination coverage (if available - 2020 data may not have this)
        vacc_counts = {0: 0, 1: 0, 2: 0}
        for agent in self.cluster_agents:
            vacc_counts[agent.profile['vaccination_status']] += 1
        
        if vacc_counts[2] > 0:  # Only show if anyone vaccinated
            observations.append(
                f"💉 Vaccination: {vacc_counts[2]} fully, {vacc_counts[1]} partial, "
                f"{vacc_counts[0]} unvaccinated"
            )
        
        monitoring_summary = {
            'timestep': timestep,
            'state_distribution': state_dist,
            'proportions': proportions,
            'observations': observations,
            'elderly_infected': elderly_infected,
            'cluster_type': cluster_type,
            'quarantined_count': quarantined_count,
            'imported_count': imported_count
        }
        
        # Store in memory (compressed)
        obs_text = (
            f"T{timestep}: S={state_dist['S']},E={state_dist['E']},I={state_dist['I']},"
            f"R={state_dist['R']},D={state_dist['D']} [{cluster_type} cluster]"
        )
        self.memory.add(obs_text, importance=5.0, memory_type="observation")
        
        return monitoring_summary
    
    def coordinate_cluster_reasoning(
        self,
        timestep: int,
        neighbor_context: str
    ) -> Dict:
        """
        Coordinate full cluster reasoning process.
        
        FIXED: Now predicts ALL transitions in parallel (no gating).
        This is the key fix that enables realistic SEIRD dynamics.
        
        Args:
            timestep: Current timestep
            neighbor_context: Context from neighboring clusters
            
        Returns:
            Dict with ALL transition probabilities:
            {
                'predictions': {
                    'S->E': {'probability': ..., 'reasoning': ..., 'confidence': ...},
                    'E->I': {'probability': ..., 'reasoning': ..., 'confidence': ...},
                    'I->R': {'probability': ..., 'reasoning': ..., 'confidence': ...},
                    'I->D': {'probability': ..., 'reasoning': ..., 'confidence': ...},
                    'R->S': {'probability': ..., 'reasoning': ..., 'confidence': ...}
                },
                'monitoring': {...}
            }
        """
        # Monitor cluster
        monitoring = self.monitor_cluster(timestep)
        state_dist = monitoring['state_distribution']
        
        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: Predict ALL transitions in parallel (no gating)
        # This is the single most important change for realistic SEIRD
        # ═══════════════════════════════════════════════════════════════
        predictions = {}
        
        # S→E: Susceptible to Exposed
        # Predict if there are susceptible agents
        if 'S' in self.state_agents and state_dist['S'] > 0:
            try:
                s_prediction = self.state_agents['S'].predict_transition_probability(
                    timestep=timestep,
                    neighbor_cluster_context=neighbor_context,
                    transition_type='S->E'
                )
                predictions['S->E'] = s_prediction
            except Exception as e:
                print(f"  ⚠️ State Agent S prediction failed: {e}")
                predictions['S->E'] = {
                    'probability': 0.03,  # Fallback baseline
                    'reasoning': f'Fallback: {str(e)[:50]}',
                    'confidence': 0.5
                }
        else:
            predictions['S->E'] = {
                'probability': 0.0,
                'reasoning': 'No susceptible agents in cluster',
                'confidence': 1.0
            }
        
        # E→I: Exposed to Infected
        # Predict if there are exposed agents
        if 'E' in self.state_agents and state_dist['E'] > 0:
            try:
                e_prediction = self.state_agents['E'].predict_transition_probability(
                    timestep=timestep,
                    neighbor_cluster_context=neighbor_context,
                    transition_type='E->I'
                )
                predictions['E->I'] = e_prediction
            except Exception as e:
                print(f"  ⚠️ State Agent E prediction failed: {e}")
                predictions['E->I'] = {
                    'probability': 0.25,  # Fallback baseline
                    'reasoning': f'Fallback: {str(e)[:50]}',
                    'confidence': 0.5
                }
        else:
            predictions['E->I'] = {
                'probability': 0.0,
                'reasoning': 'No exposed agents in cluster',
                'confidence': 1.0
            }
        
        # I→R and I→D: Infected to Recovered or Dead
        # Predict if there are infected agents
        if 'I' in self.state_agents and state_dist['I'] > 0:
            try:
                # I→R: Recovery prediction
                i_recovery = self.state_agents['I'].predict_transition_probability(
                    timestep=timestep,
                    neighbor_cluster_context=neighbor_context,
                    transition_type='I->R'
                )
                predictions['I->R'] = i_recovery
                
                # I→D: Death prediction (same State Agent I)
                i_death = self.state_agents['I'].predict_transition_probability(
                    timestep=timestep,
                    neighbor_cluster_context=neighbor_context,
                    transition_type='I->D'
                )
                predictions['I->D'] = i_death
            except Exception as e:
                print(f"  ⚠️ State Agent I prediction failed: {e}")
                predictions['I->R'] = {
                    'probability': 0.12,  # Fallback baseline
                    'reasoning': f'Fallback: {str(e)[:50]}',
                    'confidence': 0.5
                }
                predictions['I->D'] = {
                    'probability': 0.005,  # Fallback baseline (Singapore lower mortality)
                    'reasoning': f'Fallback: {str(e)[:50]}',
                    'confidence': 0.5
                }
        else:
            predictions['I->R'] = {
                'probability': 0.0,
                'reasoning': 'No infected agents in cluster',
                'confidence': 1.0
            }
            predictions['I->D'] = {
                'probability': 0.0,
                'reasoning': 'No infected agents in cluster',
                'confidence': 1.0
            }
        
        # R→S: Recovered to Susceptible (reinfection)
        # ONLY predict if:
        # 1. There are recovered agents
        # 2. Sufficient time has passed (immunity waning after ~180 days)
        if 'R' in self.state_agents and state_dist['R'] > 0 and timestep >= 180:
            try:
                r_prediction = self.state_agents['R'].predict_transition_probability(
                    timestep=timestep,
                    neighbor_cluster_context=neighbor_context,
                    transition_type='R->S'
                )
                predictions['R->S'] = r_prediction
            except Exception as e:
                print(f"  ⚠️ State Agent R prediction failed: {e}")
                predictions['R->S'] = {
                    'probability': 0.005,  # Fallback baseline
                    'reasoning': f'Fallback: {str(e)[:50]}',
                    'confidence': 0.5
                }
        else:
            # Suppress reinfection early in simulation (immunity still strong)
            if timestep < 180:
                reason = 'Reinfection suppressed (immunity still strong, t<180 days)'
            else:
                reason = 'No recovered agents in cluster'
            
            predictions['R->S'] = {
                'probability': 0.0,
                'reasoning': reason,
                'confidence': 1.0
            }
        
        # D state: No transitions (terminal state)
        # Dead agents don't transition - included for completeness
        
        # ═══════════════════════════════════════════════════════════════
        # End of parallel prediction
        # ═══════════════════════════════════════════════════════════════
        
        # Store coordination summary in memory (compressed)
        summary = (
            f"T{timestep} [{monitoring['cluster_type']}]: "
            f"SE={predictions['S->E']['probability']:.3f}, "
            f"EI={predictions['E->I']['probability']:.3f}, "
            f"IR={predictions['I->R']['probability']:.3f}, "
            f"ID={predictions['I->D']['probability']:.3f}, "
            f"RS={predictions['R->S']['probability']:.3f}"
        )
        self.memory.add(summary, importance=6.0, memory_type="coordination")
        
        return {
            'predictions': predictions,
            'monitoring': monitoring,
            'timestep': timestep
        }
    
    def __repr__(self):
        return f"MetaAgent(Cluster {self.cluster_id}, {len(self.cluster_agents)} agents)"
