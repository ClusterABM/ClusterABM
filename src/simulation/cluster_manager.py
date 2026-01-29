"""
Manages all cluster agent teams.

UPDATED FOR SEIRD MODEL:
- 5 states: S, E, I, R, D
- Death handling
- Network mourning effects
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class ClusterManager:
    """Manages all cluster agent teams for simulation (SEIRD model)."""
    
    def __init__(self, cluster_assignments: Dict, agent_population):
        """
        Initialize cluster manager.
        
        Args:
            cluster_assignments: Dict mapping cluster_id to list of agent_ids
            agent_population: AgentPopulation object
        """
        self.cluster_assignments = cluster_assignments
        self.agent_population = agent_population
        self.cluster_teams = {}
        
        # Initialize LLM clients for all agents
        print("Initializing LLM clients for agents...")
        self.agent_population.initialize_all_llm_clients()
        
        # Create cluster teams
        self._create_cluster_teams()
    
    def _create_cluster_teams(self):
        """Create ClusterAgentTeam for each cluster."""
        from src.agents.meta_agent import ClusterAgentTeam
        
        print(f"Creating agent teams for {len(self.cluster_assignments)} clusters...")
        
        for cluster_id, agent_ids in self.cluster_assignments.items():
            # Get ReasoningAgent objects for this cluster
            cluster_agents = [
                self.agent_population.agents[aid - 1]  # agent_id is 1-indexed
                for aid in agent_ids
            ]
            
            # Create team
            team = ClusterAgentTeam(cluster_id, cluster_agents)
            self.cluster_teams[cluster_id] = team
            
            print(f"  ✓ Cluster {cluster_id}: {len(cluster_agents)} agents")
        
        print(f"✓ Created {len(self.cluster_teams)} cluster agent teams")
    
    def simulate_timestep(self, timestep: int):
        """
        Simulate one timestep across all clusters (SEIRD model).
        
        Args:
            timestep: Current timestep
        """
        print(f"\n{'='*80}")
        print(f"TIMESTEP {timestep}")
        print(f"{'='*80}")
        
        # Step 1: Cluster-level reasoning for all clusters
        cluster_probabilities = {}
        
        for cluster_id, team in self.cluster_teams.items():
            cluster_probs = team.compute_cluster_probability(
                all_clusters=self.cluster_assignments,
                timestep=timestep
            )
            cluster_probabilities[cluster_id] = cluster_probs
        
        # Step 2: Individual agent reasoning and state updates (SEIRD)
        print(f"\nIndividual agent reasoning...")
        
        for cluster_id, team in self.cluster_teams.items():
            cluster_probs = cluster_probabilities[cluster_id]
            
            for agent in team.cluster_agents:
                # Skip dead agents (they don't transition)
                if agent.state == 'D':
                    if hasattr(agent, 'days_since_death'):
                        agent.days_since_death += 1
                    continue
                
                # S → E transition
                if agent.state == 'S' and cluster_probs.get('S', 0) > 0:
                    cluster_context = f"Cluster-level S→E probability: {cluster_probs['S']:.3f}"
                    
                    personal_prob = agent.reason_about_personal_risk(
                        cluster_probability=cluster_probs['S'],
                        cluster_context=cluster_context
                    )
                    
                    agent.update_state(personal_prob)
                
                # E → I transition
                elif agent.state == 'E' and cluster_probs.get('E', 0) > 0:
                    cluster_context = f"Cluster-level E→I probability: {cluster_probs['E']:.3f}"
                    
                    personal_prob = agent.reason_about_personal_risk(
                        cluster_probability=cluster_probs['E'],
                        cluster_context=cluster_context
                    )
                    
                    # Sample transition
                    u = np.random.random()
                    if u < personal_prob:
                        agent.state = 'I'
                        agent.days_in_state = 0
                        print(f"  Agent {agent.agent_id} ({agent.profile['name']}): E → I")
                
                # I → R or I → D transition
                elif agent.state == 'I' and cluster_probs.get('I', 0) > 0:
                    # Check for death first (age-stratified IFR)
                    death_prob = self._calculate_death_probability(agent)
                    
                    u_death = np.random.random()
                    if u_death < death_prob:
                        # Agent dies
                        agent.state = 'D'
                        agent.days_in_state = 0
                        agent.is_dead = True
                        agent.days_since_death = 0
                        print(f"  Agent {agent.agent_id} ({agent.profile['name']}): I → D (death)")
                    else:
                        # Check for recovery
                        cluster_context = f"Cluster-level I→R probability: {cluster_probs['I']:.3f}"
                        
                        personal_prob = agent.reason_about_personal_risk(
                            cluster_probability=cluster_probs['I'],
                            cluster_context=cluster_context
                        )
                        
                        u = np.random.random()
                        if u < personal_prob:
                            agent.state = 'R'
                            agent.days_in_state = 0
                            print(f"  Agent {agent.agent_id} ({agent.profile['name']}): I → R")
                
                # R → S transition (reinfection)
                elif agent.state == 'R' and cluster_probs.get('R', 0) > 0:
                    cluster_context = f"Cluster-level R→S probability: {cluster_probs['R']:.3f}"
                    
                    personal_prob = agent.reason_about_personal_risk(
                        cluster_probability=cluster_probs['R'],
                        cluster_context=cluster_context
                    )
                    
                    u = np.random.random()
                    if u < personal_prob:
                        agent.state = 'S'
                        agent.days_in_state = 0
                        print(f"  Agent {agent.agent_id} ({agent.profile['name']}): R → S (reinfection)")
        
        # Step 3: Print state distribution (SEIRD)
        self._print_state_distribution()
    
    def _calculate_death_probability(self, agent) -> float:
        """
        Calculate age-stratified infection fatality rate (IFR).
        
        Args:
            agent: Agent object
            
        Returns:
            Daily death probability
        """
        age = agent.profile['age']
        
        # Base IFR by age
        if age < 20:
            base_ifr = 0.00002
        elif age < 50:
            base_ifr = 0.0005
        elif age < 70:
            base_ifr = 0.005
        else:
            base_ifr = 0.07
        
        # Convert to daily probability (peak around day 10-14)
        days = agent.days_in_state
        if days < 7:
            daily_ifr = base_ifr * 0.1
        elif days < 14:
            daily_ifr = base_ifr * 0.3
        else:
            daily_ifr = base_ifr * 0.05
        
        # Comorbidity multiplier
        comorbidity_multiplier = 1.0 + (agent.profile.get('comorbidity_count', 0) * 0.8)
        daily_ifr *= comorbidity_multiplier
        
        # Vaccination protection (90% reduction)
        vacc_status = agent.profile.get('vaccination_status', 0)
        if vacc_status == 2:
            daily_ifr *= 0.1
        elif vacc_status == 1:
            daily_ifr *= 0.3
        
        return min(daily_ifr, 0.20)
    
    def _print_state_distribution(self):
        """Print current state distribution (SEIRD)."""
        total_states = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        
        for agent in self.agent_population.agents:
            total_states[agent.state] += 1
        
        total = sum(total_states.values())
        print(f"\nCurrent state distribution:")
        print(f"  S={total_states['S']} ({total_states['S']/total:.0%}), "
              f"E={total_states['E']} ({total_states['E']/total:.0%}), "
              f"I={total_states['I']} ({total_states['I']/total:.0%}), "
              f"R={total_states['R']} ({total_states['R']/total:.0%}), "
              f"D={total_states['D']} ({total_states['D']/total:.0%})")
    
    def get_state_history(self) -> List[Dict]:
        """Get state history for all agents."""
        history = []
        
        for agent in self.agent_population.agents:
            trajectory = {
                'agent_id': agent.agent_id,
                'name': agent.profile['name'],
                'state_history': agent.state_history if hasattr(agent, 'state_history') else [agent.state],
                'final_state': agent.state,
                'is_dead': agent.state == 'D'
            }
            history.append(trajectory)
        
        return history
