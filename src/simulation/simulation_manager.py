"""
Complete simulation manager orchestrating all cluster teams.

UPDATED FOR SEIRD MODEL:
- 5 states: S, E, I, R, D
- Death handling
- Dead agent network effects
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle

from src.agents.entity_agent import EntityAgent
from src.agents.epidemic_tools import EpidemicToolRegistry
from src.agents.cluster_team import ClusterTeam


class SimulationManager:
    """
    Manages complete simulation with all cluster teams.
    
    Orchestrates:
    - All cluster teams
    - Neighbor context sharing
    - Global state tracking (SEIRD)
    - Dead agent handling
    """
    
    def __init__(
        self,
        profiles: List[Dict],
        edges: List[Dict],
        cluster_assignments: Dict[int, List[int]],
        graphsage_embeddings: np.ndarray
    ):
        """
        Initialize simulation manager.
        
        Args:
            profiles: Agent profiles
            edges: Edge list
            cluster_assignments: Dict mapping cluster_id to agent_ids
            graphsage_embeddings: GraphSAGE embeddings
        """
        self.profiles = profiles
        self.edges = edges
        self.cluster_assignments = cluster_assignments
        self.graphsage_embeddings = graphsage_embeddings

        self._initialize_states()
        
        # Initialize tools (shared across all agents)
        self.tool_registry = EpidemicToolRegistry()
        
        # Create entity agents
        self.entity_agents: List[EntityAgent] = []
        self._create_entity_agents()
        
        # Create graph connections
        self._create_connections()
        
        # Create cluster teams
        self.cluster_teams: Dict[int, ClusterTeam] = {}
        self._create_cluster_teams()
        
        # Simulation state
        self.current_timestep = 0
        self.state_history = []


    def _initialize_states(self):
        """Initialize agent states from Kaggle distribution if not present."""
        import pandas as pd
        import random
        
        # Check if states already initialized
        if all('initial_state' in p for p in self.profiles):
            return
        
        print("Initializing agent states from Kaggle data...")
        
        # Load Kaggle data
        kaggle_path = Path("data/singapore/train.csv")
        if not kaggle_path.exists():
            print(f"  ⚠️ Kaggle data not found, using default distribution")
            
            # Define initial state counts (per 1000 agents)
            # Based on early outbreak scenario: mostly susceptible, few exposed/infected
            state_counts = {
                'S': 991,  # 99% susceptible
                'E': 8,    # 0.8% exposed
                'I': 1,    # 0.2% infected
                'R': 0,    # 0% recovered (outbreak start)
                'D': 0     # 0% dead (outbreak start)
            }
            
            # Convert to probabilities
            total_count = sum(state_counts.values())
            state_probs = {k: v / total_count for k, v in state_counts.items()}
            
            print(f"  Using counts: S={state_counts['S']}, E={state_counts['E']}, I={state_counts['I']}")
        else:
            df = pd.read_csv(kaggle_path)
            day_0_counts = df['day_0'].value_counts()
            total = len(df)
            
            # Get actual counts from Kaggle
            kaggle_counts = {
                'S': day_0_counts.get('S', 0),
                'E': day_0_counts.get('E', 0),
                'I': day_0_counts.get('I', 0),
                'R': day_0_counts.get('R', 0),
                'D': day_0_counts.get('D', 0)
            }
            
            print(f"  Kaggle counts: S={kaggle_counts['S']}, E={kaggle_counts['E']}, "
                f"I={kaggle_counts['I']}, R={kaggle_counts['R']}, D={kaggle_counts['D']}")
            
            # Convert to probabilities
            state_probs = {k: v / total for k, v in kaggle_counts.items()}
        
        # Normalize probabilities to ensure they sum to 1.0
        total_prob = sum(state_probs.values())
        if total_prob > 0:
            state_probs = {k: v/total_prob for k, v in state_probs.items()}
        else:
            state_probs = {'S': 1.0, 'E': 0.0, 'I': 0.0, 'R': 0.0, 'D': 0.0}
        
        # Assign states
        for profile in self.profiles:
            rand = random.random()
            cumsum = 0
            assigned = False
            
            for state in ['S', 'E', 'I', 'R', 'D']:
                cumsum += state_probs.get(state, 0)
                if rand < cumsum:
                    profile['initial_state'] = state
                    profile['state'] = state
                    assigned = True
                    break
            
            # Fallback if no state assigned
            if not assigned:
                profile['initial_state'] = 'S'
                profile['state'] = 'S'
        
        # Verify all profiles have states
        result = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for p in self.profiles:
            state = p.get('initial_state', 'S')
            if state not in result:
                state = 'S'
                p['initial_state'] = 'S'
                p['state'] = 'S'
            result[state] += 1
        
        print(f"  ✓ Initialized {len(self.profiles)} agents: "f"S={result['S']}, E={result['E']}, I={result['I']}, R={result['R']}, D={result['D']}")

    def _create_entity_agents(self):
        """Create all entity agents."""
        print(f"\nCreating {len(self.profiles)} entity agents...")
        
        for profile in self.profiles:
            # Add features if not present
            if 'features' not in profile:
                profile['features'] = self._create_features(profile)
            
            agent = EntityAgent(
                agent_id=profile['agent_id'],
                profile=profile,
                tool_registry=self.tool_registry
            )
            
            self.entity_agents.append(agent)
        
        print(f"✓ Created {len(self.entity_agents)} entity agents")
    
    def _create_features(self, profile: Dict) -> np.ndarray:
        """Create feature vector for agent."""
        features = []
        
        # Age (normalized)
        features.append(profile['age'] / 100.0)
        
        # Occupation (one-hot, 6 types)
        occupations = ['healthcare_worker', 'office_worker', 'retail_worker', 'teacher', 'student', 'retired']
        occ_onehot = [1 if profile['occupation'] == o else 0 for o in occupations]
        features.extend(occ_onehot)
        
        # Household size (normalized)
        features.append(1.0 / 10.0)
        
        # Vaccination (one-hot)
        vacc_onehot = [0, 0, 0]
        vacc_onehot[profile['vaccination_status']] = 1
        features.extend(vacc_onehot)
        
        # Comorbidities
        features.append(profile['comorbidity_count'] / 5.0)
        
        # Behavioral
        features.append(profile['mobility_score'])
        features.append(profile['compliance_score'])
        features.append(profile['risk_awareness'])
        
        # State (one-hot SEIRD - 5 states)
        states = ['S', 'E', 'I', 'R', 'D']
        state_onehot = [1 if profile['initial_state'] == s else 0 for s in states]
        features.extend(state_onehot)
        
        return np.array(features, dtype=np.float32)
    
    def _create_connections(self):
        """Create neighbor connections."""
        print("Creating agent connections...")
        
        agent_map = {a.agent_id: a for a in self.entity_agents}
        
        for edge in self.edges:
            a1 = agent_map.get(edge['agent_1'])
            a2 = agent_map.get(edge['agent_2'])
            
            if a1 and a2:
                a1.add_neighbor(a2)
                a2.add_neighbor(a1)
        
        avg_degree = np.mean([len(a.neighbors) for a in self.entity_agents])
        print(f"✓ Created connections (avg degree: {avg_degree:.1f})")
    
    def _create_cluster_teams(self):
        """Create cluster teams with trained neural models."""
        print(f"\nCreating {len(self.cluster_assignments)} cluster teams...")
        
        # Check for trained neural model
        neural_model_path = Path("data/processed/neural_model/best_model.pt")
        
        if neural_model_path.exists():
            print(f"✓ Found trained neural model: {neural_model_path}")
        else:
            print(f"⚠ No trained neural model found at {neural_model_path}")
            print(f"  Run: python scripts/train_neural_pathway.py")
            print(f"  Continuing with untrained model (will output poor predictions)")
        
        for cluster_id, agent_ids in self.cluster_assignments.items():
            # Get entity agents for this cluster
            cluster_agents = [a for a in self.entity_agents if a.agent_id in agent_ids]
            
            if not cluster_agents:
                continue
            
            # Create team with neural model path
            team = ClusterTeam(
                cluster_id=cluster_id,
                entity_agents=cluster_agents,
                tool_registry=self.tool_registry,
                graphsage_embeddings=self.graphsage_embeddings,
                neural_model_path=neural_model_path if neural_model_path.exists() else None
            )
            
            self.cluster_teams[cluster_id] = team
        
        print(f"✓ Created {len(self.cluster_teams)} cluster teams")
    
    def simulate_timestep(self, timestep: int):
        """
        Simulate one timestep.
        
        Args:
            timestep: Current timestep
        """
        print(f"\n{'='*80}")
        print(f"TIMESTEP {timestep}")
        print(f"{'='*80}")
        
        # Perception: All agents perceive environment
        for agent in self.entity_agents:
            if agent.is_active():  # Only active (non-dead) agents perceive
                agent.perceive_environment(timestep)
        
        # Cluster-level reasoning
        cluster_results = {}
        
        for cluster_id, team in self.cluster_teams.items():
            # Get neighbor context
            neighbor_context = self._get_neighbor_context(cluster_id)
            
            # Full reasoning cycle
            result = team.full_reasoning_cycle(timestep, neighbor_context)
            cluster_results[cluster_id] = result
        
        # Update days in state for all agents
        for agent in self.entity_agents:
            if agent.is_dead:
                agent.days_since_death += 1
            else:
                agent.days_in_state += 1
        
        # Remove dead agents who have exceeded mourning period
        self._remove_expired_dead_agents(timestep)
        
        # Record state
        self._record_state(timestep)
        
        # Print summary
        self._print_summary()
    
    def _get_neighbor_context(self, cluster_id: int) -> str:
        """Get context from neighboring clusters."""
        cluster_agent_ids = set(self.cluster_assignments[cluster_id])
        
        neighbor_clusters = {}
        
        # Find connections to other clusters
        for agent_id in cluster_agent_ids:
            agent = next((a for a in self.entity_agents if a.agent_id == agent_id), None)
            if not agent:
                continue
            
            for neighbor in agent.neighbors:
                if neighbor.agent_id not in cluster_agent_ids:
                    # Find neighbor's cluster
                    for other_id, other_ids in self.cluster_assignments.items():
                        if neighbor.agent_id in other_ids:
                            if other_id not in neighbor_clusters:
                                neighbor_clusters[other_id] = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
                            neighbor_clusters[other_id][neighbor.state] += 1
        
        if not neighbor_clusters:
            return "No significant connections to other clusters."
        
        context = "Connected clusters:\n"
        for other_id, states in neighbor_clusters.items():
            total = sum(states.values())
            context += f"  Cluster {other_id}: {states['E']}/{total} exposed, {states['I']}/{total} infected, {states['R']}/{total} recovered"
            if states['D'] > 0:
                context += f", {states['D']}/{total} deceased"
            context += "\n"
        
        return context
    
    def _remove_expired_dead_agents(self, timestep: int):
        """
        Remove dead agents who have exceeded mourning period.
        
        Dead agents remain in network for 7-10 days, then are removed.
        """
        for agent in self.entity_agents:
            if agent.should_be_removed_from_network(timestep):
                # Remove from all neighbor lists
                for neighbor in agent.neighbors:
                    if agent in neighbor.neighbors:
                        neighbor.neighbors.remove(agent)
                
                print(f"  Removed deceased agent {agent.agent_id} ({agent.name}) from network (day {agent.days_since_death} since death)")
    
    def _record_state(self, timestep: int):
        """Record current state (SEIRD)."""
        state_dist = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.entity_agents:
            state_dist[agent.state] += 1
        
        self.state_history.append({
            'timestep': timestep,
            'distribution': state_dist
        })
    
    def _print_summary(self):
        """Print state summary (SEIRD)."""
        state_dist = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.entity_agents:
            state_dist[agent.state] += 1
        
        total = len(self.entity_agents)
        print(f"\nCurrent state: "
              f"S={state_dist['S']} ({state_dist['S']/total:.0%}), "
              f"E={state_dist['E']} ({state_dist['E']/total:.0%}), "
              f"I={state_dist['I']} ({state_dist['I']/total:.0%}), "
              f"R={state_dist['R']} ({state_dist['R']/total:.0%}), "
              f"D={state_dist['D']} ({state_dist['D']/total:.0%})")
    
    def run_simulation(self, num_timesteps: int = 10):
        """
        Run complete simulation.
        
        Args:
            num_timesteps: Number of timesteps to simulate
        """
        print("\n" + "="*80)
        print("STARTING STATEAGENTNET SIMULATION (SEIRD MODEL)")
        print("="*80)
        
        for t in range(num_timesteps):
            self.simulate_timestep(t)
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
    
    def save_results(self, output_dir: Path):
        """Save simulation results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save state history (aggregated) - SEIRD
        with open(output_dir / "state_history.json", 'w') as f:
            json.dump(self.state_history, f, indent=2)
        print(f"  ✓ Saved state_history.json")
        
        # 2. Save individual agent trajectories - SEIRD
        agent_trajectories = []
        for agent in self.entity_agents:
            trajectory = {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'profile': {
                    'age': agent.profile['age'],
                    'occupation': agent.profile['occupation'],
                    'household_id': agent.profile['household_id'],
                    'vaccination_status': agent.profile['vaccination_status'],
                    'comorbidity_count': agent.profile['comorbidity_count']
                },
                'initial_state': agent.profile['initial_state'],
                'final_state': agent.state,
                'state_history': agent.state_history if hasattr(agent, 'state_history') else [agent.state],
                'days_in_final_state': agent.days_in_state,
                'is_dead': agent.is_dead,
                'days_since_death': agent.days_since_death if agent.is_dead else None
            }
            agent_trajectories.append(trajectory)
        
        with open(output_dir / "agent_trajectories.json", 'w') as f:
            json.dump(agent_trajectories, f, indent=2)
        print(f"  ✓ Saved agent_trajectories.json")
        
        # 3. Save cluster statistics - SEIRD
        cluster_stats = {}
        for cluster_id, team in self.cluster_teams.items():
            state_dist = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
            for agent in team.entity_agents:
                state_dist[agent.state] += 1
            
            cluster_stats[str(cluster_id)] = {
                'size': len(team.entity_agents),
                'final_distribution': state_dist,
                'agent_ids': [a.agent_id for a in team.entity_agents]
            }
        
        with open(output_dir / "cluster_statistics.json", 'w') as f:
            json.dump(cluster_stats, f, indent=2)
        print(f"  ✓ Saved cluster_statistics.json")
        
        print(f"\n✓ All results saved to {output_dir}")
