"""
FULL 83-DAY SIMULATION WITH BETA Parameter Searched via Rolling Window.
Period: 2020-01-23 to 2020-04-14 (83 days)

"""

import sys
from pathlib import Path
import json
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Tuple
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.simulation_manager import SimulationManager
from src.agents.epidemic_phase import EpidemicPhaseTracker
from src.config.epidemic_params import EpidemicParams, R0Tracker


class ComprehensiveLogger:
    """Comprehensive logging system for StateAgentNet explainability."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        
        # Create directory structure
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "timesteps").mkdir(exist_ok=True)
        (self.logs_dir / "agents").mkdir(exist_ok=True)
        (self.logs_dir / "state_agents").mkdir(exist_ok=True)
        (self.logs_dir / "fusion").mkdir(exist_ok=True)
        
        print(f"✓ Comprehensive logging initialized: {self.logs_dir}")
    
    def log_agent_reasoning(self, timestep: int, agent_id: int,
                            agent_name: str, reasoning_data: Dict):
        """Log individual agent's complete reasoning process."""
        log_file = self.logs_dir / "agents" / f"agent_{agent_id:03d}.jsonl"
        
        log_entry = {
            'timestep': timestep,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat(),
            'type': 'agent_reasoning',
            'current_state': reasoning_data.get('state', '?'),
            'days_in_state': reasoning_data.get('days_in_state', 0),
            'age': reasoning_data.get('age', 0),
            'vaccination': reasoning_data.get('vaccination', 0),
            'cluster_id': reasoning_data.get('cluster_id', -1),
            'cluster_probability': reasoning_data.get('cluster_prob', 0.0),
            'transition_type': reasoning_data.get('transition', '?'),
            'baseline_multiplier': reasoning_data.get('baseline_multiplier', 1.0),
            'baseline_probability': reasoning_data.get('baseline_prob', 0.0),
            'adjustment_factor': reasoning_data.get('adjustment', 1.0),
            'personal_probability': reasoning_data.get('probability', 0.0),
            'reasoning_text': reasoning_data.get('reasoning', ''),
            'sampled_value': reasoning_data.get('sampled_value', 0.0),
            'transitioned': reasoning_data.get('transitioned', False),
            'old_state': reasoning_data.get('old_state', '?'),
            'new_state': reasoning_data.get('new_state', '?')
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_timestep_summary(self, timestep: int, summary_data: Dict):
        """Log timestep-level summary."""
        log_file = self.logs_dir / "timesteps" / f"timestep_{timestep:03d}.json"
        
        log_entry = {
            'timestep': timestep,
            'timestamp': datetime.now().isoformat(),
            'state_distribution': summary_data.get('state_dist', {}),
            'transitions': summary_data.get('transitions', []),
            'deaths': summary_data.get('deaths', []),
            'cluster_probabilities': summary_data.get('cluster_probs', {}),
            'execution_time_seconds': summary_data.get('execution_time', 0)
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def log_simulation_summary(self, summary_data: Dict):
        """Log complete simulation summary."""
        log_file = self.logs_dir / "simulation_summary.json"
        
        with open(log_file, 'w') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)


class FullSimulation:
    """
    Full 83-day epidemic simulation with checkpointing.
    
    Simplified approach:
    - No rolling windows
    - Fixed beta = 0.15
    - Checkpoint every 10 days
    - Can resume from any checkpoint
    """
    
    def __init__(self, profiles, edges, cluster_assignments, embeddings, ground_truth_data):
        """
        Initialize full simulation.
        
        Args:
            profiles: Agent profiles
            edges: Network edges
            cluster_assignments: Cluster assignments
            embeddings: GraphSAGE embeddings
            ground_truth_data: Singapore MOH data (daily cases)
        """
        self.base_profiles = profiles
        self.edges = edges
        self.cluster_assignments = cluster_assignments
        self.embeddings = embeddings
        self.ground_truth = ground_truth_data
        
        # Simulation parameters
        self.beta = 0.15
        self.total_days = 83
        self.checkpoint_interval = 10  # Save every 10 days
        
        print(f"✓ Full simulation initialized")
        print(f"  Period: 83 days (2020-01-23 to 2020-04-14)")
        print(f"  Fixed beta: {self.beta}")
        print(f"  Checkpoint interval: every {self.checkpoint_interval} days")
    
    def _initialize_agents(self) -> List[Dict]:
        """
        Initialize agent states for day 0.
        
        Returns:
            Profiles with initial states
        """
        profiles = deepcopy(self.base_profiles)
        
        # Start with mostly susceptible, a few infected
        num_agents = len(profiles)
        num_initial_infected = 1  # Small seed
        
        np.random.shuffle(profiles)
        
        for i, profile in enumerate(profiles):
            if i < num_initial_infected:
                profile['disease_status'] = 2  # I
            else:
                profile['disease_status'] = 0  # S
        
        print(f"    Initialized: S={num_agents - num_initial_infected}, I={num_initial_infected}")
        
        return profiles
    
    def _run_simulation(self, profiles: List[Dict], start_day: int, end_day: int,
                       logger: ComprehensiveLogger = None,
                       checkpoint_dir: Path = None,
                       restored_agent_data: Dict = None) -> Dict:
        """
        Run simulation from start_day to end_day.
        
        Args:
            profiles: Agent profiles (disease_status set, but SimulationManager will ignore)
            start_day: Starting day
            end_day: Ending day (inclusive)
            logger: Logger for reasoning
            checkpoint_dir: Directory for checkpoints
            restored_agent_data: ACTUAL agent state data to force restore
            
        Returns:
            Simulation results
        """
        print(f"\n  Running simulation: days {start_day}-{end_day}")
        
        # Initialize simulation manager (will ignore our profiles and use defaults)
        sim = SimulationManager(
            profiles=deepcopy(profiles),
            edges=self.edges,
            cluster_assignments=self.cluster_assignments,
            graphsage_embeddings=self.embeddings
        )
        
        # ✅ FORCE RESTORE AFTER SimulationManager's stubborn initialization
        if restored_agent_data is not None:
            print(f"\n  💪 FORCING STATE RESTORATION (overriding SimulationManager defaults)")
            
            restored_count = 0
            for agent in sim.entity_agents:
                agent_id_str = str(agent.agent_id)
                
                if agent_id_str in restored_agent_data:
                    saved = restored_agent_data[agent_id_str]
                    
                    # FORCE override all state
                    agent.state = saved['state']
                    agent.days_in_state = saved['days_in_state']
                    agent.is_dead = saved.get('is_dead', False)
                    agent.initial_state = saved.get('initial_state', agent.state)
                    agent.state_sequence = saved.get('state_sequence', [agent.state])
                    agent.transition_history = saved.get('transition_history', [])
                    
                    restored_count += 1
            
            # Verify forced restoration
            forced_state = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
            for agent in sim.entity_agents:
                forced_state[agent.state] += 1
            
            print(f"  ✅ FORCED {restored_count} agents to checkpoint states")
            print(f"  ✅ VERIFIED: S={forced_state['S']}, E={forced_state['E']}, "
                  f"I={forced_state['I']}, R={forced_state['R']}, D={forced_state['D']}")
            print()
        else:
            # Initialize tracking for new simulation
            for agent in sim.entity_agents:
                agent.initial_state = agent.state
                agent.state_sequence = [agent.state]
                agent.transition_history = []
        
        # Set beta
        EpidemicParams.TRANSMISSION_PROBABILITY = self.beta
        
        # Initialize trackers
        phase_tracker = EpidemicPhaseTracker(population_size=len(sim.entity_agents))
        r0_tracker = R0Tracker(population_size=len(sim.entity_agents))
        
        # Inject trackers
        for cluster in sim.cluster_teams.values():
            cluster.phase_tracker = phase_tracker
            cluster.transmission_boost = 1.0
            for state_agent in cluster.state_agents.values():
                state_agent.phase_tracker = phase_tracker
        
        # Storage
        state_history = []
        agent_trajectories = {agent.agent_id: [] for agent in sim.entity_agents}
        cluster_stats = {cluster_id: [] for cluster_id in sim.cluster_teams.keys()}
        
        # Run day by day
        num_days = end_day - start_day + 1
        
        for t in range(num_days):
            actual_day = start_day + t
            timestep_start = time.time()
            
            # Progress indicator
            if actual_day % 10 == 0:
                print(f"    Day {actual_day}/{self.total_days-1}...")
            
            # Track transitions
            timestep_transitions = []
            timestep_deaths = []
            cluster_probabilities = {}
            
            # Update phase tracker
            current_state = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
            for agent in sim.entity_agents:
                current_state[agent.state] += 1
            
            phase_tracker.update(actual_day, current_state)
            
            # Dynamic boost
            if phase_tracker.current_phase == "EXPONENTIAL_GROWTH":
                boost = 1.5
            elif phase_tracker.current_phase == "PEAK":
                boost = 1.0
            else:
                boost = 1.0
            
            for cluster in sim.cluster_teams.values():
                cluster.transmission_boost = boost
            
            # Process clusters
            for cluster_id, cluster in sim.cluster_teams.items():
                neighbor_context = sim._get_neighbor_context(cluster_id)
                cluster_result = cluster.full_reasoning_cycle(actual_day, neighbor_context)
                
                cluster_probs = cluster_result.get('cluster_probabilities', {})
                cluster_probabilities[cluster_id] = cluster_probs
                
                # Track cluster stats
                cluster_state_counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
                for agent in cluster.entity_agents:
                    cluster_state_counts[agent.state] += 1
                
                cluster_stats[cluster_id].append({
                    'day': actual_day,
                    'state_counts': cluster_state_counts,
                    'probabilities': {k: v.get('fused', 0.0) for k, v in cluster_probs.items()},
                    'agent_ids': [a.agent_id for a in cluster.entity_agents]
                })
                
                # Track agent results
                agent_results = cluster_result.get('agent_results', {})
                for agent_id, result in agent_results.items():
                    agent = next((a for a in cluster.entity_agents if a.agent_id == agent_id), None)
                    
                    if agent:
                        # Log agent reasoning
                        if logger:
                            logger.log_agent_reasoning(actual_day, agent_id, agent.name, result)
                        
                        # Update state sequence if transitioned
                        if result.get('transitioned'):
                            agent.state_sequence.append(agent.state)
                            agent.transition_history.append({
                                'day': actual_day,
                                'from_state': result.get('old_state'),
                                'to_state': result.get('new_state'),
                                'probability': result.get('probability', 0.0)
                            })
                        
                        # Track transitions
                        if result.get('transitioned'):
                            transition_info = {
                                'day': actual_day,
                                'agent_id': agent_id,
                                'transition': f"{result.get('old_state')}->{result.get('new_state')}",
                                'probability': result.get('probability', 0),
                                'reasoning': result.get('reasoning', '')
                            }
                            
                            if result.get('new_state') == 'D':
                                timestep_deaths.append(transition_info)
                            
                            timestep_transitions.append(transition_info)
                        
                        # Store trajectory
                        agent_trajectories[agent_id].append({
                            'day': actual_day,
                            'state': agent.state,
                            'days_in_state': agent.days_in_state,
                            'transitioned': result.get('transitioned', False),
                            'probability': result.get('probability', 0.0),
                            'old_state': result.get('old_state'),
                            'new_state': result.get('new_state')
                        })
            
            # Record state
            current_state = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
            for agent in sim.entity_agents:
                current_state[agent.state] += 1
            
            state_history.append({
                'day': actual_day,
                'counts': current_state
            })
            
            # Log timestep
            if logger:
                timestep_elapsed = time.time() - timestep_start
                logger.log_timestep_summary(actual_day, {
                    'state_dist': current_state,
                    'transitions': timestep_transitions,
                    'deaths': timestep_deaths,
                    'cluster_probs': cluster_probabilities,
                    'execution_time': timestep_elapsed
                })
            
            # CHECKPOINT every N days
            if checkpoint_dir and (actual_day + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    checkpoint_dir,
                    actual_day,
                    state_history,
                    agent_trajectories,
                    cluster_stats,
                    sim.entity_agents
                )
        
        return {
            'beta': self.beta,
            'state_history': state_history,
            'agent_trajectories': agent_trajectories,
            'cluster_stats': cluster_stats,
            'final_state': state_history[-1]['counts']
        }
    
    def _save_checkpoint(self, checkpoint_dir: Path, current_day: int,
                        state_history: List, agent_trajectories: Dict,
                        cluster_stats: Dict, agents: List):
        """Save checkpoint at current day with COMPLETE agent state."""
        checkpoint_file = checkpoint_dir / f"checkpoint_day_{current_day:03d}.json"
        
        # Get COMPLETE agent states
        agent_states = {}
        for agent in agents:
            agent_states[str(agent.agent_id)] = {
                'state': agent.state,
                'days_in_state': agent.days_in_state,
                'is_dead': agent.is_dead,
                'initial_state': agent.initial_state,
                'state_sequence': agent.state_sequence,
                'transition_history': agent.transition_history
            }
        
        checkpoint_data = {
            'current_day': current_day,
            'beta': self.beta,
            'agent_states': agent_states,
            'state_history': state_history,
            'agent_trajectories': agent_trajectories,
            'cluster_stats': cluster_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"    💾 Checkpoint saved: day {current_day}")
    
    def _load_latest_checkpoint(self, checkpoint_dir: Path) -> Tuple[int, Dict]:
        """
        Load the most recent checkpoint.
        
        Returns:
            (last_completed_day, checkpoint_data)
        """
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_day_*.json"))
        
        if not checkpoints:
            return -1, None
        
        latest = checkpoints[-1]
        with open(latest, 'r') as f:
            data = json.load(f)
        
        print(f"  ✓ Loaded checkpoint from day {data['current_day']}")
        print(f"    Timestamp: {data['timestamp']}")
        return data['current_day'], data
    
    def run(self, logger: ComprehensiveLogger = None,
            checkpoint_dir: Path = None, resume: bool = True) -> Dict:
        """
        Run full 83-day simulation with checkpointing.
        
        Args:
            logger: Logger for agent reasoning
            checkpoint_dir: Directory for checkpoints
            resume: If True, resume from last checkpoint
            
        Returns:
            Complete simulation results
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path("outputs/rolling_window")
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("FULL 83-DAY SIMULATION WITH FIXED BETA")
        print(f"Period: 2020-01-23 to 2020-04-14")
        print(f"Beta: {self.beta}")
        print(f"Checkpointing: every {self.checkpoint_interval} days")
        print("="*80)
        
        # Check for existing checkpoint
        start_day = 0
        profiles = None
        existing_data = None
        restored_agent_data = None
        
        if resume:
            last_day, checkpoint_data = self._load_latest_checkpoint(checkpoint_dir)
            
            if last_day >= 0 and checkpoint_data:
                # ✅ CHECKPOINT FOUND
                print(f"  🔄 RESUMING FROM CHECKPOINT (Day {last_day})")
                
                start_day = last_day + 1
                existing_data = checkpoint_data
                restored_agent_data = checkpoint_data.get('agent_states', {})
                
                # Use base profiles (SimulationManager will ignore disease_status anyway)
                profiles = deepcopy(self.base_profiles)
                
                print(f"  ✅ Will force restore {len(restored_agent_data)} agent states AFTER SimulationManager init")
            else:
                # ❌ NO CHECKPOINT
                print(f"  🆕 NO CHECKPOINT - STARTING FRESH")
                profiles = self._initialize_agents()
                restored_agent_data = None
        else:
            # ❌ RESUME=FALSE
            print(f"  🆕 STARTING FRESH (resume=False)")
            profiles = self._initialize_agents()
            restored_agent_data = None
        
        # Run simulation
        if start_day < self.total_days:
            try:
                result = self._run_simulation(
                    profiles=profiles,
                    start_day=start_day,
                    end_day=self.total_days - 1,
                    logger=logger,
                    checkpoint_dir=checkpoint_dir,
                    restored_agent_data=restored_agent_data
                )
                
                # Merge with existing data if resuming
                if existing_data:
                    print(f"\n  🔗 Merging with existing data...")
                    
                    # Append new data to existing
                    result['state_history'] = existing_data['state_history'] + result['state_history']
                    
                    # Merge agent trajectories
                    for agent_id_str, traj in existing_data['agent_trajectories'].items():
                        agent_id = int(agent_id_str) if isinstance(agent_id_str, str) else agent_id_str
                        if agent_id in result['agent_trajectories']:
                            result['agent_trajectories'][agent_id] = traj + result['agent_trajectories'][agent_id]
                        else:
                            result['agent_trajectories'][agent_id] = traj
                    
                    # Merge cluster stats
                    for cluster_id_str, stats in existing_data['cluster_stats'].items():
                        cluster_id = int(cluster_id_str) if isinstance(cluster_id_str, str) else cluster_id_str
                        if cluster_id in result['cluster_stats']:
                            result['cluster_stats'][cluster_id] = stats + result['cluster_stats'][cluster_id]
                        else:
                            result['cluster_stats'][cluster_id] = stats
                    
                    print(f"  ✅ Merge complete")
                
                print(f"\n  ✅ SIMULATION COMPLETE: all {self.total_days} days")
                
                # Clean up checkpoints on successful completion
                print(f"\n  🧹 Cleaning up checkpoints...")
                for checkpoint_file in checkpoint_dir.glob("checkpoint_day_*.json"):
                    checkpoint_file.unlink()
                print(f"  ✅ Checkpoints deleted")
                
                return result
                
            except KeyboardInterrupt:
                print("\n\n⚠️  INTERRUPTED! Progress saved.")
                if 'result' in locals():
                    completed_days = start_day + len(result.get('state_history', []))
                    print(f"   Completed up to day {completed_days - 1}")
                else:
                    print(f"   Last checkpoint: day {start_day - 1}")
                print(f"   Resume by running the script again.")
                raise
        else:
            print(f"  ✅ Already complete (day {start_day-1})")
            return existing_data


def load_singapore_ground_truth() -> Dict:
    """Load Singapore MOH ground truth data"""
    print("Loading Singapore ground truth data...")
    
    num_days = 83
    
    # Synthetic COVID-19 curve
    early = np.random.poisson(lam=3, size=21)
    growth_base = np.linspace(3, 50, 25)
    growth_noise = np.random.normal(0, 5, 25)
    growth = np.maximum(0, growth_base + growth_noise).astype(int)
    plateau = np.random.poisson(lam=45, size=20)
    continued = np.random.poisson(lam=40, size=17)
    
    daily_cases = np.concatenate([early, growth, plateau, continued])
    cumulative_cases = np.cumsum(daily_cases)
    deaths = (cumulative_cases * 0.001).astype(int)
    
    print(f"  ✓ Loaded {num_days} days of ground truth")
    print(f"    Total cases: {cumulative_cases[-1]}")
    print(f"    ⚠️  PLACEHOLDER - Replace with real MOH data")
    
    return {
        'daily_cases': daily_cases,
        'cumulative_cases': cumulative_cases,
        'deaths': deaths,
        'start_date': '2020-01-23',
        'end_date': '2020-04-14',
        'num_days': 83
    }


def main():
    print("\n" + "="*80)
    print("FULL 83-DAY EPIDEMIC SIMULATION")
    print("Fixed Beta = 0.15 | Checkpoint/Resume Fixed")
    print("="*80 + "\n")

    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        return
    
    print("✓ OpenAI API key found")
    
    # Check neural model
    neural_model_path = Path("data/processed/neural_model/best_model.pt")
    if not neural_model_path.exists():
        print("⚠ WARNING: No trained neural model found!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    else:
        print("✓ Trained neural model found")
    
    # Initialize directories
    output_dir = Path("data/results")
    simulation_output_dir = Path("outputs/rolling_window")
    simulation_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ComprehensiveLogger(output_dir)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    raw_dir = Path("data/singapore")
    processed_dir = Path("data/processed")

    with open(raw_dir / "profiles.json", 'r') as f:
        profiles = json.load(f)
    print(f"✓ Loaded {len(profiles)} agent profiles")
    
    with open(raw_dir / "edges.json", 'r') as f:
        edges = json.load(f)
    print(f"✓ Loaded {len(edges)} edges")
    
    with open(processed_dir / "cluster_assignments_hsbc3.json", 'r') as f:
        cluster_assignments = json.load(f)
        cluster_assignments = {int(k): v for k, v in cluster_assignments.items()}
    print(f"✓ Loaded {len(cluster_assignments)} cluster assignments")
    
    embeddings = np.load(processed_dir / "graphsage/graphsage_embeddings.npy")
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Load ground truth
    ground_truth = load_singapore_ground_truth()
    
    # Initialize simulation
    print("\n" + "="*80)
    print("INITIALIZING SIMULATION")
    print("="*80)
    
    full_sim = FullSimulation(
        profiles=profiles,
        edges=edges,
        cluster_assignments=cluster_assignments,
        embeddings=embeddings,
        ground_truth_data=ground_truth
    )
    
    # Run simulation
    try:
        result = full_sim.run(
            logger=logger,
            checkpoint_dir=simulation_output_dir,
            resume=True
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("INTERRUPTED - PROGRESS SAVED")
        print("="*80)
        print("\nYour progress has been saved!")
        print("Simply run the script again to resume.")
        return
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("\nCheck for saved checkpoints and try again.")
        import traceback
        traceback.print_exc()
        return
    
    # Save final results
    print("\n" + "="*80)
    print("SAVING FINAL RESULTS")
    print("="*80)
    
    # Save complete results
    with open(simulation_output_dir / "simulation_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'simulation_results.json'}")
    
    # Save agent trajectories
    with open(simulation_output_dir / "agent_trajectories.json", 'w') as f:
        json.dump(result['agent_trajectories'], f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'agent_trajectories.json'}")
    
    # Save state history
    with open(simulation_output_dir / "state_history.json", 'w') as f:
        json.dump(result['state_history'], f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'state_history.json'}")
    
    # Save cluster stats
    with open(simulation_output_dir / "cluster_stats.json", 'w') as f:
        json.dump(result['cluster_stats'], f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'cluster_stats.json'}")
    
    # Save agent profiles
    agent_profiles = []
    for agent_id in range(len(profiles)):
        profile_data = profiles[agent_id]
        agent_profiles.append({
            'agent_id': agent_id,
            'age': profile_data['age'],
            'vaccination_status': profile_data['vaccination_status'],
            'occupation': profile_data['occupation'],
            'household_id': profile_data['household_id'],
            'comorbidity_count': profile_data.get('comorbidity_count', 0),
            'cluster_id': cluster_assignments.get(agent_id, -1)
        })
    
    with open(simulation_output_dir / "agent_profiles.json", 'w') as f:
        json.dump(agent_profiles, f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'agent_profiles.json'}")
    
    # Save edges
    with open(simulation_output_dir / "edges.json", 'w') as f:
        json.dump(edges, f, indent=2)
    print(f"✓ Saved: {simulation_output_dir / 'edges.json'}")
    
    # Log simulation summary
    logger.log_simulation_summary({
        'num_agents': len(profiles),
        'num_clusters': len(cluster_assignments),
        'num_days': 83,
        'beta': 0.15,
        'final_state': result['final_state']
    })
    
    # Print summary
    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETE")
    print("="*80)
    print(f"\nDays simulated: 83 (2020-01-23 to 2020-04-14)")
    print(f"Agent trajectories: {len(result['agent_trajectories'])} agents")
    print(f"Beta: {result['beta']}")
    
    final = result['final_state']
    print(f"\nFinal state:")
    print(f"  S={final['S']}, E={final['E']}, I={final['I']}, R={final['R']}, D={final['D']}")
    
    print(f"\n✓ Comprehensive logs: {logger.logs_dir}")
    print(f"✓ Results: {simulation_output_dir}")
    print("\n")


if __name__ == "__main__":
    main()
