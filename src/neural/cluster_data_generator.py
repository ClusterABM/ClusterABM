"""
Cluster-level MULTIMODAL training data generation for neural pathway.
UPDATED FOR SINGAPORE COVID-19 DATA.

MAINTAINS FULL MULTIMODAL STRUCTURE:
- Tabular: Cluster-aggregated static + dynamic features (Singapore-specific)
- Temporal: Cluster-level epidemic time series (28-day lookback)
- Graph: Cluster-level graph embeddings
- Text: Cluster clinical summary embeddings (optional)

Output: 5 transition rates [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]

SINGAPORE ENHANCEMENTS:
- Dormitory cluster detection and features
- Quarantine Order (QO) effects
- Imported case tracking
- Singapore healthcare quality (0.3x IFR)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from pathlib import Path
import json
from collections import defaultdict, Counter


class SingaporeClusterMultimodalDataGenerator:
    """
    Generate CLUSTER-LEVEL multimodal training data for Singapore COVID-19.
    
    Maintains tabular, temporal, and graph modalities at cluster granularity.
    """
    
    def __init__(
        self,
        profiles: List[Dict],
        edges: List[Dict],
        cluster_assignments: Dict[int, int],
        graphsage_embeddings: np.ndarray
    ):
        self.profiles = profiles
        self.edges = edges
        self.cluster_assignments = cluster_assignments
        self.graphsage_embeddings = graphsage_embeddings
        
        self.agent_dict = {p['agent_id']: p for p in profiles}
        self.adj_list = self._build_adjacency_list()
        self.cluster_ids = sorted(set(cluster_assignments.values()))
        # Load Kaggle Singapore data for initial states
        self.kaggle_data = self._load_kaggle_initial_states()
        
        print(f"  Kaggle initial state distribution:")
        for state, count in sorted(self.kaggle_data['state_counts'].items()):
            print(f"    {state}: {count} ({count/len(profiles)*100:.1f}%)")
        
        self._init_viral_load_curves()
        self._init_symptom_progressions()
        self._init_contact_patterns()
        
        print(f"✓ Singapore cluster multimodal data generator initialized")
        print(f"  Agents: {len(profiles)}, Clusters: {len(self.cluster_ids)}")
    
    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge['agent_1']].append(edge['agent_2'])
            adj[edge['agent_2']].append(edge['agent_1'])
        return dict(adj)

    def _load_kaggle_initial_states(self) -> Dict:
        """
        Load initial state distribution from Kaggle Singapore data (day_0 column).
        
        Logic:
        - day_0 column contains STRINGS: 'S', 'E', 'I', 'R', 'D'
        - Count actual E and I from Kaggle
        - R and D are 0 at outbreak start
        - S = 1000 - (E + I)
        
        Returns:
            Dict with state_counts and mapping
        """
        import pandas as pd
        from pathlib import Path
        
        # Try to load Kaggle data
        kaggle_path = Path("singapore_covid19_cases.csv")
        if not kaggle_path.exists():
            print("  ⚠️  Kaggle data not found, using default distribution")
            return {
                'state_counts': {'S': 991, 'E': 8, 'I': 1, 'R': 0, 'D': 0},
                'available': False
            }
        
        df = pd.read_csv(kaggle_path)
        total_agents = 1000  # Our simulation size
        
        # Count states from day_0 column (STRINGS, not integers!)
        day_0_counts = df['day_0'].value_counts().to_dict()
        
        # Extract counts from Kaggle data (use string keys)
        kaggle_e = day_0_counts.get('E', 0)  # Exposed
        kaggle_i = day_0_counts.get('I', 0)  # Infected
        kaggle_r = day_0_counts.get('R', 0)  # Recovered (should be 0)
        kaggle_d = day_0_counts.get('D', 0)  # Dead (should be 0)
        kaggle_s = day_0_counts.get('S', 0)  # Susceptible
        
        kaggle_total = len(df)
        
        # Scale to our 1000 agents
        # Preserve E and I proportions, force R=D=0 for outbreak start
        if kaggle_e + kaggle_i > 0:
            # Scale infected population to our size
            scale_factor = total_agents / kaggle_total
            e_count = int(kaggle_e * scale_factor)
            i_count = int(kaggle_i * scale_factor)
        else:
            # Fallback if no infections in Kaggle data
            e_count = 10
            i_count = 10
        
        # Outbreak start: no recovered or dead yet
        r_count = 0
        d_count = 0
        
        # Remainder are susceptible
        s_count = total_agents - (e_count + i_count + r_count + d_count)
        
        # Ensure no negatives
        if s_count < 0:
            print(f"  ⚠️  WARNING: Negative S count ({s_count}), adjusting")
            total_infected = e_count + i_count
            scale = (total_agents * 0.98) / total_infected  # Leave 2% as S
            e_count = int(e_count * scale)
            i_count = int(i_count * scale)
            s_count = total_agents - (e_count + i_count)
        
        state_counts = {
            'S': s_count,
            'E': e_count,
            'I': i_count,
            'R': r_count,
            'D': d_count
        }
        
        return {
            'state_counts': state_counts,
            'available': True,
            'total': total_agents,
            'kaggle_total': kaggle_total,
            'kaggle_raw_counts': {
                'S': kaggle_s,
                'E': kaggle_e,
                'I': kaggle_i,
                'R': kaggle_r,
                'D': kaggle_d
            },
            'scale_factor': total_agents / kaggle_total
        }
    
    def _init_viral_load_curves(self):
        """Singapore-calibrated viral load curves."""
        self.viral_curves = {
            'typical': np.array([0, 0, 2, 4, 6, 7, 8, 7.5, 7, 6, 5, 4, 3, 2, 1, 0.5, 0, 0, 0, 0]),
            'vaccinated': np.array([0, 0, 2, 4, 5, 6, 5.5, 5, 4, 3, 2, 1, 0.5, 0, 0, 0, 0, 0, 0, 0]),
            'elderly': np.array([0, 0, 2, 5, 7, 8, 8.5, 8, 7.5, 7, 6, 5, 4, 3, 2, 1.5, 1, 0.5, 0, 0]),
        }
    
    def _init_symptom_progressions(self):
        self.symptom_curves = {
            'mild': np.array([0, 0, 1, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'moderate': np.array([0, 0, 2, 4, 6, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        }
    
    def _init_contact_patterns(self):
        """Singapore contact patterns by age group."""
        self.contact_rates = {
            '0-17': (12.5, 8.2),
            '18-29': (10.8, 9.5),
            '30-49': (9.2, 7.8),
            '50-64': (7.5, 6.2),
            '65+': (5.3, 4.8)
        }
    
    def _is_dormitory_cluster(self, cluster_agents: List[int]) -> bool:
        """
        Detect if cluster is a dormitory cluster (Singapore-specific).
        
        Returns:
            True if dormitory cluster detected
        """
        total = len(cluster_agents)
        if total == 0:
            return False
        
        # Check cluster names
        dorm_agents = sum(1 for a in cluster_agents 
                         if 'dorm' in str(self.agent_dict[a].get('cluster', '')).lower())
        
        if dorm_agents > total * 0.3:  # >30% have dorm in cluster name
            return True
        
        # Check occupation (manual workers)
        manual_workers = sum(1 for a in cluster_agents 
                            if self.agent_dict[a].get('occupation') == 'manual_worker')
        
        if manual_workers > total * 0.5:  # >50% manual workers
            return True
        
        return False
    
    # =========================================================================
    # TABULAR: Cluster-level static + dynamic features (SINGAPORE-SPECIFIC)
    # =========================================================================
    
    def _generate_cluster_tabular(
        self,
        cluster_agents: List[int],
        day: int,
        sim_data: Dict
    ) -> np.ndarray:
        """
        Cluster-level tabular features (37 dimensions for Singapore).
        
        Features:
        - State distribution (5): S, E, I, R, D proportions
        - Demographics (11): age, vaccination, comorbidities, Singapore-specific
        - Network (7): degree, density, external exposure
        - Epidemic (10): prevalence, days in state, contact exposure, trends
        - Neural-specific (4): household mixing, age assortativity, behavioral heterogeneity, SAR potential
        
        COMPLEMENTARITY: Neural captures micro-heterogeneities that LLMs cannot reason about.
        """
        features = []
        total = len(cluster_agents)
        
        # State distribution (5)
        state_counts = Counter([sim_data[a]['states'][day] for a in cluster_agents])
        for state in ['S', 'E', 'I', 'R', 'D']:
            features.append(state_counts.get(state, 0) / total)
        
        # Demographics (11) - includes Singapore-specific
        ages = [self.agent_dict[a]['age'] for a in cluster_agents]
        features.append(np.mean(ages) / 100.0)
        features.append(np.std(ages) / 50.0)
        
        vacc = [self.agent_dict[a]['vaccination_status'] for a in cluster_agents]
        features.append(np.mean(vacc) / 2.0)
        
        comorbid = [self.agent_dict[a]['comorbidity_count'] for a in cluster_agents]
        features.append(np.mean(comorbid) / 5.0)
        
        high_risk = ['healthcare_worker', 'teacher', 'retail_worker']
        occupations = [self.agent_dict[a]['occupation'] for a in cluster_agents]
        features.append(sum(1 for o in occupations if o in high_risk) / total)
        
        features.append(np.mean([self.agent_dict[a].get('compliance_score', 0.7) for a in cluster_agents]))
        features.append(np.mean([self.agent_dict[a].get('mobility_score', 0.7) for a in cluster_agents]))
        
        # ⭐ SINGAPORE-SPECIFIC FEATURES
        # Gender distribution (male-dominated dormitories)
        male_count = sum(1 for a in cluster_agents 
                        if self.agent_dict[a].get('gender', 'unknown') == 'm')
        features.append(male_count / total)
        
        # Imported case rate
        imported = sum(1 for a in cluster_agents 
                      if self.agent_dict[a].get('is_imported', False))
        features.append(imported / total)
        
        # Quarantine rate
        quarantined = sum(1 for a in cluster_agents 
                         if self.agent_dict[a].get('quarantine_status', False))
        features.append(quarantined / total)
        
        # Dormitory cluster indicator
        is_dormitory = 1.0 if self._is_dormitory_cluster(cluster_agents) else 0.0
        features.append(is_dormitory)
        
        # Network (7)
        degrees = [len(self.adj_list.get(a, [])) for a in cluster_agents]
        features.append(np.mean(degrees) / 50.0)
        features.append(np.std(degrees) / 25.0)
        
        # Intra-cluster density
        intra_edges = sum(
            sum(1 for n in self.adj_list.get(a, []) if n in cluster_agents)
            for a in cluster_agents
        )
        max_intra = total * (total - 1)
        features.append(intra_edges / max(max_intra, 1))
        
        # External exposure
        external_infected = sum(
            sum(1 for n in self.adj_list.get(a, []) 
                if n not in cluster_agents and sim_data[n]['states'][day] in ['E', 'I'])
            for a in cluster_agents
        )
        features.append(external_infected / max(total, 1))
        
        # Graph embedding stats
        cluster_emb = self.graphsage_embeddings[cluster_agents]
        features.append(cluster_emb.mean())
        features.append(cluster_emb.std())
        features.append(0.5)  # Placeholder
        
        # Epidemic dynamics (10)
        infected = state_counts.get('I', 0) + state_counts.get('E', 0)
        features.append(infected / total)  # Prevalence
        
        # Avg days in state by state
        for state in ['S', 'E', 'I', 'R']:
            agents_in_state = [a for a in cluster_agents if sim_data[a]['states'][day] == state]
            if agents_in_state:
                avg_days = np.mean([sim_data[a]['days_in_state'][day] for a in agents_in_state])
                max_days = {'S': 30, 'E': 10, 'I': 20, 'R': 90}[state]
                features.append(min(avg_days / max_days, 1.0))
            else:
                features.append(0.0)
        
        # Contact exposure for susceptible
        s_agents = [a for a in cluster_agents if sim_data[a]['states'][day] == 'S']
        if s_agents:
            s_exposure = np.mean([
                sum(1 for n in self.adj_list.get(a, []) if sim_data[n]['states'][day] in ['E', 'I'])
                for a in s_agents
            ])
            features.append(s_exposure / 10.0)
        else:
            features.append(0.0)
        
        # Recent trend
        if day >= 3:
            prev_infected = sum(1 for a in cluster_agents if sim_data[a]['states'][day-3] in ['E', 'I'])
            trend = (infected - prev_infected) / total
            features.append(np.clip(trend, -0.5, 0.5) + 0.5)
        else:
            features.append(0.5)
        
        # Cumulative attack rate
        cumulative = sum(
            1 for a in cluster_agents 
            if any(s in ['E', 'I', 'R', 'D'] for s in sim_data[a]['states'][:day+1])
        )
        features.append(cumulative / total)
        
        # Day (simulation time)
        features.append(day / 200.0)
        
        # Vaccination campaign progress (Singapore started Dec 2020, data is early 2020)
        features.append(0.0)  # No vaccination in early 2020 Singapore

        # 1. Network mixing patterns (household vs community structure)
        household_edges = sum(
            1 for a in cluster_agents
            for n in self.adj_list.get(a, [])
            if n in cluster_agents and self.agent_dict[n].get('household_id') == self.agent_dict[a].get('household_id')
        )
        total_edges = sum(len(self.adj_list.get(a, [])) for a in cluster_agents)
        features.append(household_edges / max(total_edges, 1))  # Household contact fraction
        
        # 2. Age-contact assortativity (key for transmission heterogeneity)
        young_agents = [a for a in cluster_agents if self.agent_dict[a]['age'] < 18]
        old_agents = [a for a in cluster_agents if self.agent_dict[a]['age'] > 65]
        young_to_old_contacts = sum(
            1 for a in young_agents
            for n in self.adj_list.get(a, [])
            if n in old_agents
        )
        total_young_contacts = sum(len(self.adj_list.get(a, [])) for a in young_agents)
        features.append(young_to_old_contacts / max(total_young_contacts, 1))  # Intergenerational mixing

        # 3. Behavioral heterogeneity (std of compliance/mobility)
        compliance_scores = [self.agent_dict[a].get('compliance_score', 0.7) for a in cluster_agents]
        features.append(np.std(compliance_scores))  # High variance = heterogeneous response
        
        # 4. Secondary attack potential (infected × their susceptible contacts)
        infected_agents = [a for a in cluster_agents if sim_data[a]['states'][day] in ['E', 'I']]
        sar_potential = sum(
            sum(1 for n in self.adj_list.get(a, []) if sim_data[n]['states'][day] == 'S')
            for a in infected_agents
        )
        features.append(sar_potential / max(total, 1))  # Potential for within-cluster spread
        
        # Ensure exactly 37 dimensions (33 + 4 neural-specific)
        return np.array(features[:37], dtype=np.float32)
    
    # =========================================================================
    # TEMPORAL: Cluster-level time series (lookback window)
    # =========================================================================
    
    def _generate_cluster_temporal(
        self,
        cluster_agents: List[int],
        current_day: int,
        sim_data: Dict,
        lookback: int = 7
    ) -> np.ndarray:
        """
        Cluster-level temporal features [lookback+1, 10].
        
        Features per timestep:
        - State proportions (5): S, E, I, R, D
        - Incidence (2): new E, new I
        - Prevalence (1): active infections
        - Contact rate (1): avg contacts
        - Policy/environmental (1): Circuit Breaker stringency
        """
        temporal_seq = []
        
        for t in range(max(0, current_day - lookback), current_day + 1):
            features_t = []
            
            # State proportions
            state_counts = Counter([sim_data[a]['states'][t] for a in cluster_agents])
            total = len(cluster_agents)
            for state in ['S', 'E', 'I', 'R', 'D']:
                features_t.append(state_counts.get(state, 0) / total)
            
            # Incidence (new transitions)
            if t > 0:
                new_e = sum(
                    1 for a in cluster_agents
                    if sim_data[a]['states'][t] == 'E' and sim_data[a]['states'][t-1] == 'S'
                )
                new_i = sum(
                    1 for a in cluster_agents
                    if sim_data[a]['states'][t] == 'I' and sim_data[a]['states'][t-1] == 'E'
                )
                features_t.append(new_e / total)
                features_t.append(new_i / total)
            else:
                features_t.extend([0.0, 0.0])
            
            # Active infections
            active = (state_counts.get('E', 0) + state_counts.get('I', 0)) / total
            features_t.append(active)
            
            # Contact rate (Singapore-specific: dormitory vs community)
            is_dormitory = self._is_dormitory_cluster(cluster_agents)
            avg_contacts = 15.0 if is_dormitory else 8.0  # Higher in dormitories
            features_t.append(avg_contacts / 20.0)
            
            # Policy stringency (Singapore Circuit Breaker approximation)
            # Day 0-50: Pre-CB (low stringency)
            # Day 50-110: Circuit Breaker (high stringency)
            # Day 110+: Phased reopening (medium stringency)
            if t < 50:
                stringency = 30.0  # Pre-CB
            elif t < 110:
                stringency = 90.0  # Circuit Breaker
            else:
                stringency = 60.0  # Post-CB
            
            features_t.append(stringency / 100.0)
            
            temporal_seq.append(features_t)
        
        # Pad if needed
        while len(temporal_seq) < lookback + 1:
            temporal_seq.insert(0, np.zeros(10, dtype=np.float32))
        
        return np.array(temporal_seq[-lookback-1:], dtype=np.float32)  # [8, 10]
    
    # =========================================================================
    # GRAPH: Cluster-level graph embedding
    # =========================================================================
    
    def _generate_cluster_graph(
        self,
        cluster_agents: List[int]
    ) -> np.ndarray:
        """
        Cluster-level graph embedding (128 dimensions).
        
        Aggregate agent embeddings to cluster level.
        """
        cluster_embeddings = self.graphsage_embeddings[cluster_agents]
        
        # Mean pooling
        cluster_embedding = cluster_embeddings.mean(axis=0)
        
        return cluster_embedding.astype(np.float32)
    
    # =========================================================================
    # LABELS: Transition rates
    # =========================================================================
    
    def _calculate_transition_rates(
        self,
        cluster_agents: List[int],
        day: int,
        sim_data: Dict
    ) -> np.ndarray:
        """
        Calculate actual transition rates [5,].
        
        [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
        """
        rates = np.zeros(5, dtype=np.float32)
        
        # S→E
        s_agents = [a for a in cluster_agents if sim_data[a]['states'][day] == 'S']
        if s_agents:
            s_to_e = sum(1 for a in s_agents if sim_data[a]['states'][day+1] == 'E')
            rates[0] = s_to_e / len(s_agents)
        
        # E→I
        e_agents = [a for a in cluster_agents if sim_data[a]['states'][day] == 'E']
        if e_agents:
            e_to_i = sum(1 for a in e_agents if sim_data[a]['states'][day+1] == 'I')
            rates[1] = e_to_i / len(e_agents)
        
        # I→R
        i_agents = [a for a in cluster_agents if sim_data[a]['states'][day] == 'I']
        if i_agents:
            i_to_r = sum(1 for a in i_agents if sim_data[a]['states'][day+1] == 'R')
            rates[2] = i_to_r / len(i_agents)
        
        # I→D
        if i_agents:
            i_to_d = sum(1 for a in i_agents if sim_data[a]['states'][day+1] == 'D')
            rates[3] = i_to_d / len(i_agents)
        
        # R→S
        r_agents = [a for a in cluster_agents if sim_data[a]['states'][day] == 'R']
        if r_agents:
            r_to_s = sum(1 for a in r_agents if sim_data[a]['states'][day+1] == 'S')
            rates[4] = r_to_s / len(r_agents)
        
        return rates
    
    # =========================================================================
    # MAIN GENERATION
    # =========================================================================
    
    def generate_multimodal_cluster_data(
        self,
        num_simulations: int = 100,
        timesteps_per_sim: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cluster-level multimodal training data.
        
        Returns:
            tabular: [N, 33] (SINGAPORE: 30 → 33)
            temporal: [N, 8, 10]
            graph: [N, 128]
            labels: [N, 5]
        """
        print(f"\nGenerating Singapore cluster-level multimodal data:")
        print(f"  {num_simulations} sims × {timesteps_per_sim} days × {len(self.cluster_ids)} clusters")
        
        all_tabular = []
        all_temporal = []
        all_graph = []
        all_labels = []
        
        for sim_idx in range(num_simulations):
            if (sim_idx + 1) % 20 == 0:
                print(f"  Simulation {sim_idx + 1}/{num_simulations}...")
            
            sim_data = self._run_singapore_seird_simulation(timesteps_per_sim)
            
            for day in range(timesteps_per_sim - 1):
                for cluster_id in self.cluster_ids:
                    cluster_agents = [
                        aid for aid, cid in self.cluster_assignments.items()
                        if cid == cluster_id
                    ]
                    
                    if not cluster_agents:
                        continue
                    
                    # Generate all modalities
                    tabular = self._generate_cluster_tabular(cluster_agents, day, sim_data)
                    temporal = self._generate_cluster_temporal(cluster_agents, day, sim_data)
                    graph = self._generate_cluster_graph(cluster_agents)
                    rates = self._calculate_transition_rates(cluster_agents, day, sim_data)
                    
                    all_tabular.append(tabular)
                    all_temporal.append(temporal)
                    all_graph.append(graph)
                    all_labels.append(rates)
        
        tabular = np.array(all_tabular, dtype=np.float32)
        temporal = np.array(all_temporal, dtype=np.float32)
        graph = np.array(all_graph, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.float32)
        
        print(f"\n✓ Generated {len(tabular)} cluster samples")
        print(f"  Tabular: {tabular.shape}")
        print(f"  Temporal: {temporal.shape}")
        print(f"  Graph: {graph.shape}")
        print(f"  Labels: {labels.shape}")
        
        return tabular, temporal, graph, labels
    
    # REPLACEMENT 3: COMPLETELY REPLACE _run_singapore_seird_simulation:

    def _run_singapore_seird_simulation(self, timesteps: int) -> Dict:
        """
        Run SEIRD simulation with KAGGLE-INITIALIZED states.
        
        KEY: Use actual Singapore day_0 distribution from Kaggle.
        """
        import random
        
        agent_states = {}
        
        # Get state distribution from Kaggle
        if self.kaggle_data['available']:
            total = self.kaggle_data['total']
            state_probs = {
                state: count / total 
                for state, count in self.kaggle_data['state_counts'].items()
            }
        else:
            # Fallback if no Kaggle data
            state_probs = {'S': 0.98, 'E': 0.01, 'I': 0.008, 'R': 0.002, 'D': 0.0}
        
        # Initialize agents with Kaggle-realistic distribution
        for agent in self.profiles:
            agent_id = agent['agent_id']
            
            # Sample initial state from Kaggle distribution
            rand = random.random()
            cumsum = 0
            initial_state = 'S'
            for state in ['S', 'E', 'I', 'R', 'D']:
                cumsum += state_probs.get(state, 0)
                if rand < cumsum:
                    initial_state = state
                    break
            
            agent_states[agent_id] = {
                'states': [initial_state],
                'days_in_state': [random.randint(0, 3) if initial_state != 'S' else 0]
            }
        
        # Run forward simulation
        for day in range(1, timesteps):
            new_states = {}
            for agent_id in range(len(self.profiles)):
                current_state = agent_states[agent_id]['states'][-1]
                days_in_state = agent_states[agent_id]['days_in_state'][-1]
                next_state = self._sample_singapore_next_state(
                    agent_id, current_state, days_in_state, agent_states
                )
                new_states[agent_id] = next_state
            
            for agent_id in range(len(self.profiles)):
                current_state = agent_states[agent_id]['states'][-1]
                next_state = new_states[agent_id]
                new_days = agent_states[agent_id]['days_in_state'][-1] + 1 if next_state == current_state else 0
                agent_states[agent_id]['states'].append(next_state)
                agent_states[agent_id]['days_in_state'].append(new_days)
        
        return agent_states
    
    def _sample_singapore_next_state(
        self,
        agent_id: int,
        current_state: str,
        days_in_state: int,
        all_agent_states: Dict
    ) -> str:
        """
        Sample next state with SINGAPORE-SPECIFIC parameters.
        
        Key differences:
        - Dormitory outbreaks: 2.5x transmission
        - Quarantine: 80% reduction
        - Healthcare: 0.3x IFR (lower mortality)
        """
        agent = self.agent_dict[agent_id]

        if current_state == 'S':
            # ═══════════════════════════════════════════════════════════════
            # S→E: NETWORK-BASED TRANSMISSION (Neural model specialization)
            # ═══════════════════════════════════════════════════════════════
            neighbors = self.adj_list.get(agent_id, [])
            
            if not neighbors:
                # Isolated agent - only community transmission
                base_prob = 0.002  # 0.2% per day baseline
            else:
                # Count infected contacts by relationship type
                infected_neighbors = [n for n in neighbors if all_agent_states[n]['states'][-1] in ['E', 'I']]
                
                if not infected_neighbors:
                    # No infected contacts - community baseline
                    base_prob = 0.002
                else:
                    # Household vs non-household transmission (KEY HETEROGENEITY)
                    household_infected = sum(
                        1 for n in infected_neighbors 
                        if self.agent_dict[n].get('household_id') == agent.get('household_id')
                    )
                    other_infected = len(infected_neighbors) - household_infected
                    
                    # Singapore household SAR: 18.3% over ~14 days = 1.4% per day
                    household_daily = 0.014 * household_infected
                    
                    # Other contacts: 8-10% SAR over 14 days = 0.6-0.7% per day
                    other_daily = 0.007 * other_infected
                    
                    base_prob = household_daily + other_daily
            
            # ⭐ DORMITORY AMPLIFICATION (2.5x in crowded settings)
            cluster_name = agent.get('cluster', '')
            occupation = agent.get('occupation', '')
            if 'dorm' in cluster_name.lower() or occupation == 'manual_worker':
                base_prob *= 2.5
            
            # ⭐ QUARANTINE EFFECT (80% reduction)
            if agent.get('quarantine_status', False):
                base_prob *= 0.20
            
            # Vaccination efficacy
            vacc_efficacy = [1.0, 0.65, 0.35][agent['vaccination_status']]  # Adjusted
            base_prob *= vacc_efficacy
            
            # Age susceptibility (KEY: children less susceptible, elderly more)
            age = agent['age']
            if age < 18:
                base_prob *= 0.6
            elif age > 65:
                base_prob *= 1.4
            
            # Behavioral compliance
            compliance = agent.get('compliance_score', 0.7)
            base_prob *= (2.2 - 1.5 * compliance)  # 0.7-2.2x multiplier
            
            # Cap at reasonable max (30% per day)
            prob_s_to_e = min(base_prob, 0.30)
            return 'E' if np.random.random() < prob_s_to_e else 'S'
        
        elif current_state == 'E':
            # ═══════════════════════════════════════════════════════════════
            # E→I: PROGRESSION (unchanged from generic)
            # ═══════════════════════════════════════════════════════════════
            if days_in_state < 2:
                return 'E'
            
            if days_in_state == 2:
                base_prob = 0.05
            elif days_in_state == 3:
                base_prob = 0.15
            elif days_in_state == 4:
                base_prob = 0.25
            else:
                base_prob = 0.35
            
            vacc_effect = [1.0, 0.85, 0.70][agent['vaccination_status']]
            base_prob *= vacc_effect
            
            age = agent['age']
            if age < 18:
                base_prob *= 0.9
            elif age > 60:
                base_prob *= 1.3
            
            prob_e_to_i = min(base_prob, 0.50)
            return 'I' if np.random.random() < prob_e_to_i else 'E'
        
        elif current_state == 'I':
            # ═══════════════════════════════════════════════════════════════
            # I→R or I→D (SINGAPORE HEALTHCARE QUALITY)
            # ═══════════════════════════════════════════════════════════════
            if days_in_state < 7:
                return 'I'  # Too early for recovery/death
            
            # RECOVERY
            if days_in_state == 7:
                base_prob_recovery = 0.05
            elif days_in_state < 10:
                base_prob_recovery = 0.05 + 0.02 * (days_in_state - 7)
            elif days_in_state < 14:
                base_prob_recovery = 0.10 + 0.025 * (days_in_state - 10)
            else:
                base_prob_recovery = 0.20 + 0.02 * min(days_in_state - 14, 5)
            
            # ⭐ SINGAPORE: Faster recovery (excellent healthcare)
            base_prob_recovery *= 1.3
            
            vacc_boost = [1.0, 1.2, 1.4][agent['vaccination_status']]
            base_prob_recovery *= vacc_boost
            
            age = agent['age']
            if age < 18:
                base_prob_recovery *= 1.3
            elif age > 60:
                base_prob_recovery *= 0.7
            
            prob_i_to_r = min(base_prob_recovery, 0.40)
            
            # DEATH
            # ⭐ SINGAPORE IFR (age-stratified, 0.3x global)
            if age < 18:
                ifr = 0.00002 * 0.3  # Singapore: virtually zero
            elif age < 50:
                ifr = 0.0005 * 0.3  # Singapore: 0.02%
            elif age < 70:
                ifr = 0.005 * 0.3   # Singapore: 0.2%
            else:
                ifr = 0.02 * 0.3    # Singapore: 2% (vs 7% global)
            
            daily_death_hazard = 1.0 - (1.0 - ifr) ** (1.0 / 10.0)
            
            # Days infected modifier
            if 7 <= days_in_state <= 14:
                daily_death_hazard *= 1.5  # Peak risk (lower than global 2.0)
            elif days_in_state < 7:
                daily_death_hazard *= 0.05
            else:
                daily_death_hazard *= 0.2
            
            # Comorbidities
            daily_death_hazard *= (1.0 + agent['comorbidity_count'] * 0.5)
            
            # Vaccination protection
            vacc_protection = [1.0, 0.5, 0.2][agent['vaccination_status']]
            daily_death_hazard *= vacc_protection
            
            prob_i_to_d = min(daily_death_hazard, 0.02)
            
            # Competing risks (no normalization - use raw hazards)
            rand = np.random.random()
            if rand < prob_i_to_d:
                return 'D'
            elif rand < prob_i_to_d + prob_i_to_r:
                return 'R'
            else:
                return 'I'
        
        elif current_state == 'R':
            # ═══════════════════════════════════════════════════════════════
            # R→S: REINFECTION (unchanged from generic)
            # ═══════════════════════════════════════════════════════════════
            if days_in_state < 180:
                return 'R'  # Immunity still strong
            
            base_prob = 0.001
            if days_in_state > 365:
                base_prob = 0.005
            
            # Exposure
            neighbors = self.adj_list.get(agent_id, [])
            infected_neighbors = sum(1 for n in neighbors if all_agent_states[n]['states'][-1] in ['E', 'I'])
            if infected_neighbors > 0:
                base_prob *= (1 + infected_neighbors)
            
            # Vaccination
            vacc_effect = [1.0, 0.5, 0.2][agent['vaccination_status']]
            base_prob *= vacc_effect
            
            prob_r_to_s = min(base_prob, 0.03)
            return 'S' if np.random.random() < prob_r_to_s else 'R'
        
        else:  # D
            return 'D'
    
    def save_training_data(
        self,
        tabular: np.ndarray,
        temporal: np.ndarray,
        graph: np.ndarray,
        labels: np.ndarray,
        output_dir: Path
    ):
        """Save multimodal cluster data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'cluster_tabular.npy', tabular)
        np.save(output_dir / 'cluster_temporal.npy', temporal)
        np.save(output_dir / 'cluster_graph.npy', graph)
        np.save(output_dir / 'cluster_labels.npy', labels)
        
        # REPLACEMENT 6: Update metadata in save_training_data to reflect new dimensions:

        metadata = {
            'task': 'singapore_cluster_multimodal_multi_output_regression',
            'data_source': 'Singapore COVID-19 (MOH/Kaggle)',
            'num_samples': len(tabular),
            'tabular_dim': 37,  # UPDATED: 33 + 4 neural-specific features
            'temporal_shape': list(temporal.shape[1:]),
            'graph_dim': graph.shape[1],
            'num_outputs': 5,
            'output_names': ['S->E', 'E->I', 'I->R', 'I->D', 'R->S'],
            'initial_state_source': 'Kaggle day_0 distribution',
            'singapore_features': {
                'imported_case_rate': True,
                'quarantine_rate': True,
                'dormitory_indicator': True,
                'healthcare_factor': 0.3,
                'household_sar': 0.183
            },
            'neural_specialization': {
                'network_heterogeneity': 'household vs community mixing',
                'age_assortativity': 'intergenerational contact patterns',
                'behavioral_variance': 'compliance heterogeneity within clusters',
                'sar_potential': 'micro-level transmission risk'
            },
            'complementarity_with_llm': {
                'llm_strength': 'symbolic reasoning about interventions, disease biology',
                'neural_strength': 'fine-grained network dynamics, behavioral heterogeneity',
                'fusion_benefit': 'LLM macro policy + Neural micro transmission = accurate predictions'
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved to {output_dir}")


# Backward compatibility
ClusterMultimodalDataGenerator = SingaporeClusterMultimodalDataGenerator
