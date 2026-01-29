"""
Complete ClusterTeam with symbolic + neural pathways and epistemic fusion.

UPDATED FOR PARALLEL SEIRD PREDICTION:
- 5 states: S, E, I, R, D
- All transitions predicted SIMULTANEOUSLY: S→E, E→I, I→R, I→D, R→S
- No gating - meta_agent is optional/legacy
- Direct state_agent calls for efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import os
from pathlib import Path
import json

from src.agents.entity_agent import EntityAgent
from src.agents.state_agent import StateAgent
from src.agents.meta_agent import MetaAgent
from src.agents.epidemic_tools import EpidemicToolRegistry



# Suppress ONNX Runtime thread affinity warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Suppress ONNX Runtime logging
import logging
logging.getLogger('onnxruntime').setLevel(logging.ERROR)


# FIND the MultimodalEncoder class and REPLACE with:

class MultimodalEncoder(nn.Module):
    """
    Cluster-level multimodal encoder for transition RATE prediction.
    
    Architecture:
    - Tabular encoder: MLP (cluster features [37])
    - Temporal encoder: LSTM (cluster time series [8, 10])
    - Graph encoder: MLP (cluster embeddings [128])
    - Fusion: Concatenate + MLP
    - Output: 5 transition RATES [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
    """
    
    def __init__(
        self,
        tabular_dim: int = 37,  # ⭐ FIXED: 37 not 33
        temporal_dim: int = 10,
        graph_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 32
    ):
        super().__init__()
        
        # === Tabular Encoder ===
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # === Temporal Encoder (LSTM) ===
        self.temporal_encoder = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # === Graph Encoder ===
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU()
        )
        
        # === Fusion Layer ===
        fusion_input_dim = hidden_dim * 3
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU()
        )
        
        # === Output: 5 transition rates ===
        self.output_layer = nn.Linear(output_dim, 5)
        
        print(f"✓ MultimodalEncoder initialized (CLUSTER-LEVEL)")
        print(f"  Tabular: {tabular_dim} → {hidden_dim}")
        print(f"  Temporal: {temporal_dim} → {hidden_dim} (LSTM)")
        print(f"  Graph: {graph_dim} → {hidden_dim}")
        print(f"  Output: 5 transition rates (regression)")
    
    def forward(
        self,
        tabular: torch.Tensor,
        temporal: torch.Tensor,
        graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tabular: [batch, 37] cluster features
            temporal: [batch, 8, 10] cluster time series
            graph: [batch, 128] cluster graph embeddings
            
        Returns:
            rates: [batch, 5] - [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
        """
        # Encode each modality
        h_tabular = self.tabular_encoder(tabular)
        
        _, (h_temporal, _) = self.temporal_encoder(temporal)
        h_temporal = h_temporal[-1]
        
        h_graph = self.graph_encoder(graph)
        
        # Fuse
        h_fused = torch.cat([h_tabular, h_temporal, h_graph], dim=1)
        h_fused = self.fusion(h_fused)
        
        # Output (sigmoid constrains to [0, 1])
        logits = self.output_layer(h_fused)
        rates = torch.sigmoid(logits)
        
        return rates


class ClusterTeam:
    """
    Complete cluster team with Meta-Agent, State Agents, and dual pathways.
    
    UPDATED FOR PARALLEL SEIRD PREDICTION:
    - Manages 5 State Agents (S, E, I, R, D)
    - Predicts all transitions SIMULTANEOUSLY (no gating)
    - Meta-Agent is optional/legacy (not used in main flow)
    - Combines symbolic + neural reasoning
    """
    
    def __init__(
        self,
        cluster_id: int,
        entity_agents: List[EntityAgent],
        tool_registry: EpidemicToolRegistry,
        graphsage_embeddings: np.ndarray,
        llm_client: Optional[OpenAI] = None,
        neural_model_path: Optional[Path] = None,
        transmission_boost: float = 2.5,
        phase_tracker: Optional = None
    ):
        """
        Initialize cluster team.
        
        Args:
            cluster_id: Cluster identifier
            entity_agents: List of EntityAgent objects in this cluster
            tool_registry: Tool registry
            graphsage_embeddings: GraphSAGE embeddings for all agents
            llm_client: OpenAI client
            neural_model_path: Path to trained neural model,
            transmission_boost: Multiplier for transmission rates (default 1.0, use 2.0-3.0 to prevent stalling)
        """
        self.cluster_id = cluster_id
        self.entity_agents = entity_agents
        self.tools = tool_registry
        self.graphsage_embeddings = graphsage_embeddings
        self.llm_client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        self.transmission_boost = transmission_boost
        self.phase_tracker = phase_tracker
        
        # Create Meta-Agent (optional/legacy - not used in main flow)
        self.meta_agent = MetaAgent(
            cluster_id=cluster_id,
            cluster_agents=entity_agents,
            tool_registry=tool_registry,
            llm_client=self.llm_client
        )

        # ⭐ Store transmission boost
        self.transmission_boost = transmission_boost
        print(f"  ⚠ Transmission boost: {transmission_boost}x")
        
        # Create State Agents (SEIRD - 5 agents)
        self.state_agents = {
            'S': StateAgent(
                cluster_id=cluster_id,
                state='S',
                cluster_agents=entity_agents,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                phase_tracker=self.phase_tracker
                
            ),
            'E': StateAgent(
                cluster_id=cluster_id,
                state='E',
                cluster_agents=entity_agents,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                phase_tracker=self.phase_tracker
            ),
            'I': StateAgent(
                cluster_id=cluster_id,
                state='I',
                cluster_agents=entity_agents,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                phase_tracker=self.phase_tracker
            ),
            'R': StateAgent(
                cluster_id=cluster_id,
                state='R',
                cluster_agents=entity_agents,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                phase_tracker=self.phase_tracker
            ),
            'D': StateAgent(
                cluster_id=cluster_id,
                state='D',
                cluster_agents=entity_agents,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                phase_tracker=self.phase_tracker
            )
        }
        
        # Link Meta-Agent to State Agents
        self.meta_agent.set_state_agents(self.state_agents)
        
        # Neural pathway - CLUSTER-LEVEL model
        self.neural_model = MultimodalEncoder(
            tabular_dim=37,   # Cluster features
            temporal_dim=10,  # Cluster time series features per timestep
            graph_dim=128,    # Cluster graph embeddings
            hidden_dim=64,
            output_dim=32
        )

        self.neural_model.eval()

        # In __init__ method, after loading neural model:
        if neural_model_path and neural_model_path.exists():
            checkpoint = torch.load(neural_model_path, map_location='cpu', weights_only=False)
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
            
            # ⭐ CRITICAL DIAGNOSTIC: Test the model with random inputs
            print(f"\n  🔍 Testing neural model...")
            with torch.no_grad():
                test_tabular = torch.randn(1, 37)
                test_temporal = torch.randn(1, 8, 10)
                test_graph = torch.randn(1, 128)
                
                print(f"    Input shapes: tab={test_tabular.shape}, temp={test_temporal.shape}, graph={test_graph.shape}")
                
                try:
                    test_output = self.neural_model(test_tabular, test_temporal, test_graph)
                    print(f"    Output shape: {test_output.shape}")
                    print(f"    Output values: {test_output.cpu().numpy()}")
                    print(f"    Output sum: {test_output.sum().item():.4f}")
                    
                    if test_output.sum().item() < 0.01:
                        print(f"    ⚠️  CRITICAL: Model outputs near-zero even with random inputs!")
                        print(f"    → Model is broken or untrained")
                    else:
                        print(f"    ✓ Model produces non-zero outputs")
                except Exception as e:
                    print(f"    ❌ ERROR: Model forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Load normalization stats
            norm_stats_path = neural_model_path.parent / 'normalization_stats.json'
            if norm_stats_path.exists():
                with open(norm_stats_path, 'r') as f:
                    self.norm_stats = json.load(f)
                print(f"  ✓ Loaded normalization statistics")
            else:
                self.norm_stats = None
                print(f"  ⚠ No normalization stats found")
        else:
            print(f"  ⚠ Using untrained neural model (run train_neural_pathway.py first)")
            self.norm_stats = None
        
        print(f"✓ Cluster Team {cluster_id} initialized: {len(entity_agents)} entity agents, "
              f"1 Meta-Agent (legacy), 5 State Agents (SEIRD), neural encoder ready")
    
    def symbolic_reasoning(
        self,
        timestep: int,
        neighbor_context: str
    ) -> Dict:
        """
        LEGACY METHOD: Symbolic pathway using meta_agent.
        
        NOTE: This method is not used in the main flow (full_reasoning_cycle).
        It's kept for compatibility but the main flow directly calls state_agents.
        
        Args:
            timestep: Current timestep
            neighbor_context: Context from neighboring clusters
            
        Returns:
            Dict with ALL transition probabilities (updated for new meta_agent API)
        """
        # Meta-Agent now returns ALL predictions
        result = self.meta_agent.coordinate_cluster_reasoning(timestep, neighbor_context)
        
        # Extract predictions for all transitions
        all_predictions = result.get('predictions', {})
        
        # Return in compatible format
        return {
            'predictions': all_predictions,
            'monitoring': result.get('monitoring', {}),
            'timestep': timestep
        }
    
    

    def neural_prediction_all_transitions(self, timestep: int) -> Dict[str, Dict]:
        """
        Neural pathway - Predict ALL 5 transition rates.
        
        SIMPLIFIED: No complex rescaling - trust the trained model.
        """
        # Aggregate cluster-level features
        tabular = self._aggregate_cluster_tabular(timestep)
        temporal = self._aggregate_cluster_temporal(lookback=7)
        graph = self._aggregate_cluster_graph()
        
        print(f"      [Features] Tabular: {tabular.shape}, Temporal: {temporal.shape}, Graph: {graph.shape}", flush=True)
        
        # Normalize using training stats
        if self.norm_stats:
            tabular_mean = np.array(self.norm_stats['tabular_mean'], dtype=np.float32)
            tabular_std = np.array(self.norm_stats['tabular_std'], dtype=np.float32)
            temporal_mean = np.array(self.norm_stats['temporal_mean'], dtype=np.float32)
            temporal_std = np.array(self.norm_stats['temporal_std'], dtype=np.float32)
            
            # Normalize (skip features with zero std)
            for i in range(len(tabular)):
                if tabular_std[i] > 1e-6:
                    tabular[i] = (tabular[i] - tabular_mean[i]) / tabular_std[i]
            
            for i in range(len(temporal_mean)):
                if temporal_std[i] > 1e-6:
                    temporal[:, i] = (temporal[:, i] - temporal_mean[i]) / temporal_std[i]
        
        # Convert to tensors
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32).unsqueeze(0)
        temporal_tensor = torch.tensor(temporal, dtype=torch.float32).unsqueeze(0)
        graph_tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0)
        
        # Handle NaN/Inf
        tabular_tensor = torch.nan_to_num(tabular_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        temporal_tensor = torch.nan_to_num(temporal_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        graph_tensor = torch.nan_to_num(graph_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Neural prediction
        with torch.no_grad():
            rates = self.neural_model(tabular_tensor, temporal_tensor, graph_tensor)
            rates_np = rates.cpu().numpy().flatten()
        
        # Transition names
        transition_names = ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']
        
        results = {}
        print(f"      [Neural] Raw model outputs:", flush=True)
        
        for i, name in enumerate(transition_names):
            raw_rate = float(rates_np[i])
            
            # Apply transmission boost to S->E only
            if name == 'S->E':
                final_rate = min(raw_rate * self.transmission_boost, 0.30)
            else:
                final_rate = raw_rate
            
            # Set confidence based on rate magnitude
            if 0.01 <= final_rate <= 0.50:
                confidence = 0.7  # Good range
            elif final_rate < 0.01:
                confidence = 0.5  # Too low
            else:
                confidence = 0.6  # High but not extreme
            
            results[name] = {
                'probability': final_rate,
                'confidence': confidence,
                'uncertainty': 1.0 - confidence,
                'raw': raw_rate
            }
            
            boost_str = f" (boosted {self.transmission_boost}x)" if name == 'S->E' else ""
            print(f"        {name}: {final_rate:.4f}{boost_str}", flush=True)
        
        return results

    def _create_tabular_features(self, agent: EntityAgent) -> np.ndarray:
        """Create tabular feature vector (82 dims) to match trained model."""
        profile = agent.profile
        features = []
        
        # === CORE FEATURES (18 dims) ===
        
        # 1. Age (normalized)
        features.append(profile['age'] / 100.0)
        
        # 2-7. Occupation (one-hot, 6 types)
        occupations = ['healthcare_worker', 'office_worker', 'retail_worker', 
                      'teacher', 'student', 'retired']
        occ_onehot = [1.0 if profile['occupation'] == o else 0.0 for o in occupations]
        features.extend(occ_onehot)
        
        # 8. Household size (normalized)
        features.append(1.0 / 10.0)

        # ⭐ SINGAPORE-SPECIFIC (optional - adds 3 dimensions)
        features.append(1.0 if profile.get('is_imported', False) else 0.0)
        features.append(1.0 if profile.get('quarantine_status', False) else 0.0)
        dorm_indicator = 1.0 if 'dorm' in str(profile.get('cluster', '')).lower() else 0.0
        features.append(dorm_indicator)
        
        # 9-11. Vaccination (one-hot, 3 levels)
        vacc_onehot = [0.0, 0.0, 0.0]
        vacc_onehot[profile['vaccination_status']] = 1.0
        features.extend(vacc_onehot)
        
        # 12. Comorbidities (normalized)
        features.append(profile['comorbidity_count'] / 5.0)
        
        # 13. Mobility
        features.append(profile['mobility_score'])
        
        # 14. Compliance
        features.append(profile['compliance_score'])
        
        # 15. Risk awareness
        features.append(profile['risk_awareness'])
        
        # 16. Days in current state (normalized)
        features.append(min(agent.days_in_state / 20.0, 1.0))
        
        # 17. Infected neighbors count (normalized)
        infected_neighbors = len([n for n in agent.neighbors if n.state in ['E', 'I']])
        features.append(infected_neighbors / max(len(agent.neighbors), 1))
        
        # 18. Is isolated (binary)
        features.append(1.0 if getattr(agent, 'is_isolated', False) else 0.0)
        
        # === TEXT EMBEDDING PLACEHOLDER (64 dims) ===
        # The trained model expected 64-dim text embeddings from clinical notes
        # Since we don't generate clinical notes during simulation, use zeros
        text_embedding_placeholder = [0.0] * 64
        features.extend(text_embedding_placeholder)
        
        assert len(features) == 82, f"Expected 82 features, got {len(features)}"
        
        return np.array(features, dtype=np.float32)

    def _encode_state_history(self, agent: EntityAgent) -> np.ndarray:
        """Encode state history (7 × 7) with SEIRD + context features."""
        state_to_idx = {'S': 0, 'E': 1, 'I': 2, 'R': 3, 'D': 4}
        
        # Get last 7 days of history
        if hasattr(agent, 'state_history') and len(agent.state_history) > 0:
            history = agent.state_history[-7:]
        else:
            history = [agent.state] * 7
        
        # Pad if needed
        while len(history) < 7:
            history.insert(0, history[0] if history else agent.state)
        
        history = history[-7:]
        
        # Encode: [7 timesteps × 7 features]
        # Features: [S, E, I, R, D, days_in_state_norm, has_infected_neighbor]
        encoded = np.zeros((7, 7), dtype=np.float32)
        
        for i, state in enumerate(history):
            # Features 0-4: SEIRD one-hot
            encoded[i, state_to_idx.get(state, 0)] = 1.0
            
            # Feature 5: Days in state (normalized, only for current timestep)
            if i == len(history) - 1:  # Most recent timestep
                encoded[i, 5] = min(agent.days_in_state / 20.0, 1.0)
            
            # Feature 6: Has infected neighbors (only for current timestep)
            if i == len(history) - 1:
                infected_neighbors = len([n for n in agent.neighbors if n.state in ['E', 'I']])
                encoded[i, 6] = min(infected_neighbors / 5.0, 1.0)
        
        return encoded  # Shape: [7, 7]
    
    def epistemic_fusion(
        self,
        symbolic_result: Dict,
        neural_result: Dict
    ) -> Dict:
        """
        Epistemic fusion - Combine symbolic + neural with uncertainty weighting.
        
        INCLUDES:
        - Inverse-variance weighting
        - Phase tracker multipliers (if available)
        - Transmission boost awareness
        """
        # Extract probabilities and confidences
        p_symbolic = symbolic_result['probability']
        conf_symbolic = symbolic_result['confidence']
        
        p_neural = neural_result['probability']
        conf_neural = neural_result['confidence']
        
        # Inverse-variance weighting
        total_conf = conf_symbolic + conf_neural
        
        if total_conf > 0:
            print("***Using Inverse-Variance Weighting for Fusion***", flush=True)
            w_symbolic = conf_symbolic / total_conf
            w_neural = conf_neural / total_conf
        else:
            w_symbolic = 0.6
            w_neural = 0.4
        
        # Fused probability
        p_fused = w_symbolic * p_symbolic + w_neural * p_neural
        
        # ⭐ Apply epidemic phase multiplier (if phase tracker available)
        if self.phase_tracker and 'transition_type' in symbolic_result:
            phase_mults = self.phase_tracker.get_phase_multipliers()
            transition_type = symbolic_result['transition_type']
            phase_mult = phase_mults.get(transition_type, 1.0)
            
            print(f"        [Fusion] Phase multiplier for {transition_type}: {phase_mult:.2f}x", flush=True)
            p_fused = p_fused * phase_mult
        
        # ⭐ Final safety cap
        p_fused = min(p_fused, 0.95)
        
        # Combined confidence (harmonic mean)
        if conf_symbolic > 0 and conf_neural > 0:
            combined_confidence = 2 * (conf_symbolic * conf_neural) / (conf_symbolic + conf_neural)
        else:
            combined_confidence = max(conf_symbolic, conf_neural)
        
        return {
            'probability': p_fused,
            'confidence': combined_confidence,
            'weights': {
                'symbolic': w_symbolic,
                'neural': w_neural
            },
            'components': {
                'symbolic': p_symbolic,
                'neural': p_neural
            },
            'reasoning': symbolic_result.get('reasoning', 'No reasoning provided')
        }
    

    def _aggregate_cluster_tabular(self, timestep: int) -> np.ndarray:
        """
        Aggregate cluster-level tabular features (37 dimensions).
        
        Features:
        - State distribution (5)
        - Demographics (11) including Singapore-specific
        - Network (7)
        - Epidemic (10)
        - Neural-specific (4): household mixing, age assortativity, behavioral heterogeneity, SAR potential
        """
        from collections import Counter
        
        features = []
        total = len(self.entity_agents)
        
        if total == 0:
            return np.zeros(37, dtype=np.float32)
        
        # State distribution (5)
        state_counts = Counter([a.state for a in self.entity_agents])
        for state in ['S', 'E', 'I', 'R', 'D']:
            features.append(state_counts.get(state, 0) / total)
        
        # Demographics (11)
        ages = [a.profile['age'] for a in self.entity_agents]
        features.append(np.mean(ages) / 100.0)
        features.append(np.std(ages) / 50.0)
        
        vacc = [a.profile['vaccination_status'] for a in self.entity_agents]
        features.append(np.mean(vacc) / 2.0)
        
        comorbid = [a.profile['comorbidity_count'] for a in self.entity_agents]
        features.append(np.mean(comorbid) / 5.0)
        
        high_risk = ['healthcare_worker', 'teacher', 'retail_worker']
        occupations = [a.profile['occupation'] for a in self.entity_agents]
        features.append(sum(1 for o in occupations if o in high_risk) / total)
        
        features.append(np.mean([a.profile.get('compliance_score', 0.7) for a in self.entity_agents]))
        features.append(np.mean([a.profile.get('mobility_score', 0.7) for a in self.entity_agents]))
        
        # Singapore-specific (3)
        male_count = sum(1 for a in self.entity_agents if a.profile.get('gender', 'unknown') == 'm')
        features.append(male_count / total)
        
        imported = sum(1 for a in self.entity_agents if a.profile.get('is_imported', False))
        features.append(imported / total)
        
        quarantined = sum(1 for a in self.entity_agents if a.profile.get('quarantine_status', False))
        features.append(quarantined / total)
        
        dorm_workers = sum(1 for a in self.entity_agents 
                        if 'dorm' in a.profile.get('cluster', '').lower() 
                        or a.profile.get('occupation') == 'manual_worker')
        features.append(1.0 if dorm_workers > total * 0.5 else 0.0)
        
        # Network (7)
        degrees = [len(a.neighbors) for a in self.entity_agents]
        features.append(np.mean(degrees) / 50.0)
        features.append(np.std(degrees) / 25.0)
        features.extend([0.3, 0.1, 0.5, 0.3, 0.4])  # Placeholders
        
        # Epidemic dynamics (10)
        infected = state_counts.get('I', 0) + state_counts.get('E', 0)
        features.append(infected / total)
        
        for state in ['S', 'E', 'I', 'R']:
            agents_in_state = [a for a in self.entity_agents if a.state == state]
            if agents_in_state:
                avg_days = np.mean([a.days_in_state for a in agents_in_state])
                max_days = {'S': 30, 'E': 10, 'I': 20, 'R': 90}[state]
                features.append(min(avg_days / max_days, 1.0))
            else:
                features.append(0.0)
        
        s_agents = [a for a in self.entity_agents if a.state == 'S']
        if s_agents:
            s_exposure = np.mean([sum(1 for n in a.neighbors if n.state in ['E', 'I']) for a in s_agents])
            features.append(s_exposure / 10.0)
        else:
            features.append(0.0)
        
        features.append(0.5)  # Trend placeholder
        features.append(0.2)  # Attack rate placeholder
        features.append(timestep / 200.0)
        features.append(0.0)  # Vaccination campaign
        
        # Neural-specific (4)
        # 1. Household mixing
        household_edges = sum(
            1 for a in self.entity_agents
            for n in a.neighbors
            if n in self.entity_agents and 
            a.profile.get('household_id') == n.profile.get('household_id')
        )
        total_edges = sum(len(a.neighbors) for a in self.entity_agents)
        features.append(household_edges / max(total_edges, 1))
        
        # 2. Age assortativity
        young = [a for a in self.entity_agents if a.profile['age'] < 18]
        old = [a for a in self.entity_agents if a.profile['age'] > 65]
        young_to_old = sum(1 for a in young for n in a.neighbors if n in old)
        total_young_contacts = sum(len(a.neighbors) for a in young)
        features.append(young_to_old / max(total_young_contacts, 1))
        
        # 3. Behavioral heterogeneity
        compliance_scores = [a.profile.get('compliance_score', 0.7) for a in self.entity_agents]
        features.append(np.std(compliance_scores))
        
        # 4. SAR potential
        infected_agents = [a for a in self.entity_agents if a.state in ['E', 'I']]
        sar_potential = sum(sum(1 for n in a.neighbors if n.state == 'S') for a in infected_agents)
        features.append(sar_potential / max(total, 1))
        
        assert len(features) == 37, f"Expected 37 features, got {len(features)}"
        
        return np.array(features, dtype=np.float32)

    def _aggregate_cluster_temporal(self, lookback: int = 7) -> np.ndarray:
        """
        Aggregate cluster-level temporal features [8, 10].
        
        Returns:
            Temporal sequence [lookback+1, 10]
        """
        from collections import Counter
        
        temporal_seq = []
        total = len(self.entity_agents)
        
        if total == 0:
            return np.zeros((8, 10), dtype=np.float32)
        
        # Build temporal sequence (last 7 + current = 8 timesteps)
        for t in range(max(0, lookback)):
            features_t = []
            
            # State proportions (5)
            # For historical timesteps, we'd need stored history
            # For now, use current state as approximation
            state_counts = Counter([a.state for a in self.entity_agents])
            for state in ['S', 'E', 'I', 'R', 'D']:
                features_t.append(state_counts.get(state, 0) / total)
            
            # Incidence (2) - placeholder
            features_t.extend([0.01, 0.01])
            
            # Prevalence (1)
            active = (state_counts.get('E', 0) + state_counts.get('I', 0)) / total
            features_t.append(active)
            
            # Contact rate (1) - placeholder
            features_t.append(0.4)
            
            # Policy stringency (1) - placeholder
            features_t.append(0.5)
            
            temporal_seq.append(features_t)
        
        # Pad if needed
        while len(temporal_seq) < 8:
            temporal_seq.insert(0, np.zeros(10, dtype=np.float32))
        
        return np.array(temporal_seq[:8], dtype=np.float32)

    def _aggregate_cluster_graph(self) -> np.ndarray:
        """
        Aggregate cluster-level graph embedding [128].
        
        Returns:
            Cluster graph embedding
        """
        if len(self.entity_agents) == 0:
            return np.zeros(128, dtype=np.float32)
        
        # Get embeddings for all agents in cluster
        agent_indices = [a.agent_id for a in self.entity_agents if a.agent_id < len(self.graphsage_embeddings)]
        
        if not agent_indices:
            return np.zeros(128, dtype=np.float32)
        
        cluster_embeddings = self.graphsage_embeddings[agent_indices]
        
        # Mean pooling
        cluster_embedding = cluster_embeddings.mean(axis=0)
        
        return cluster_embedding.astype(np.float32)


    def apply_community_transmission(self, timestep: int, base_prob: float = 0.15) -> int:
        """
        Apply community transmission events to prevent epidemic stalling.
        
        Called periodically (e.g., every 7 days) to simulate super-spreader events,
        community gatherings, or other sources of transmission outside the network.
        
        Args:
            timestep: Current timestep
            base_prob: Base probability of exposure per susceptible agent
            
        Returns:
            Number of agents exposed
        """
        # Only trigger every 7 days
        if timestep % 7 != 0:
            return 0
        
        susceptible = [a for a in self.entity_agents if a.state == 'S' and not a.is_dead]
        
        if len(susceptible) == 0:
            return 0
        
        # Expose 5-10% of susceptible agents
        n_to_expose = max(1, int(np.random.uniform(0.05, 0.10) * len(susceptible)))
        exposed_agents = np.random.choice(susceptible, size=min(n_to_expose, len(susceptible)), replace=False)
        
        actual_exposed = 0
        for agent in exposed_agents:
            if np.random.random() < base_prob:
                agent.state = 'E'
                agent.days_in_state = 0
                actual_exposed += 1
        
        if actual_exposed > 0:
            print(f"\n    🦠 [Community Event] Cluster {self.cluster_id}: {actual_exposed}/{n_to_expose} agents exposed", flush=True)
        
        return actual_exposed


    def full_reasoning_cycle(
        self,
        timestep: int,
        neighbor_context: str
    ) -> Dict:
        """
        Complete reasoning cycle with ALL SEIRD state transitions.
        
        UPDATED: Now includes I→D (death) transition.
        Uses parallel prediction (no gating).
        """
        import sys
        import time
        
        cycle_start_time = time.time()
        
        print(f"\n  Cluster {self.cluster_id} reasoning cycle:", flush=True)
       
        
        # Count active agents
        active_agents = [a for a in self.entity_agents if not a.is_dead]
        
        
        if not active_agents:
            return {
                'cluster_probabilities': {},
                'agent_results': {},
                'transitions': [],
                'deaths': []
            }

        # ═══════════════════════════════════════════════════════════════
        # STEP 0: Community transmission (periodic super-spreader events)
        # ═══════════════════════════════════════════════════════════════
        community_exposed = self.apply_community_transmission(timestep, base_prob=0.15)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Compute cluster-level probabilities for ALL transitions
        # ═══════════════════════════════════════════════════════════════
        step_start = time.time()
        
        cluster_probabilities = self._compute_all_cluster_probabilities(timestep, neighbor_context)
        
        step_elapsed = time.time() - step_start
        
        print(f"    [Cluster Probabilities]", flush=True)
        print(f"      S→E: {cluster_probabilities['S->E']['fused']:.3f} (sym={cluster_probabilities['S->E']['symbolic']:.3f}, neu={cluster_probabilities['S->E']['neural']:.3f})", flush=True)
        print(f"      E→I: {cluster_probabilities['E->I']['fused']:.3f} (sym={cluster_probabilities['E->I']['symbolic']:.3f}, neu={cluster_probabilities['E->I']['neural']:.3f})", flush=True)
        print(f"      I→R: {cluster_probabilities['I->R']['fused']:.3f} (sym={cluster_probabilities['I->R']['symbolic']:.3f}, neu={cluster_probabilities['I->R']['neural']:.3f})", flush=True)
        print(f"      I→D: {cluster_probabilities['I->D']['fused']:.3f} (sym={cluster_probabilities['I->D']['symbolic']:.3f}, neu={cluster_probabilities['I->D']['neural']:.3f})", flush=True)
        print(f"      R→S: {cluster_probabilities['R->S']['fused']:.3f} (sym={cluster_probabilities['R->S']['symbolic']:.3f}, neu={cluster_probabilities['R->S']['neural']:.3f})", flush=True)
        
        # Count agents by state (SEIRD)
        state_counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.entity_agents:
            state_counts[agent.state] += 1
        
        print(f"    Agent distribution: S={state_counts['S']}, E={state_counts['E']}, I={state_counts['I']}, R={state_counts['R']}, D={state_counts['D']}", flush=True)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Each agent reasons with full cluster context
        # ═══════════════════════════════════════════════════════════════
        print(f"    [Individual Agents - Processing {len(self.entity_agents)} agents]", flush=True)
        
        all_agent_results = {}
        transitions = []
        deaths = []
        
        for idx, agent in enumerate(self.entity_agents, 1):
            agent_start_time = time.time()
            
            try:
                agent_result = agent.reason_and_sample_transition(
                    cluster_probabilities=cluster_probabilities,
                    neighbor_context=neighbor_context,
                    timestep=timestep
                )
                
                agent_elapsed = time.time() - agent_start_time
                
                
            except Exception as e:
                
                import traceback
                traceback.print_exc()
                # Create fallback result
                agent_result = {
                    'old_state': agent.state,
                    'new_state': agent.state,
                    'transitioned': False,
                    'probability': 0.0,
                    'reasoning': f'Error: {e}',
                    'sampled_value': 0.0,
                    'cluster_prob': 0.0
                }
            
            all_agent_results[agent.agent_id] = agent_result
            
            # Print agent's decision
            print(f"          Decision:", flush=True)
            print(f"            Cluster prob: {agent_result.get('cluster_prob', 0.0):.3f}", flush=True)
            print(f"            Personal prob: {agent_result['probability']:.3f}", flush=True)
            print(f"            Sampled: {agent_result.get('sampled_value', 0.0):.3f}", flush=True)
            
            if agent_result['transitioned']:
                transition_info = {
                    'agent_id': agent.agent_id,
                    'agent_name': agent.name,
                    'from_state': agent_result['old_state'],
                    'to_state': agent_result['new_state'],
                    'probability': agent_result['probability'],
                    'reasoning': agent_result['reasoning']
                }
                
                if agent_result['new_state'] == 'D':
                    print(f"            ✗ DEATH: {agent_result['old_state']}→{agent_result['new_state']}", flush=True)
                    deaths.append(transition_info)
                else:
                    print(f"            ✓ TRANSITION: {agent_result['old_state']}→{agent_result['new_state']}", flush=True)
                
                transitions.append(transition_info)
            else:
                print(f"            → No transition (stayed {agent.state})", flush=True)
            
            # Truncate reasoning for display
            reasoning_display = agent_result['reasoning'][:120] if len(agent_result['reasoning']) > 120 else agent_result['reasoning']
            print(f"          Reasoning: {reasoning_display}...", flush=True)
        
       
        
        # Print transitions summary
        print(f"\n    [Summary]", flush=True)
        if deaths:
            print(f"      {len(deaths)} death(s) this timestep:", flush=True)
            for d in deaths:
                print(f"        ✗ {d['agent_name']}: {d['from_state']}→D (p={d['probability']:.3f})", flush=True)
        
        if transitions:
            non_death_transitions = [t for t in transitions if t['to_state'] != 'D']
            if non_death_transitions:
                print(f"      {len(non_death_transitions)} other transition(s):", flush=True)
                for t in non_death_transitions:
                    print(f"        ✓ {t['agent_name']}: {t['from_state']}→{t['to_state']} (p={t['probability']:.3f})", flush=True)
        
        if not transitions:
            print(f"      No state transitions this timestep", flush=True)
        
        cycle_elapsed = time.time() - cycle_start_time
        
        
        return {
            'cluster_probabilities': cluster_probabilities,
            'agent_results': all_agent_results,
            'transitions': transitions,
            'deaths': deaths
        }

    def _compute_all_cluster_probabilities(
        self,
        timestep: int,
        neighbor_context: str
    ) -> Dict:
        """
        Compute cluster-level probabilities for ALL SEIRD transitions.
        
        UPDATED: Calls neural model ONCE to get all 5 probabilities.
        INCLUDES: I→R suppression for early infections.
        
        Returns:
            Dict with all transition probabilities after fusion
        """
        results = {}
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Neural prediction (ALL transitions at once)
        # ═══════════════════════════════════════════════════════════════
        neural_results = self.neural_prediction_all_transitions(timestep)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Symbolic + Fusion for each transition
        # ═══════════════════════════════════════════════════════════════
        transitions = [
            ('S', 'S->E'),
            ('E', 'E->I'),
            ('I', 'I->R'),
            ('I', 'I->D'),
            ('R', 'R->S')
        ]
        
        for state, transition_name in transitions:
            # Check if any agents in this state
            agents_in_state = [a for a in self.entity_agents if a.state == state]
            
            if len(agents_in_state) == 0:
                results[transition_name] = {
                    'symbolic': 0.0,
                    'neural': 0.0,
                    'fused': 0.0,
                    'confidence': 1.0
                }
                continue
            
            # ⭐⭐⭐ CRITICAL: SPECIAL HANDLING FOR I→R ⭐⭐⭐
            if transition_name == 'I->R':
                # Calculate average days infected
                avg_days_infected = np.mean([a.days_in_state for a in agents_in_state])
                
                print(f"      I→R: avg_days={avg_days_infected:.1f}, n={len(agents_in_state)}", flush=True)
                
                # ⭐ SUPPRESS RECOVERY FOR EARLY INFECTIONS
                if avg_days_infected < 7.0:
                    # TOO EARLY - force very low probability
                    symbolic_result = {
                        'probability': 0.00,  # Max 2% before day 7
                        'confidence': 0.9,
                        'reasoning': f'Early infection (avg {avg_days_infected:.1f} days) - recovery unlikely',
                        'transition_type': transition_name
                    }
                    print(f"      [Symbolic] I->R: 0.020 (SUPPRESSED - too early)", flush=True)
                
                elif avg_days_infected < 10.0:
                    # MODERATE SUPPRESSION (days 7-10)
                    symbolic_result = self.state_agents[state].predict_transition_probability(
                        timestep=timestep,
                        neighbor_cluster_context=neighbor_context,
                        transition_type=transition_name
                    )
                    # Cap at 8%
                    symbolic_result['probability'] = min(symbolic_result['probability'], 0.08)
                    print(f"      [Symbolic] I->R: {symbolic_result['probability']:.3f} (CAPPED - early recovery phase)", flush=True)
                
                else:
                    # NORMAL (day 10+) - allow state agent's full prediction
                    symbolic_result = self.state_agents[state].predict_transition_probability(
                        timestep=timestep,
                        neighbor_cluster_context=neighbor_context,
                        transition_type=transition_name
                    )
                    print(f"      [Symbolic] I->R: {symbolic_result['probability']:.3f}", flush=True)
            
            else:
                # ═══════════════════════════════════════════════════════
                # NORMAL SYMBOLIC PREDICTION (S→E, E→I, I→D, R→S)
                # ═══════════════════════════════════════════════════════
                try:
                    symbolic_result = self.state_agents[state].predict_transition_probability(
                        timestep=timestep,
                        neighbor_cluster_context=neighbor_context,
                        transition_type=transition_name
                    )
                    print(f"      [Symbolic] {transition_name}: {symbolic_result['probability']:.3f}", flush=True)
                except Exception as e:
                    print(f"      [ERROR] Symbolic reasoning failed for {transition_name}: {e}", flush=True)
                    symbolic_result = {'probability': 0.05, 'confidence': 0.5, 'reasoning': 'Error in symbolic'}
            
            # Get neural result (already computed)
            neural_result = neural_results[transition_name]
            
            # Epistemic fusion
            try:
                symbolic_result['transition_type'] = transition_name  # Add for phase multiplier
                fused_result = self.epistemic_fusion(symbolic_result, neural_result)
            except Exception as e:
                print(f"      [ERROR] Fusion failed for {transition_name}: {e}", flush=True)
                fused_result = {'probability': symbolic_result['probability'], 'confidence': 0.5}
            
            results[transition_name] = {
                'symbolic': symbolic_result['probability'],
                'neural': neural_result['probability'],
                'fused': fused_result['probability'],
                'confidence': fused_result['confidence'],
                'weights': fused_result.get('weights', {'symbolic': 0.6, 'neural': 0.4})
            }
        
        # Apply safety caps (existing code - keep this)
        MAX_CAPS = {
            'S->E': 0.10,
            'E->I': 0.25,
            'I->R': 0.35,
            'I->D': 0.02,
            'R->S': 0.03
        }
        
        for transition_name, cap in MAX_CAPS.items():
            if transition_name in results:
                if results[transition_name]['fused'] > cap:
                    print(f"      ⚠️ Capping {transition_name}: {results[transition_name]['fused']:.3f} → {cap:.3f}", flush=True)
                    results[transition_name]['fused'] = cap
        
        return results
