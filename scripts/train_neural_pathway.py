"""
Train cluster-level multimodal neural pathway for Singapore COVID-19.
UPDATED FOR SINGAPORE DATA.

ARCHITECTURE:
- Tabular encoder: MLP (cluster features, 37-dim for Singapore + neural-specific)
- Temporal encoder: LSTM (cluster time series)
- Graph encoder: MLP (cluster graph embeddings)
- Fusion: Concatenate + MLP
- Output: 5 transition RATES (continuous [0,1])

DATA SOURCE:
- Singapore COVID-19 case line list (Kaggle)
- 1,000 agents extracted from 3,252 real cases
- Contact network from TraceTogether/SafeEntry
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from src.neural.cluster_data_generator import SingaporeClusterMultimodalDataGenerator
from src.neural.trainer import MultimodalTrainer, create_balanced_sampler
from torch.utils.data import DataLoader



# ============================================================================
# MULTIMODAL ENCODER: Full Architecture
# ============================================================================

class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder for cluster-level transition RATE prediction.
    
    Architecture:
    - Tabular encoder: MLP for cluster features [37] → [64] (SINGAPORE + NEURAL-SPECIFIC)
    - Temporal encoder: LSTM for time series [8, 10] → [64]
    - Graph encoder: MLP for embeddings [128] → [64]
    - Fusion: Concatenate [192] → [64] → [5]
    
    Tabular features (37):
    - Base (33): Demographics, network, epidemic, Singapore-specific
    - Neural-specific (4): household mixing, age assortativity, behavioral heterogeneity, SAR potential
    
    Output: Transition RATES [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
    """
    
    def __init__(
        self,
        tabular_dim: int,
        temporal_dim: int,
        graph_dim: int,
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
        
        # === Output Layer: 5 transition rates ===
        self.output_layer = nn.Linear(output_dim, 5)
        
        print(f"✓ MultimodalEncoder initialized")
        print(f"  Tabular: {tabular_dim} → {hidden_dim}")
        print(f"  Temporal: {temporal_dim} → {hidden_dim} (LSTM)")
        print(f"  Graph: {graph_dim} → {hidden_dim}")
        print(f"  Fusion: {fusion_input_dim} → {output_dim} → 5")
    
    def forward(
        self,
        tabular: torch.Tensor,
        temporal: torch.Tensor,
        graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tabular: [batch, 37] cluster features (SINGAPORE + NEURAL-SPECIFIC)
            temporal: [batch, 8, 10] cluster time series
            graph: [batch, 128] cluster graph embeddings
            
        Returns:
            rates: [batch, 5] transition rates (continuous [0,1])
                   [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
        """
        # Encode tabular
        h_tabular = self.tabular_encoder(tabular)
        
        # Encode temporal (LSTM)
        _, (h_temporal, _) = self.temporal_encoder(temporal)
        h_temporal = h_temporal[-1]  # Take last layer hidden state
        
        # Encode graph
        h_graph = self.graph_encoder(graph)
        
        # Fuse modalities
        h_fused = torch.cat([h_tabular, h_temporal, h_graph], dim=1)
        h_fused = self.fusion(h_fused)
        
        # Output transition rates (sigmoid constrains to [0, 1])
        logits = self.output_layer(h_fused)
        rates = torch.sigmoid(logits)
        
        return rates


def main():
    print("\n" + "="*80)
    print("SINGAPORE COVID-19 CLUSTER-LEVEL MULTIMODAL NEURAL PATHWAY TRAINING")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: LOAD SINGAPORE DATA
    # ========================================================================
    print("STEP 1: Loading Singapore COVID-19 data")
    print("-" * 80)
    
    processed_dir = Path("data/processed")
    
    # Load agent profiles (Singapore COVID-19)
    with open(processed_dir / "agent_profiles.json", 'r') as f:
        profiles = json.load(f)
    
    # Load contact network
    with open(processed_dir / "contact_network.json", 'r') as f:
        edges = json.load(f)
    
    # Load cluster assignments
    cluster_file = processed_dir / "cluster_assignments_hsbc3.json"
    if cluster_file.exists():
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        if isinstance(cluster_data, dict):
            first_value = next(iter(cluster_data.values()))
            
            if isinstance(first_value, list):
                cluster_assignments = {int(k): v[0] if v else 0 for k, v in cluster_data.items()}
            else:
                cluster_assignments = {int(k): int(v) for k, v in cluster_data.items()}
        else:
            print("  ⚠ Warning: Using default cluster assignments")
            cluster_assignments = {i: 0 for i in range(len(profiles))}
    else:
        print("  ⚠ Warning: No cluster assignments found, using default")
        cluster_assignments = {i: 0 for i in range(len(profiles))}
    
    # Load GraphSAGE embeddings
    embeddings = np.load(processed_dir / "graphsage/graphsage_embeddings.npy")
    
    print(f"✓ Loaded Singapore data:")
    print(f"  Profiles: {len(profiles)} agents")
    print(f"  Edges: {len(edges)} connections")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Clusters: {len(set(cluster_assignments.values()))} clusters")
    
    # Check for Singapore-specific fields
    sample_profile = profiles[0]
    has_singapore_fields = all(k in sample_profile for k in ['cluster', 'is_imported', 'nationality'])
    print(f"  Singapore fields detected: {has_singapore_fields}")
    
    # Validate Kaggle data availability
    kaggle_train_path = Path("data/singapore/train.csv")
    if kaggle_train_path.exists():
        print(f"\n✓ Kaggle Singapore data found: {kaggle_train_path}")
        import pandas as pd
        df = pd.read_csv(kaggle_train_path)
        day_0_counts = df['day_0'].value_counts()
        print(f"  Kaggle dataset size: {len(df)} rows")
        print(f"  Initial state distribution (day_0):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = day_0_counts.get(state, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    {state}: {count} ({pct:.1f}%)")
    else:
        print(f"\n⚠️  WARNING: Kaggle data not found at {kaggle_train_path}")
        print(f"  Data generator will use fallback distribution")
    
    # ========================================================================
    # STEP 2: GENERATE MULTIMODAL CLUSTER DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: GENERATE SINGAPORE MULTIMODAL CLUSTER DATA")
    print("="*80)
    
    generator = SingaporeClusterMultimodalDataGenerator(
        profiles=profiles,
        edges=edges,
        cluster_assignments=cluster_assignments,
        graphsage_embeddings=embeddings
    )
    
    # Generate multimodal data
    tabular, temporal, graph, labels = generator.generate_multimodal_cluster_data(
        num_simulations=100,
        timesteps_per_sim=200
    )
    
    # Save training data
    neural_data_dir = processed_dir / "neural_training_data"
    generator.save_training_data(tabular, temporal, graph, labels, neural_data_dir)
    
    print(f"\n✓ Generated Singapore cluster data:")
    print(f"  Tabular features: {tabular.shape[1]} dimensions")
    print(f"  Expected: 37 (33 Singapore base + 4 neural-specific)")
    print(f"    - Base (33): Demographics, network, epidemic, Singapore-specific")
    print(f"    - Neural-specific (4): household mixing, age assortativity, behavioral variance, SAR potential")
    
    if tabular.shape[1] != 37:
        print(f"  ⚠️  WARNING: Expected 37 features but got {tabular.shape[1]}!")
    
    # DIAGNOSTIC: Check target distributions
    print(f"\n📊 TARGET TRANSITION RATE ANALYSIS:")
    print(f"  {'Transition':<10} {'Mean':<12} {'Std':<12} {'NonZero%':<12} {'Max':<12}")
    print(f"  {'-'*60}")
    all_zero = True
    for i, name in enumerate(['S->E', 'E->I', 'I->R', 'I->D', 'R->S']):
        trans_labels = labels[:, i]
        nonzero_pct = (trans_labels > 0.001).sum() / len(trans_labels) * 100
        print(f"  {name:<10} {trans_labels.mean():<12.6f} {trans_labels.std():<12.6f} "
              f"{nonzero_pct:<12.1f} {trans_labels.max():<12.6f}")
        if trans_labels.sum() > 0.001:
            all_zero = False
    
    if all_zero:
        print(f"\n  ❌ CRITICAL: ALL TRANSITION RATES ARE ZERO!")
        print(f"  Your simulation is not generating state transitions.")
        print(f"  Check _run_singapore_seird_simulation() and _sample_singapore_next_state()")
        sys.exit(1)
    
    if labels.mean() < 0.001:
        print(f"\n  ⚠️  WARNING: Transition rates very sparse (mean={labels.mean():.6f})")
        print(f"  Consider: more initial infections, longer simulations, or higher transmission rates")
    
    # ========================================================================
    # STEP 3: CREATE DATASET
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CREATE DATASET")
    print("="*80)
    
    from src.neural.dataset import TransitionDataset, create_dataloaders
    
    dataset = TransitionDataset(
        tabular,
        temporal,
        graph,
        labels,
        normalize=True
    )
    
    from torch.utils.data import random_split
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create balanced sampler for training (oversample transitions)
    train_labels = labels[train_dataset.indices]
    balanced_sampler = create_balanced_sampler(train_labels, oversample_factor=3.0)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=balanced_sampler,  # Use balanced sampler
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n✓ Created dataloaders:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # ========================================================================
    # STEP 4: CREATE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: CREATE MULTIMODAL MODEL")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = MultimodalEncoder(
        tabular_dim=tabular.shape[1],  # Should be 37
        temporal_dim=temporal.shape[2],  # 10
        graph_dim=graph.shape[1],  # 128
        hidden_dim=64,
        output_dim=32
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel statistics:")
    print(f"  Total parameters: {num_params:,}")
    
    # ========================================================================
    # STEP 5: TRAIN MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: TRAIN MODEL")
    print("="*80)
    
    # Emphasize S->E (outbreak prediction) and I->D (mortality)
    transition_weights = torch.tensor([5.0, 2.0, 1.5, 3.0, 1.0])
    
    trainer = MultimodalTrainer(
        model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        transition_weights=transition_weights
    )
    
    model_dir = processed_dir / "neural_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=300,
        early_stopping_patience=15,
        save_dir=model_dir
    )
    
    # ========================================================================
    # STEP 6: SAVE ARTIFACTS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVE ARTIFACTS")
    print("="*80)
    
    model_config = {
        'task': 'singapore_cluster_multimodal_transition_rate_regression',
        'data_source': 'Singapore COVID-19 (MOH/Kaggle)',
        'tabular_dim': tabular.shape[1],
        'temporal_shape': list(temporal.shape[1:]),
        'graph_dim': graph.shape[1],
        'hidden_dim': 64,
        'output_dim': 32,
        'num_outputs': 5,
        'output_names': ['S->E', 'E->I', 'I->R', 'I->D', 'R->S'],
        'output_type': 'transition_rates',
        'num_parameters': num_params,
        'singapore_features': {
            'imported_case_rate': True,
            'quarantine_rate': True,
            'dormitory_indicator': True,
            'household_sar': 0.183,
            'healthcare_quality_factor': 0.3
        },
        'neural_specific_features': {
            'household_mixing_ratio': 'fraction of household vs community contacts',
            'age_assortativity': 'young-to-old contact mixing',
            'behavioral_heterogeneity': 'std of compliance/mobility scores',
            'sar_potential': 'infected agents × their susceptible contacts'
        },
        'complementarity_with_llm': {
            'llm_strength': 'symbolic reasoning about interventions, disease biology, policy effects',
            'neural_strength': 'fine-grained network dynamics, behavioral heterogeneity, micro-transmission',
            'fusion_benefit': 'LLM provides macro understanding + Neural provides micro predictions = accurate forecasts'
        }
    }
    
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save normalization stats
    norm_stats = {
        'tabular_mean': dataset.tabular_mean.tolist(),
        'tabular_std': dataset.tabular_std.tolist(),
        'temporal_mean': dataset.temporal_mean.tolist(),
        'temporal_std': dataset.temporal_std.tolist(),
        'graph_mean': dataset.graph_mean.tolist(),
        'graph_std': dataset.graph_std.tolist()
    }
    
    with open(model_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"✓ Saved model config and normalization stats")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✓ SINGAPORE COVID-19 NEURAL PATHWAY TRAINING COMPLETE")
    print("="*80)
    
    print(f"\n📊 Performance:")
    print(f"  Best val loss: {trainer.best_val_loss:.6f}")
    
    print(f"\n  Per-Transition Final Performance:")
    for name in ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']:
        mae = trainer.history[f'val_mae_{name}'][-1]
        rmse = trainer.history[f'val_rmse_{name}'][-1]
        r2 = trainer.history[f'val_r2_{name}'][-1]
        r2_str = f"{r2:.4f}" if r2 > -100 else "undefined"
        print(f"    {name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2_str}")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Update cluster_team.py to use 37-dim tabular encoder")
    print(f"  2. Verify transitions are being generated (check diagnostics above)")
    print(f"  3. Run full simulation with epistemic fusion (LLM + Neural)")
    print(f"  4. Compare against Singapore MOH ground truth")
    print(f"  5. Analyze complementarity: LLM macro vs Neural micro predictions")
    
    print(f"\n📁 Outputs saved to:")
    print(f"  Model: {model_dir / 'best_model.pt'}")
    print(f"  Config: {model_dir / 'model_config.json'}")
    print(f"  Training data: {neural_data_dir}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
