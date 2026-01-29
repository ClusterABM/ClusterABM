"""
Complete pipeline: Train GraphSAGE on agent graph and prepare for clustering.
Updated for Singapore COVID-19 data format.

IMPORTANT: Initial states come directly from Kaggle day_0 column (January 23, 2020).
No inference needed - states are already in profiles.json as disease_status field.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.clustering.graph_builder import GraphBuilder
from src.clustering.graphsage_trainer import GraphSAGETrainer


def status_to_state(status: int) -> str:
    """
    Convert disease_status number to SEIRD state letter.
    
    This is just for display/analysis - the actual state data comes from
    Kaggle day_0 column and is already stored in disease_status field.
    """
    return {0: 'S', 1: 'E', 2: 'I', 3: 'R', 4: 'D'}.get(status, 'S')


def main():
    print("\n" + "="*80)
    print("GRAPHSAGE TRAINING PIPELINE - SINGAPORE COVID-19")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("STEP 1: Loading Singapore COVID-19 profiles and edges...")
    print("-" * 80)
    
    # Load from Singapore data directory
    data_dir = Path("data/singapore")
    
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("\n💡 First extract Singapore data:")
        print("   python scripts/extract_singapore_data.py data/raw/singapore_covid19_cases.csv")
        sys.exit(1)
    
    profiles_path = data_dir / "profiles.json"
    edges_path = data_dir / "edges.json"
    
    if not profiles_path.exists() or not edges_path.exists():
        print(f"❌ Error: profiles.json or edges.json not found in {data_dir}")
        print("\n💡 First extract Singapore data:")
        print("   python scripts/extract_singapore_data.py data/raw/singapore_covid19_cases.csv")
        sys.exit(1)
    
    with open(profiles_path, 'r') as f:
        profiles = json.load(f)
    
    with open(edges_path, 'r') as f:
        edges = json.load(f)
    
    print(f"✓ Loaded {len(profiles)} profiles from {profiles_path}")
    print(f"✓ Loaded {len(edges)} edges from {edges_path}")
    
    # ========================================================================
    # READ INITIAL STATES FROM PROFILES (ALREADY FROM KAGGLE day_0)
    # ========================================================================
    # IMPORTANT: disease_status field in profiles.json already contains the
    # actual states from Kaggle day_0 column (January 23, 2020).
    # We just convert the number (0-4) to letters (S/E/I/R/D) for display.
    # NO INFERENCE HAPPENS HERE - just reading what's already in the data!
    
    print("\n📖 Reading initial states from profiles (from Kaggle day_0 column)...")
    
    for p in profiles:
        if 'disease_status' in p:
            # Convert number to letter for display/analysis
            # disease_status is the ACTUAL state from Kaggle day_0
            p['initial_state'] = status_to_state(p['disease_status'])
        elif 'initial_state' not in p:
            print(f"⚠️  Warning: Agent {p['agent_id']} missing disease_status field!")
            p['initial_state'] = 'S'
    
    # Print initial state distribution (these are from Kaggle day_0, not inferred!)
    from collections import Counter
    state_dist = Counter(p['initial_state'] for p in profiles)
    
    print(f"\n📊 Initial state distribution (January 23, 2020 - from Kaggle day_0):")
    for state in ['S', 'E', 'I', 'R', 'D']:
        count = state_dist.get(state, 0)
        pct = 100 * count / len(profiles) if len(profiles) > 0 else 0
        print(f"   {state}: {count:4d} ({pct:5.1f}%)")
    
    # Check infected seed
    infected = state_dist.get('E', 0) + state_dist.get('I', 0)
    if infected == 0:
        print(f"\n⚠️  WARNING: No infected agents (E or I) on January 23, 2020!")
        print(f"   Check your Kaggle day_0 column has proper states")
    else:
        print(f"\n✅ Infected seed (Jan 23, 2020): {infected} agents ({100*infected/len(profiles):.1f}%)")
    
    # Print Singapore-specific stats
    imported = sum(1 for p in profiles if p.get('is_imported', False))
    print(f"✅ Imported cases: {imported} ({100*imported/len(profiles):.1f}%)")
    
    clusters = set(p.get('cluster', '') for p in profiles 
                   if p.get('cluster') and str(p.get('cluster')).lower() not in ['', 'nan', 'none'])
    print(f"✅ Unique clusters: {len(clusters)}")
    
    # ========================================================================
    # STEP 2: BUILD GRAPH
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Building graph structures...")
    print("-" * 80)
    
    # Build directly from profiles and edges
    builder = GraphBuilder(profiles, edges)
    
    # Build NetworkX graph
    nx_graph = builder.build_networkx_graph()
    
    # Get graph statistics
    print("\n📈 Graph Statistics:")
    stats = builder.get_graph_statistics()
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Edges: {stats['num_edges']}")
    print(f"   Average degree: {stats['avg_degree']:.2f}")
    print(f"   Density: {stats['density']:.4f}")
    print(f"   Connected: {stats['is_connected']}")
    print(f"   Average clustering: {stats['avg_clustering']:.3f}")
    print(f"\n   Edge types:")
    for edge_type, count in stats['edge_types'].items():
        pct = 100 * count / stats['num_edges']
        print(f"     {edge_type}: {count} ({pct:.1f}%)")
    
    # Convert to PyTorch Geometric
    pyg_data = builder.to_pyg_data()
    print(f"\n✓ PyG data created: {pyg_data}")
    
    # ========================================================================
    # STEP 3: VISUALIZE GRAPH
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Visualizing agent graph...")
    print("-" * 80)
    
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize the graph
    try:
        builder.visualize_graph(save_path=output_dir / "agent_graph.png")
        print(f"✓ Graph visualization saved to {output_dir / 'agent_graph.png'}")
    except Exception as e:
        print(f"⚠️  Warning: Could not visualize graph: {e}")
    
    # ========================================================================
    # STEP 4: TRAIN GRAPHSAGE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Training GraphSAGE model...")
    print("-" * 80)
    
    # GraphSAGE configuration
    config = {
        'hidden_channels': 64,
        'out_channels': 128,
        'learning_rate': 0.01,
        'dropout': 0.1,
        'epochs': 200
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"   Hidden channels: {config['hidden_channels']}")
    print(f"   Output channels: {config['out_channels']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Epochs: {config['epochs']}")
    
    trainer = GraphSAGETrainer(
        data=pyg_data,
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        learning_rate=config['learning_rate'],
        dropout=config['dropout']
    )
    
    # Train unsupervised (link prediction)
    print("\n🚀 Training GraphSAGE (unsupervised link prediction)...")
    embeddings_tensor, losses = trainer.train_unsupervised(
        epochs=config['epochs'],
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Embedding shape: {embeddings_tensor.shape}")
    
    # Plot training loss
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('GraphSAGE Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=300, bbox_inches='tight')
        print(f"✓ Training loss plot saved to {output_dir / 'training_loss.png'}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not plot training loss: {e}")
    
    # Get embeddings as numpy
    embeddings_np = trainer.get_embeddings()
    
    # ========================================================================
    # STEP 5: SAVE OUTPUTS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Saving outputs...")
    print("-" * 80)
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    model_dir = processed_dir / "graphsage"
    model_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    embeddings_path = model_dir / "graphsage_embeddings.npy"
    np.save(embeddings_path, embeddings_np)
    print(f"✓ GraphSAGE embeddings saved: {embeddings_path}")
    print(f"   Shape: {embeddings_np.shape}")
    
    # Save model
    model_path = model_dir / "graphsage_model.pt"
    trainer.save_model(str(model_path))
    print(f"✓ Model saved: {model_path}")
    
    # Save graph
    graph_path = processed_dir / "networkx_graph.pkl"
    with open(graph_path, 'wb') as f:
        pickle.dump(nx_graph, f)
    print(f"✓ NetworkX graph saved: {graph_path}")
    
    # Save PyG data
    pyg_path = processed_dir / "pyg_data.pt"
    torch.save(pyg_data, pyg_path)
    print(f"✓ PyG data saved: {pyg_path}")
    
    # Save agent metadata
    # Keep both formats: disease_status (0-4) and state (S/E/I/R/D)
    # Both come from Kaggle day_0 - disease_status is the original,
    # state is just the letter version for easy reading
    metadata_path = processed_dir / "agent_metadata.json"
    agent_data = [
        {
            'agent_id': p['agent_id'],
            'name': p['name'],
            'state': p['initial_state'],  # Letter format (S/E/I/R/D)
            'disease_status': p.get('disease_status', 0),  # Number format (0-4)
            'household_id': p['household_id'],
            'age': p['age'],
            'gender': p.get('gender', 'unknown'),
            'nationality': p.get('nationality', 'unknown'),
            'occupation': p['occupation'],
            'is_imported': p.get('is_imported', False),
            'cluster': p.get('cluster', ''),
            'days_since_start': p.get('days_since_start', 0)
        }
        for p in profiles
    ]
    with open(metadata_path, 'w') as f:
        json.dump(agent_data, f, indent=2)
    print(f"✓ Agent metadata saved: {metadata_path}")
    
    # Save training metadata
    training_metadata = {
        'dataset': 'Singapore COVID-19 (Kaggle)',
        'data_source': str(data_dir),
        'simulation_start_date': '2020-01-23',
        'note': 'Initial states from Kaggle day_0 column (not inferred)',
        'num_agents': len(profiles),
        'num_edges': len(edges),
        'embedding_dim': int(embeddings_np.shape[1]),
        'initial_state_distribution': dict(state_dist),
        'infected_seed': infected,
        'graphsage_config': config,
        'final_loss': float(losses[-1]),
        'graph_stats': stats
    }
    
    training_metadata_path = model_dir / "graphsage_metadata.json"
    with open(training_metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    print(f"✓ Training metadata saved: {training_metadata_path}")
    
    # ========================================================================
    # STEP 6: ANALYZE EMBEDDINGS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Analyzing embeddings...")
    print("-" * 80)
    
    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings_np)
    
    print("\n🔍 Most similar agent pairs (cosine similarity > 0.9):")
    # Get top similar pairs (excluding self-similarity)
    similar_pairs = []
    for i in range(len(profiles)):
        for j in range(i+1, len(profiles)):
            if similarities[i, j] > 0.9:  # High similarity threshold
                similar_pairs.append((i, j, similarities[i, j]))
    
    # Sort by similarity and show top 5
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    if similar_pairs:
        for idx, (i, j, sim) in enumerate(similar_pairs[:5], 1):
            p_i = profiles[i]
            p_j = profiles[j]
            print(f"   {idx}. {p_i['name']} ({p_i['initial_state']}) <-> {p_j['name']} ({p_j['initial_state']}): {sim:.3f}")
            if p_i.get('household_id') == p_j.get('household_id'):
                print(f"      (Same household: {p_i['household_id']})")
            if p_i.get('cluster') and p_i.get('cluster') == p_j.get('cluster'):
                print(f"      (Same cluster: {p_i['cluster']})")
    else:
        print("   (No highly similar pairs found with threshold > 0.9)")
    
    # Embedding statistics
    print(f"\n📊 Embedding Statistics:")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embeddings_np, axis=1)):.3f}")
    print(f"   Std norm: {np.std(np.linalg.norm(embeddings_np, axis=1)):.3f}")
    print(f"   Mean pairwise similarity: {np.mean(similarities[np.triu_indices_from(similarities, k=1)]):.3f}")
    
    # ========================================================================
    # STEP 7: VISUALIZE EMBEDDINGS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Visualizing embeddings with UMAP...")
    print("-" * 80)
    
    try:
        import umap
        import matplotlib.pyplot as plt
        
        print("🎨 Creating UMAP visualization...")
        
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Color by state (from Kaggle day_0)
        state_colors = {
            'S': '#2ecc71',  # Green - Susceptible
            'E': '#f39c12',  # Orange - Exposed
            'I': '#e74c3c',  # Red - Infected
            'R': '#3498db',  # Blue - Recovered
            'D': '#34495e'   # Dark gray - Dead
        }
        colors = [state_colors.get(p['initial_state'], '#95a5a6') for p in profiles]
        
        plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=colors, 
            s=150, 
            alpha=0.6, 
            edgecolors='white', 
            linewidths=1.5
        )
        
        # Add labels for small populations
        if len(profiles) <= 100:
            for i, p in enumerate(profiles):
                plt.annotate(
                    p['name'].split()[0],
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=7, alpha=0.8
                )
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label=f'Susceptible (S={state_dist["S"]})'),
            Patch(facecolor='#f39c12', label=f'Exposed (E={state_dist["E"]})'),
            Patch(facecolor='#e74c3c', label=f'Infected (I={state_dist["I"]})'),
            Patch(facecolor='#3498db', label=f'Recovered (R={state_dist["R"]})'),
            Patch(facecolor='#34495e', label=f'Dead (D={state_dist["D"]})')
        ]
        plt.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.9)
        
        plt.title("Agent Embeddings - UMAP Projection\n(colored by Kaggle day_0 states - January 23, 2020)", 
                 fontsize=15, fontweight='bold', pad=20)
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        umap_path = output_dir / "embeddings_umap.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        print(f"✓ UMAP visualization saved to {umap_path}")
        plt.close()
        
    except ImportError:
        print("⚠️  Warning: umap-learn not installed. Run: pip install umap-learn")
    except Exception as e:
        print(f"⚠️  Warning: Could not create UMAP visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ GRAPHSAGE TRAINING COMPLETE")
    print("="*80)
    
    print(f"\n📁 Output Files:")
    print(f"\n   Embeddings:")
    print(f"   ├── {embeddings_path}")
    print(f"   └── Shape: {embeddings_np.shape}")
    print(f"\n   Models:")
    print(f"   └── {model_path}")
    print(f"\n   Graphs:")
    print(f"   ├── {graph_path}")
    print(f"   └── {pyg_path}")
    print(f"\n   Metadata:")
    print(f"   ├── {training_metadata_path}")
    print(f"   └── {metadata_path}")
    print(f"\n   Visualizations:")
    print(f"   ├── {output_dir / 'agent_graph.png'} ← Network structure")
    print(f"   ├── {output_dir / 'training_loss.png'} ← Training progress")
    print(f"   └── {output_dir / 'embeddings_umap.png'} ← Embedding space")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review graph structure:")
    print(f"      open {output_dir / 'agent_graph.png'}")
    print(f"   2. Review embeddings:")
    print(f"      open {output_dir / 'embeddings_umap.png'}")
    print(f"   3. Collect LLM reasoning traces:")
    print(f"      python scripts/collect_reasoning_traces.py")
    print(f"   4. Run HSBC³ clustering:")
    print(f"      python scripts/cluster_agents_hsbc.py")
    print(f"   5. Train neural transition model:")
    print(f"      python scripts/train_neural_model.py")
    print(f"   6. Run full simulation:")
    print(f"      python scripts/run_rolling_window_simulation.py")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
