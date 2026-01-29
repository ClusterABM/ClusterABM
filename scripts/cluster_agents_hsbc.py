"""
Cluster reasoning agents using HSBC³ algorithm.
Complete pipeline from data loading to evaluation and visualization.

Singapore COVID-19 dataset.

Usage:
    # Standard clustering
    python scripts/cluster_agents_hsbc.py
    
    # Publication-quality figures (600 DPI)
    python scripts/cluster_agents_hsbc.py --publication-quality
    
    # Ablation experiments (disable components)
    python scripts/cluster_agents_hsbc.py --no-motifs
    python scripts/cluster_agents_hsbc.py --no-contrastive
    python scripts/cluster_agents_hsbc.py --no-graph

Author: HSBC³ Clustering System
Date: 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
import pickle
import networkx as nx
import argparse
sys.path.append(str(Path(__file__).parent.parent))

from src.clustering.hsbc_clustering import HSBC3Clustering
from src.clustering.cluster_visualizer import ClusterVisualizer, generate_publication_figures
from dotenv import load_dotenv


def load_data():
    """
    Load all required data for HSBC³ clustering.
        
    Returns:
        dict with embeddings, graph, reasoning_traces, agent_profiles
    """
    print("\n" + "="*80)
    print("HSBC³: DATA LOADING")
    print("="*80)
    
    data_dir = Path("data/processed")
    singapore_dir = Path("data/singapore")

    print(f"\n[1/4] Loading GraphSAGE embeddings...")
    
    # Load embeddings
    embeddings_file = data_dir / "graphsage" / "graphsage_embeddings.npy"
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"GraphSAGE embeddings not found at {embeddings_file}\n"
            f"Please run: python scripts/train_graphsage.py"
        )
    
    embeddings = np.load(embeddings_file)
    print(f"  ✓ Loaded embeddings: {embeddings.shape}")
    
    print(f"\n[2/4] Loading contact network...")
    
    # Load graph
    graph_file = data_dir / "networkx_graph.pkl"
    if not graph_file.exists():
        raise FileNotFoundError(
            f"Graph not found at {graph_file}\n"
            f"Please run data extraction script first"
        )
    
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"  ✓ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    print(f"\n[3/4] Loading reasoning traces...")
    
    # Load reasoning traces
    traces_file = data_dir / "reasoning_traces_conditional.json"
    if not traces_file.exists():
        raise FileNotFoundError(
            f"Reasoning traces not found at {traces_file}\n"
            f"Please run: python scripts/collect_reasoning_traces.py"
        )
    
    with open(traces_file, 'r') as f:
        reasoning_traces = json.load(f)
    print(f"  ✓ Loaded traces: {reasoning_traces['total_agents']} agents, "
          f"{reasoning_traces['total_conversations']} conversations")
    
    print(f"\n[4/4] Loading agent profiles...")
    
    # Load agent profiles
    profiles_file = singapore_dir / "profiles.json"
    if not profiles_file.exists():
        raise FileNotFoundError(f"Agent profiles not found at {profiles_file}")
    
    with open(profiles_file, 'r') as f:
        agent_profiles = json.load(f)
    print(f"  ✓ Loaded profiles: {len(agent_profiles)} agents")
    
    # Verify data consistency
    print(f"\n✓ Data consistency check:")
    print(f"  Embeddings: {embeddings.shape[0]} agents")
    print(f"  Graph nodes: {graph.number_of_nodes()} nodes")
    print(f"  Profiles: {len(agent_profiles)} agents")
    print(f"  Traces: {reasoning_traces['total_agents']} agents")
    
    return {
        'embeddings': embeddings,
        'graph': graph,
        'reasoning_traces': reasoning_traces,
        'agent_profiles': agent_profiles,
        'output_dir': data_dir
    }


def run_clustering(data: dict, args):
    """
    Run HSBC³ clustering pipeline.
    
    Args:
        data: Dictionary with loaded data
        args: Command-line arguments
    """
    print("\n" + "="*80)
    print("HSBC³: HIERARCHICAL SEMANTIC-BEHAVIORAL CONTRASTIVE CLUSTERING")
    print("="*80)
    print("Novel algorithm for multi-agent GABM clustering")
    print("Combines: Graph Structure + Semantic Embeddings + Behavioral Motifs + Contrastive Learning")
    print("="*80 + "\n")
    
    # =====================================================================
    # Initialize HSBC³ Clustering
    # =====================================================================
    print("[STEP 1] Initializing HSBC³ clustering...")
    
    clustering = HSBC3Clustering(
        embeddings=data['embeddings'],
        graph=data['graph'],
        reasoning_traces=data['reasoning_traces'],
        agent_profiles=data['agent_profiles'],
        k_coarse=args.k_coarse,
        k_fine=args.k_fine,
        k_motifs=args.k_motifs,
        k_archetypes=args.k_archetypes,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        theta_merge=args.theta_merge,
        theta_split=args.theta_split,
        quality_threshold=args.quality_threshold,
        # Ablation toggles
        use_graph=args.use_graph,
        use_semantic=args.use_semantic,
        use_motifs=args.use_motifs,
        use_contrastive=args.use_contrastive,
        use_boundary_opt=args.use_boundary_opt,
        cache_dir=Path('data/processed')
    )
    
    # =====================================================================
    # Run HSBC³ Pipeline
    # =====================================================================
    print("\n[STEP 2] Running HSBC³ pipeline...")
    
    cluster_labels = clustering.fit()
    
    # =====================================================================
    # Evaluate Clustering
    # =====================================================================
    print("\n[STEP 3] Evaluating clustering quality...")
    
    metrics = clustering.evaluate_clustering()
    
    # =====================================================================
    # Save Results
    # =====================================================================
    print("\n[STEP 4] Saving results...")
    
    output_dir = data['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster labels
    np.save(output_dir / "cluster_labels_hsbc3.npy", cluster_labels)
    print(f"  ✓ Saved cluster labels")
    
    # Save cluster assignments
    cluster_assignments = clustering.get_cluster_assignments()
    with open(output_dir / "cluster_assignments_hsbc3.json", 'w') as f:
        assignments_serializable = {
            int(k): [int(x) for x in v] 
            for k, v in cluster_assignments.items()
        }
        json.dump(assignments_serializable, f, indent=2)
    print(f"  ✓ Saved cluster assignments")
    
    # Save discovered motifs
    motifs_data = {}
    for motif_id, info in clustering.discovered_motifs.items():
        motifs_data[int(motif_id)] = {
            'name': info['name'],
            'size': int(info['size']),
            'examples': info['examples'] if 'examples' in info else []
        }
    
    with open(output_dir / "discovered_motifs_hsbc3.json", 'w') as f:
        json.dump(motifs_data, f, indent=2)
    print(f"  ✓ Saved discovered motifs")
    
    # Save metrics
    with open(output_dir / "clustering_metrics_hsbc3.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved clustering metrics")
    
    # Save hybrid embeddings
    np.save(output_dir / "hybrid_embeddings_hsbc3.npy", clustering.hybrid_embeddings)
    print(f"  ✓ Saved hybrid embeddings")
    
    # =====================================================================
    # Create Visualizations
    # =====================================================================
    print("\n[STEP 5] Creating visualizations...")
    
    viz_dir = Path("outputs/figures")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    if args.publication_quality:
        print("  → Generating publication-quality figures (600 DPI)...")
        generate_publication_figures(clustering, data['graph'], data['agent_profiles'],
                                     output_dir=viz_dir, dpi=600)
    else:
        print("  → Generating standard figures (300 DPI)...")
        visualizer = ClusterVisualizer(clustering, data['graph'], data['agent_profiles'])
        visualizer.plot_all(output_dir=viz_dir, dpi=300)
    
    # =====================================================================
    # Print Summary
    # =====================================================================
    print("\n" + "="*80)
    print("✓ HSBC³ CLUSTERING COMPLETE")
    print("="*80)
    
    # print(f"\n📊 Results Summary:")
    # print(f"  Algorithm: HSBC³ (Hierarchical Semantic-Behavioral Contrastive Clustering)")
    # print(f"  Number of agents: {len(data['agent_profiles'])}")
    # print(f"  Number of clusters: {clustering.num_clusters}")
    # print(f"  Behavioral motifs discovered: {clustering.k_motifs}")
    # print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
    # print(f"  Davies-Bouldin index: {metrics['davies_bouldin_index']:.3f}")
    # print(f"  Modularity: {metrics['modularity']:.3f}")
    # print(f"  Motif coherence: {metrics['motif_coherence']:.3f}")
    # print(f"  Cluster entropy: {metrics['cluster_entropy']:.3f}")
    
    print(f"\n📊 Results Summary")
    print(f"  Algorithm: HSBC³ (Hierarchical Semantic–Behavioral Contrastive Clustering)")
    print(f"  Number of agents: {len(data['agent_profiles'])}")
    print(f"  Number of clusters: {metrics['num_clusters']}")
    print(f"  Behavioral motifs discovered: {clustering.k_motifs}")

    print(f"\n  ── Geometric Quality ──")
    print(f"  Silhouette score (↑):              {metrics['silhouette_score']:.3f}")
    print(f"  Davies–Bouldin index (↓):          {metrics['davies_bouldin_index']:.3f}")
    print(f"  Calinski–Harabasz score (↑):       {metrics['calinski_harabasz_score']:.1f}")
    print(f"  Avg intra-cluster distance (↓):    {metrics['avg_intra_cluster_distance']:.3f}")
    print(f"  Avg inter-centroid distance (↑):   {metrics['avg_inter_centroid_distance']:.3f}")

    print(f"\n  ── Graph / Structural Alignment ──")
    print(f"  Modularity (↑):                    {metrics['modularity']:.3f}")
    print(f"  Avg conductance (↓):               {metrics['avg_conductance']:.3f}")

    print(f"\n  ── Cluster Distribution ──")
    print(f"  Cluster entropy (↑ balance):       {metrics['cluster_entropy']:.3f}")
    print(f"  Cluster sizes:                     {metrics['cluster_sizes']}")

    if clustering.use_motifs:
        print(f"\n  ── Behavioral Semantics (HSBC³) ──")
        print(f"  Motif coherence (↑):               {metrics['motif_coherence']:.3f}")
        print(f"  Avg motif entropy (↓):             {metrics['avg_motif_entropy']:.3f}")

    if metrics.get("anchor_representativeness", 0) > 0:
        print(f"\n  ── Anchor Quality (HSBC³) ──")
        print(f"  Anchor representativeness (↑):     {metrics['anchor_representativeness']:.3f}")

    if metrics.get("stability_ari") is not None:
        print(f"\n  ── Stability ──")
        print(f"  Stability ARI (↑):                 {metrics['stability_ari']:.3f}")

    # Show ablation configuration
    if 'ablation_config' in metrics:
        print(f"\n🔬 Ablation Configuration:")
        config = metrics['ablation_config']
        print(f"  Graph structure: {config['use_graph']}")
        print(f"  Semantic embeddings: {config['use_semantic']}")
        print(f"  Behavioral motifs: {config['use_motifs']}")
        print(f"  Contrastive learning: {config['use_contrastive']}")
        print(f"  Boundary optimization: {config['use_boundary_opt']}")
    
    print(f"\n📁 Output Files:")
    print(f"  - {output_dir / 'cluster_labels_hsbc3.npy'}")
    print(f"  - {output_dir / 'cluster_assignments_hsbc3.json'}")
    print(f"  - {output_dir / 'discovered_motifs_hsbc3.json'}")
    print(f"  - {output_dir / 'clustering_metrics_hsbc3.json'}")
    print(f"  - {output_dir / 'hybrid_embeddings_hsbc3.npy'}")
    
    print(f"\n📈 Visualizations:")
    print(f"  - {viz_dir / '01_multimodal_fusion.png'}")
    print(f"  - {viz_dir / '02_hybrid_embeddings_umap.png'}")
    print(f"  - {viz_dir / '03_ablation_study.png'}")
    print(f"  - {viz_dir / '04_baseline_comparison.png'}")
    print(f"  - {viz_dir / '05_motif_discovery.png'}")
    print(f"  - {viz_dir / '06_motif_distributions.png'}")
    print(f"  - {viz_dir / '07_stage_progression.png'}")
    print(f"  - {viz_dir / '08_quality_dashboard.png'}")
    print(f"  - ... and 7 more figures")
    
    print(f"\n🎯 Cluster Assignments:")
    id_to_profile = {p['agent_id']: p for p in data['agent_profiles']}
    
    for cluster_id, agent_ids in sorted(cluster_assignments.items()):
        cluster_agents = [id_to_profile[aid]['name'] for aid in agent_ids[:5] if aid in id_to_profile]
        remaining = len(agent_ids) - 5
        agents_str = ', '.join(cluster_agents)
        if remaining > 0:
            agents_str += f", ... ({remaining} more)"
        print(f"  Cluster {cluster_id} ({len(agent_ids)} agents): {agents_str}")
    
    print("\n" + "="*80)
    print("✓ Ready for SEIRD simulation with clustered agents!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="HSBC³: Hierarchical Semantic-Behavioral Contrastive Clustering"
    )
    
    # Clustering parameters
    parser.add_argument('--k-coarse', type=int, default=8,
                       help='Number of coarse clusters (Stage 1)')
    parser.add_argument('--k-fine', type=int, default=4,
                       help='Number of final clusters (Stage 4)')
    parser.add_argument('--k-motifs', type=int, default=20,
                       help='Number of behavioral motifs to discover')
    parser.add_argument('--k-archetypes', type=int, default=25,
                       help='Number of behavioral archetypes')
    
    # Fusion weights
    parser.add_argument('--alpha', type=float, default=0.30,
                       help='GraphSAGE weight')
    parser.add_argument('--beta', type=float, default=0.45,
                       help='Contrastive weight')
    parser.add_argument('--gamma', type=float, default=0.25,
                       help='Motif weight (dominant)')
    
    # Other parameters
    parser.add_argument('--temperature', type=float, default=0.05,
                       help='Contrastive learning temperature')
    parser.add_argument('--theta-merge', type=float, default=0.03,
                       help='JS-divergence threshold for merging')
    parser.add_argument('--theta-split', type=float, default=0.85,
                       help='Entropy threshold for splitting')
    parser.add_argument('--quality-threshold', type=float, default=0.65,
                       help='Minimum silhouette score')
    
    # Ablation toggles
    parser.add_argument('--no-graph', dest='use_graph', action='store_false',
                       help='Disable GraphSAGE embeddings')
    parser.add_argument('--no-semantic', dest='use_semantic', action='store_false',
                       help='Disable sentence transformers')
    parser.add_argument('--no-motifs', dest='use_motifs', action='store_false',
                       help='Disable behavioral motif discovery')
    parser.add_argument('--no-contrastive', dest='use_contrastive', action='store_false',
                       help='Disable contrastive learning')
    parser.add_argument('--no-boundary-opt', dest='use_boundary_opt', action='store_false',
                       help='Disable boundary optimization')
    
    # Output options
    parser.add_argument('--publication-quality', action='store_true',
                       help='Generate publication-quality figures (600 DPI)')
    
    # Set defaults
    parser.set_defaults(use_graph=True, use_semantic=True, use_motifs=True,
                       use_contrastive=True, use_boundary_opt=True)
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    try:
        # Load data
        data = load_data()
        
        # Run clustering
        run_clustering(data, args)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have run the prerequisite scripts:")
        print("  1. python scripts/extract_singapore_data.py")
        print("  2. python scripts/train_graphsage.py")
        print("  3. python scripts/collect_reasoning_traces.py")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
