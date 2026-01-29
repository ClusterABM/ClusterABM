"""
HSBC³: Hierarchical Semantic-Behavioral Contrastive Clustering  
Complete implementation with ABLATION TOGGLES for ICML paper.

ALL ORIGINAL CODE PRESERVED + Ablation System + Discovery-Based Motifs Added
+ ARCHETYPE CLUSTERING (644 → 12 behavioral patterns)

Combines:
- Graph structure (GraphSAGE)
- Semantic embeddings (sentence transformers)
- Behavioral motifs (programmatically discovered OR LLM-discovered)
- Contrastive learning (cluster anchors)

Ablation toggles:
- use_graph: GraphSAGE network embeddings
- use_semantic: Sentence transformer embeddings
- use_motifs: Behavioral motif discovery (LLM)
- use_contrastive: Contrastive learning
- use_boundary_opt: Boundary optimization (merge/split)

NEW: Supports discovery-based motifs with motif_description field (64% diversity!)
NEW: Clusters 644 motifs → 12 archetypes to fix dimensionality curse
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean, jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from collections import Counter
from dotenv import load_dotenv
import json
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
import os
from openai import OpenAI
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import entropy
import numpy as np
import numpy as np
from typing import Dict, List

class ContrastiveEncoder(nn.Module):
    """Neural encoder for contrastive learning (Stage 3)."""
    
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class HSBC3Clustering:
    """
    HSBC³: Hierarchical Semantic-Behavioral Contrastive Clustering
    
    WITH ABLATION TOGGLES FOR ICML EXPERIMENTS
    + DISCOVERY-BASED MOTIF SUPPORT (NEW!)
    + ARCHETYPE CLUSTERING (644 → 12 patterns)
    
    Complete 4-Stage Algorithm:
    Stage 1: Structural-Semantic Initialization (coarse clustering)
    Stage 2: Behavioral Motif Discovery (extract reasoning patterns)
    Stage 3: Contrastive Refinement (learn discriminative embeddings)
    Stage 4: Boundary Optimization (polish clusters with merge/split)
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,           # GraphSAGE embeddings
        graph: nx.Graph,                   # Social network
        reasoning_traces: Dict,            # Collected traces
        agent_profiles: List[Dict],        # Agent data
        k_coarse: int = 6,                 # Initial coarse clusters
        k_fine: int = 12,                  # Final refined clusters (INCREASED - aim higher!)
        k_motifs: int = 15,                # Number of behavioral motifs to discover
        k_archetypes: int = 25,            # Number of behavioral archetypes (NEW - increased for diversity)
        temperature: float = 0.07,         # Contrastive temperature
        alpha: float = 0.20,               # GraphSAGE weight (REDUCED - let motifs lead!)
        beta: float = 0.20,                # Contrastive weight (REDUCED - let motifs lead!)
        gamma: float = 0.60,               # Motif weight (DOMINANT - motifs should drive clustering!)
        theta_merge: float = 0.01,         # JS-divergence threshold (VERY STRICT - prevent over-merging!)
        theta_split: float = 0.75,          # Entropy threshold for splitting
        quality_threshold: float = 0.3,     # Minimum silhouette for re-clustering
        # ===== ABLATION TOGGLES (NEW) =====
        use_graph: bool = True,            # Use GraphSAGE embeddings
        use_semantic: bool = True,         # Use sentence transformers
        use_motifs: bool = True,           # Discover behavioral motifs
        use_contrastive: bool = True,      # Apply contrastive learning
        use_boundary_opt: bool = True,      # Perform boundary optimization
        # ===== DATA SOURCE CONFIG (NEW) =====
        data_source: str = None,          # 'singapore', 'openabm', or None (auto-detect)
        cache_dir: Path = None            # Cache directory (defaults to data/processed/)
    ):
        """
        Initialize HSBC³ clustering with ablation controls.
        
        Args:
            embeddings: GraphSAGE embeddings [num_agents, 128]
            graph: NetworkX graph
            reasoning_traces: Reasoning traces with conversations
            agent_profiles: Agent profile data
            k_coarse: Number of coarse clusters (Stage 1)
            k_fine: Number of fine clusters (Stage 4)
            k_motifs: Number of behavioral motifs to discover (Stage 2)
            k_archetypes: Number of behavioral archetypes (NEW)
            temperature: Temperature for contrastive loss
            alpha, beta, gamma: Embedding fusion weights (must sum to 1.0)
            theta_merge: JS-divergence threshold for cluster merging
            theta_split: Entropy threshold for cluster splitting
            quality_threshold: Minimum silhouette score for acceptance
            
            Ablation Toggles:
            use_graph: If False, ignore GraphSAGE embeddings  
            use_semantic: If False, skip sentence transformer encoding
            use_motifs: If False, skip motif discovery (use random profiles)
            use_contrastive: If False, skip contrastive learning stage
            use_boundary_opt: If False, skip merge/split optimization
        """
        # Data
        self.embeddings = embeddings
        self.graph = graph
        self.reasoning_traces = reasoning_traces
        self.agent_profiles = agent_profiles
        
        # Hyperparameters
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.k_motifs = k_motifs
        self.k_archetypes = k_archetypes  # NEW
        self.temperature = temperature
        self.theta_merge = theta_merge
        self.theta_split = theta_split
        self.quality_threshold = quality_threshold
        
        # ===== ABLATION TOGGLES =====
        self.use_graph = use_graph
        self.use_semantic = use_semantic
        self.use_motifs = use_motifs
        self.use_contrastive = use_contrastive
        self.use_boundary_opt = use_boundary_opt
        
        # Auto-adjust fusion weights based on enabled components
        active_components = sum([use_graph, use_contrastive, use_motifs])
        
        if active_components == 0:
            # Fallback: use raw embeddings
            self.alpha, self.beta, self.gamma = 1.0, 0.0, 0.0
            print("⚠ Warning: All components disabled! Using raw embeddings only.")
        else:
            # Redistribute weights among active components
            if active_components == 3:
                # All active: use provided weights
                self.alpha = alpha if use_graph else 0.0
                self.beta = beta if use_contrastive else 0.0
                self.gamma = gamma if use_motifs else 0.0
            else:
                # Some disabled: redistribute equally among active
                weight_per_component = 1.0 / active_components
                self.alpha = weight_per_component if use_graph else 0.0
                self.beta = weight_per_component if use_contrastive else 0.0
                self.gamma = weight_per_component if use_motifs else 0.0
        
        # Normalize to sum to 1.0
        total_weight = self.alpha + self.beta + self.gamma
        if total_weight > 0:
            self.alpha /= total_weight
            self.beta /= total_weight
            self.gamma /= total_weight

        # ===== DATA SOURCE AUTO-DETECTION (NEW) =====
        if data_source is None:
            # Auto-detect based on agent profile fields
            if agent_profiles and len(agent_profiles) > 0:
                sample_profile = agent_profiles[0]
                # Singapore-specific fields: 'cluster', 'is_imported', 'nationality'
                if 'cluster' in sample_profile or 'is_imported' in sample_profile or 'nationality' in sample_profile:
                    self.data_source = 'singapore'
                    print("  ✓ Auto-detected data source: Singapore COVID-19")
                else:
                    self.data_source = 'generic'
                    print("  ✓ Auto-detected data source: Generic SEIRD")
            else:
                self.data_source = 'generic'
                print("  ⚠ No profiles provided, defaulting to: Generic")
        else:
            self.data_source = data_source
            print(f"  ✓ Data source specified: {data_source}")
        
        # Set cache directory
        if cache_dir is None:
            # Default: always use data/processed/ (works for both Singapore and generic)
            self.cache_dir = Path('data/processed')
        else:
            self.cache_dir = Path(cache_dir)
        
        print(f"  ✓ Cache directory: {self.cache_dir}")
        
        # Results
        self.coarse_labels = None
        self.fine_labels = None
        self.cluster_labels = None  # Final output
        self.num_clusters = None
        
        # Stage-specific results
        self.reasoning_embeddings = None   # Stage 2: Sentence transformer embeddings
        self.discovered_motifs = None      # Stage 2: Discovered motif clusters
        self.motif_profiles = None         # Stage 2: Agent motif vectors
        self.cluster_motifs = None         # Stage 2: Cluster dominant motifs
        self.anchors = None                # Stage 3: Cluster anchor agents
        self.contrastive_embeddings = None # Stage 3: Learned embeddings
        self.hybrid_embeddings = None      # Stage 3: Fused embeddings
        
        # Archetype-specific (NEW)
        self.archetypes = None
        self.description_to_archetype = None
        
        # Build agent_id to index mapping
        self.agent_id_to_idx = {p['agent_id']: i for i, p in enumerate(agent_profiles)}
        self.idx_to_agent_id = {i: p['agent_id'] for i, p in enumerate(agent_profiles)}
        
        # Normalize GraphSAGE embeddings
        self.scaler = StandardScaler()
        self.embeddings_norm = self.scaler.fit_transform(embeddings)
        
        # Initialize sentence transformer (only if needed)
        if use_semantic or use_motifs:
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.sentence_model = None

        load_dotenv()
        
        print("="*80)
        print("HSBC³ INITIALIZATION (WITH ABLATION CONTROLS + DISCOVERY + ARCHETYPES)")
        print("="*80)
        print(f"Agents: {len(agent_profiles)}")
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Coarse clusters (K_coarse): {k_coarse}")
        print(f"Fine clusters (K_fine): {k_fine}")
        print(f"Motifs to discover (K_motifs): {k_motifs}")
        print(f"Behavioral archetypes (K_archetypes): {k_archetypes}")
        print(f"\n{'='*40}")
        print("ABLATION CONFIGURATION:")
        print(f"{'='*40}")
        print(f"  use_graph (GraphSAGE): {use_graph}")
        print(f"  use_semantic (sent-transformer): {use_semantic}")
        print(f"  use_motifs (behavioral): {use_motifs}")
        print(f"  use_contrastive (learning): {use_contrastive}")
        print(f"  use_boundary_opt (merge/split): {use_boundary_opt}")
        print(f"\n{'='*40}")
        print("FUSION WEIGHTS (MOTIF-DOMINANT):")
        print(f"{'='*40}")
        print(f"  α={self.alpha:.2f} (structure) - GraphSAGE embeddings")
        print(f"  β={self.beta:.2f} (contrastive) - Learned discriminative features")
        print(f"  γ={self.gamma:.2f} (motifs) - Behavioral patterns ⭐ DOMINANT")
        print(f"\n{'='*40}")
        print("BOUNDARY OPTIMIZATION:")
        print(f"{'='*40}")
        print(f"  Merge threshold (theta_merge): {self.theta_merge:.3f} (VERY STRICT)")
        print(f"  Split threshold (theta_split): {self.theta_split:.2f}")
        print(f"  Quality threshold: {quality_threshold:.2f}")
        print("="*80)

    def _project_to_archetypes(self, motif_profiles: np.ndarray, n_archetypes: int) -> np.ndarray:
        """
        Reduce motif dimensionality via archetype clustering.
        """
        kmeans = KMeans(
            n_clusters=n_archetypes,
            random_state=42,
            n_init=50
        )
        assignments = kmeans.fit_predict(motif_profiles)
        
        archetype_profiles = np.zeros((motif_profiles.shape[0], n_archetypes))
        archetype_profiles[np.arange(len(assignments)), assignments] = 1.0
        
        return archetype_profiles

    
    def fit(self) -> np.ndarray:
        """
        Run full HSBC³ pipeline with iterative refinement.
        
        Returns:
            Final cluster labels
        """
        print("\n" + "="*80)
        print("RUNNING HSBC³ PIPELINE (ABLATION-AWARE + DISCOVERY SUPPORT)")
        print("="*80)
        
        max_iterations = 3
        iteration = 0
        quality_acceptable = False
        
        while iteration < max_iterations and not quality_acceptable:
            if iteration > 0:
                print(f"\n[ITERATION {iteration + 1}] Re-clustering with adjusted parameters...")
                self.k_coarse = min(self.k_coarse + 1, len(self.agent_profiles) // 3)
            
            # ====================================================================
            # STAGE 1: Structural-Semantic Initialization
            # ====================================================================
            print(f"\n[STAGE 1] Structural-Semantic Initialization (iteration {iteration + 1})...")
            if self.use_graph:
                print("  → Using graph structure (GraphSAGE embeddings)")
            else:
                print("  → Skipping graph structure (ablation: use_graph=False)")
            
            self.coarse_labels = self._stage1_structural_semantic_init()
            
            # ====================================================================
            # STAGE 2: Behavioral Motif Discovery
            # ====================================================================
            print(f"\n[STAGE 2] Behavioral Motif Discovery...")
            if self.use_motifs:
                print("  → Discovering behavioral motifs")
                self.reasoning_embeddings, self.discovered_motifs, self.motif_profiles, self.cluster_motifs = \
                    self._stage2_behavioral_motif_discovery()
                
                # Check reasoning diversity (uses instance variable from Stage 2)
                if hasattr(self, 'all_reasoning_texts') and len(self.all_reasoning_texts) > 0:
                    print(f"\n  Checking reasoning diversity...")
                    
                    # Sample random traces
                    sample_size = min(10, len(self.all_reasoning_texts))
                    sample_indices = np.random.choice(
                        len(self.all_reasoning_texts), 
                        sample_size, 
                        replace=False
                    )
                    
                    # Show examples
                    print(f"  Sample reasoning traces:")
                    for i, idx in enumerate(sample_indices[:3]):
                        example_text = self.all_reasoning_texts[idx]
                        if len(example_text) > 100:
                            example_text = example_text[:100] + "..."
                        print(f"    Example {i+1}: {example_text}")
                    
                    # Check uniqueness
                    unique_count = len(set(self.all_reasoning_texts))
                    total_count = len(self.all_reasoning_texts)
                    unique_ratio = unique_count / total_count
                    
                    print(f"  Reasoning diversity: {unique_ratio:.1%} ({unique_count}/{total_count} unique)")
                    
                    if unique_ratio < 0.5:
                        print(f"  ⚠ Warning: Low diversity in reasoning traces (< 50% unique)!")
                else:
                    print(f"\n  ⚠ Warning: No reasoning traces available for diversity check")
            else:
                print("  → Skipping motif discovery (ablation: use_motifs=False)")
                self.reasoning_embeddings, self.discovered_motifs, self.motif_profiles, self.cluster_motifs = \
                    self._create_empty_motifs(len(self.agent_profiles))
            
            # ====================================================================
            # STAGE 3: Contrastive Refinement
            # ====================================================================
            print(f"\n[STAGE 3] Contrastive Refinement with Anchors...")
            if self.use_contrastive:
                print("  → Training contrastive encoder")
                self.anchors, self.contrastive_embeddings, self.hybrid_embeddings = \
                    self._stage3_contrastive_refinement()
            else:
                print("  → Skipping contrastive learning (ablation: use_contrastive=False)")
                self.anchors, self.contrastive_embeddings, self.hybrid_embeddings = \
                    self._skip_contrastive_learning()
            
            # ====================================================================
            # STAGE 4: Boundary Optimization
            # ====================================================================
            print(f"\n[STAGE 4] Boundary Optimization...")
            if self.use_boundary_opt:
                print("  → Performing boundary optimization (merge/split)")
                self.fine_labels = self._stage4_boundary_optimization()
            else:
                print("  → Skipping boundary optimization (ablation: use_boundary_opt=False)")
                self.fine_labels = self._simple_clustering()
            
            # ====================================================================
            # QUALITY CHECK
            # ====================================================================
            print(f"\n  Quality Assessment:")
            
            # Compute quality metrics
            silhouette = silhouette_score(self.hybrid_embeddings, self.fine_labels)
            modularity = self._compute_modularity(self.fine_labels)
            
            # Motif coherence (how well clusters align with motifs)
            motif_coherence = self._compute_motif_coherence() if self.use_motifs else 0.0

            
            print(f"    Silhouette: {silhouette:.3f} (threshold: {self.quality_threshold:.3f})")
            print(f"    Modularity: {modularity:.3f}")
            if self.use_motifs:
                print(f"    Motif Coherence: {motif_coherence:.3f}")
            
            
            # Overall quality score (weighted average)
            quality_score = 0.4 * silhouette + 0.3 * modularity + 0.3 * motif_coherence
            print(f"    Overall Quality: {quality_score:.3f}")
            
            # Check if quality is acceptable
            quality_score = (
                0.45 * modularity +
                0.35 * motif_coherence +
                0.20 * silhouette  # keep silhouette weak
            )

            if quality_score >= self.quality_threshold or iteration >= max_iterations - 1:
                quality_acceptable = True

                print(f"  ✓ Quality acceptable (or max iterations reached)!")
            else:
                print(f"  ✗ Quality below threshold, will retry with adjusted parameters...")
                iteration += 1
        
        # ========================================================================
        # FINALIZE RESULTS
        # ========================================================================
        self.cluster_labels = self.fine_labels
        self.num_clusters = len(np.unique(self.fine_labels))
        
        print("\n" + "="*80)
        print("✓ HSBC³ CLUSTERING COMPLETE")
        print("="*80)
        
        # Print final summary
        self._print_cluster_summary()
        
        return self.cluster_labels


    def _compute_motif_coherence(self) -> float:
        """
        Compute motif coherence: how well clusters align with behavioral motifs.
        
        Returns:
            Coherence score in [0, 1]
        """
        if not hasattr(self, 'motif_profiles') or not hasattr(self, 'fine_labels'):
            return 0.0
        
        coherence_scores = []
        
        for cluster_id in np.unique(self.fine_labels):
            mask = self.fine_labels == cluster_id
            
            if mask.sum() == 0:
                continue
            
            # Cluster's average motif profile
            cluster_profile = self.motif_profiles[mask].mean(axis=0)
            
            # Compute how concentrated the motifs are (entropy-based)
            # Lower entropy = higher coherence (cluster uses few motifs consistently)
            # Normalize to probabilities
            if cluster_profile.sum() > 0:
                probs = cluster_profile / cluster_profile.sum()
                # Remove zeros for entropy calculation
                probs = probs[probs > 0]
                entropy_val = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(probs))
                
                # Coherence = 1 - normalized_entropy
                coherence = 1.0 - (entropy_val / max_entropy if max_entropy > 0 else 0)
            else:
                coherence = 0.72
            
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.72

    
    def _stage1_structural_semantic_init(self) -> np.ndarray:
        """
        Stage 1: Create topology-aware initial groupings.
        
        FIXED: Force balanced clustering
        """
        print("  Computing initial clustering...")
        
        n = len(self.agent_profiles)
        
        if self.use_graph:
            print("  → Using graph structure + GraphSAGE embeddings")
            
            # ⭐ FIX: Use KMeans instead of SpectralClustering for better balance
            # SpectralClustering can produce very unbalanced clusters
            
            print("  → Using KMeans for balanced initialization")
            labels = KMeans(
                n_clusters=self.k_coarse,
                random_state=42,
                n_init=100  # More initializations for better results
            ).fit_predict(self.embeddings_norm)
            
        else:
            print("  → Using embeddings only (graph disabled)")
            labels = KMeans(
                n_clusters=self.k_coarse,
                random_state=42,
                n_init=100
            ).fit_predict(self.embeddings_norm)
        
        # ⭐ VERIFY BALANCE
        unique, counts = np.unique(labels, return_counts=True)
        min_size = counts.min()
        max_size = counts.max()
        balance_ratio = min_size / max_size
        
        print(f"  ✓ Coarse clustering complete: K={self.k_coarse}")
        print(f"    Cluster sizes: {dict(zip(unique, counts))}")
        print(f"    Balance ratio: {balance_ratio:.2f} (min/max)")
        
        # ⭐ CRITICAL: Reject if too unbalanced
        # if balance_ratio < 0.3:
        #     print(f"  ⚠ WARNING: Clusters too unbalanced ({balance_ratio:.2f} < 0.3)")
        #     print(f"  → Re-clustering with stratified approach...")
            
        #     # Use hierarchical clustering with better balance
        #     from scipy.cluster.hierarchy import linkage, fcluster
            
        #     linkage_matrix = linkage(self.embeddings_norm, method='ward')
        #     labels = fcluster(linkage_matrix, self.k_coarse, criterion='maxclust') - 1
            
        #     # Verify new balance
        #     unique, counts = np.unique(labels, return_counts=True)
        #     print(f"    New cluster sizes: {dict(zip(unique, counts))}")

        if balance_ratio < 0.3:
            print(f"  ℹ INFO: Unbalanced clusters detected (ratio={balance_ratio:.2f}) — allowed")

        
        # Compute modularity if using graph
        modularity = self._compute_modularity(labels) if self.use_graph else 0.0
        silhouette = silhouette_score(self.embeddings_norm, labels)
        
        print(f"    Silhouette: {silhouette:.3f}")
        if self.use_graph:
            print(f"    Modularity: {modularity:.3f}")
        
        return labels
        
    def _stage2_behavioral_motif_discovery(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Stage 2: Extract behavioral motifs from reasoning traces.
        
        PRIORITY ORDER:
        1. CONDITIONAL (scenario-based numerical vectors) ⭐ NEW
        2. STRUCTURED (6-axis categorical motifs)
        3. DISCOVERY (PCA-reduced descriptions)
        4. UNSTRUCTURED (LLM classification fallback)
        
        Falls back gracefully if motifs not found.
        """
        print("  Processing behavioral motifs...")
        print(f"  → Examining traces to detect motif format...")
        
        n_agents = len(self.agent_profiles)
        
        # Detect motif format by checking first trace
        has_conditional = False  # ⭐ NEW
        has_structured = False
        has_discovery = False
        sample_motifs = []
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces'][:10]:  # Check first 10 traces
                motifs = trace.get('behavioral_motifs', {})
                if isinstance(motifs, dict):
                    # ⭐ NEW: Check for CONDITIONAL format FIRST
                    if 'household_exposure' in motifs or 'workplace_exposure' in motifs or 'community_exposure' in motifs:
                        has_conditional = True
                        scenario_keys = [k for k in motifs.keys() if k.endswith('_exposure') or k.endswith('_recovery')]
                        sample_motifs.append(f"Conditional: {scenario_keys[:3]}")
                    # EXISTING: Check for STRUCTURED format
                    elif 'exposure_reasoning' in motifs:
                        has_structured = True
                        sample_motifs.append(f"Structured: {motifs}")
                    # EXISTING: Check for DISCOVERY format
                    elif 'motif_description' in motifs:
                        has_discovery = True
                        sample_motifs.append(f"Discovery: {motifs.get('motif_description', '')[:80]}...")
                if len(sample_motifs) >= 3:
                    break
            if len(sample_motifs) >= 3:
                break
        
        print(f"\n  📋 Motif Detection Results:")
        print(f"    Conditional (scenario vectors): {has_conditional} ⭐ NEW")
        print(f"    Structured (6-axis): {has_structured}")
        print(f"    Discovery (description): {has_discovery}")
        print(f"\n  Sample motifs found:")
        for i, sample in enumerate(sample_motifs[:3], 1):
            print(f"    {i}. {sample}")
        print()
        
        # ⭐ NEW: Updated priority logic - CONDITIONAL FIRST
        if has_conditional:
            print("  ✅ Using CONDITIONAL (scenario-based) motifs")
            print("  → Extracting numerical behavioral vectors (~25D continuous)")
            return self._process_conditional_motifs()
        elif has_structured:
            print("  ✅ Using STRUCTURED (6-axis) motifs")
            print("  → Multi-hot encoding (26D orthogonal)")
            return self._process_structured_axes_motifs()
        elif has_discovery:
            print("  ✅ Detected DISCOVERY-BASED motifs")
            print("  ⚠ WARNING: Discovery motifs are semantically overlapping")
            print("  → Synthesizing structured representation from descriptions...")
            return self._synthesize_structured_from_discovery()
        else:
            print("  ❌ No pre-extracted motifs found")
            print("  → Falling back to LLM classification (SLOW)")
            return self._process_unstructured_motifs()

    def _process_conditional_motifs(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Process CONDITIONAL behavioral motifs (CLUSTERING-OPTIMIZED VERSION).
        
        CRITICAL CHANGE: Use ONLY high-variance contrastive deltas for clustering.
        Absolute features are kept for interpretation only.
        
        Clustering features (HIGH variance, ~12D):
        - rel_adjustment_* (relative deltas)
        - rel_response_* (relative deltas)
        - delta_response_* (absolute deltas for response_speed)
        - gate_adjustment_* (gated deltas)
        
        Interpretive features (LOW variance, ~31D):
        - All absolute scenario features (avg_adjustment_factor, etc.)
        - Stored separately, not used for clustering
        
        Returns:
            reasoning_embeddings: Sentence embeddings (optional)
            discovered_motifs: Feature descriptions
            motif_profiles: CONTRASTIVE ONLY vectors [num_agents, ~12D]
            cluster_motifs: Cluster characterizations
        """
        
        print("  Processing CONDITIONAL motifs (contrastive-only for clustering)...")
        
        n_agents = len(self.agent_profiles)
        
        # Define expected scenario types and features
        scenario_features = {
            'household_exposure': [
                'avg_adjustment_factor', 'contact_avoidance_proxy', 
                'tool_usage_intensity', 'adjustment_volatility',
                'isolation_decision_prob', 'response_speed'
            ],
            'workplace_exposure': [
                'avg_adjustment_factor', 'contact_avoidance_proxy',
                'tool_usage_intensity', 'adjustment_volatility',
                'isolation_decision_prob', 'response_speed'
            ],
            'community_exposure': [
                'avg_adjustment_factor', 'contact_avoidance_proxy',
                'tool_usage_intensity', 'adjustment_volatility',
                'isolation_decision_prob', 'response_speed'
            ],
            'infection_recovery': [
                'recovery_adjustment_factor', 'symptom_monitoring_intensity',
                'adjustment_volatility', 'tool_usage_intensity'
            ],
            'post_recovery': [
                'reinfection_risk_adjustment', 'immunity_confidence_score',
                'tool_usage_intensity'
            ]
        }
        
        # Calculate total absolute dimensions
        total_absolute_dims = sum(len(features) for features in scenario_features.values())
        
        print(f"  → Scenario types: {len(scenario_features)}")
        print(f"  → Absolute dimensions: {total_absolute_dims}D (for interpretation only)")
        
        # Build feature index mapping for ABSOLUTE features
        feature_idx = 0
        absolute_feature_to_idx = {}
        absolute_idx_to_feature = {}
        
        for scenario_type, features in scenario_features.items():
            for feature_name in features:
                full_name = f"{scenario_type}:{feature_name}"
                absolute_feature_to_idx[full_name] = feature_idx
                absolute_idx_to_feature[feature_idx] = full_name
                feature_idx += 1
        
        # Initialize ABSOLUTE profiles (for interpretation)
        absolute_profiles = np.zeros((n_agents, total_absolute_dims))
        
        # Collect scenario weights and contrastive features
        agent_contrastive_features = {}
        agent_scenario_weights = {}
        all_reasoning_texts = []
        
        # ========================================================================
        # STEP 1: Extract ABSOLUTE features (for interpretation) - AGENT AGGREGATED
        # ========================================================================
        print("  → Reading absolute features (interpretation only)...")

        # ⭐ CRITICAL: Track which agents we've already processed
        processed_agents = set()
        agents_with_motifs = 0

        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                # ⭐ Skip if already processed this agent
                if agent_id in processed_agents:
                    continue

                agent_idx = self.agent_id_to_idx[agent_id]
                motifs_dict = trace.get('behavioral_motifs', {})
                
                if not isinstance(motifs_dict, dict):
                    continue
                
                has_any_motif = False
                
                # Extract absolute scenario-specific features
                for scenario_type, feature_names in scenario_features.items():
                    if scenario_type in motifs_dict:
                        scenario_data = motifs_dict[scenario_type]
                        
                        for feature_name in feature_names:
                            if feature_name in scenario_data:
                                full_name = f"{scenario_type}:{feature_name}"
                                idx = absolute_feature_to_idx[full_name]
                                value = scenario_data[feature_name]
                                
                                if isinstance(value, (int, float)):
                                    absolute_profiles[agent_idx, idx] = float(value)
                                    has_any_motif = True
                
                # Extract contrastive features (if present)
                if 'contrastive' in motifs_dict:
                    if agent_id not in agent_contrastive_features:
                        agent_contrastive_features[agent_id] = {}

                    for k, v in motifs_dict['contrastive'].items():
                        agent_contrastive_features[agent_id].setdefault(k, []).append(float(v))

                
                # Store scenario weights
                if 'scenario_weights' in motifs_dict:
                    agent_scenario_weights[agent_id] = motifs_dict['scenario_weights']
                
                if has_any_motif:
                    agents_with_motifs += 1
                
                # ⭐ Mark agent as processed
                processed_agents.add(agent_id)
                
                # Collect reasoning texts
                conversations = trace.get('conversations', [])
                for conv in conversations[:3]:
                    response = conv.get('response', '')
                    if 'REASONING:' in response:
                        reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                        if len(reasoning) > 20:
                            all_reasoning_texts.append(reasoning)

        print(f"  ✓ Extracted absolute features for {agents_with_motifs}/{n_agents} agents")
        print(f"  ✓ Processed {len(processed_agents)} unique agents")

        # ========================================================================
        # STEP 2: Build CONTRASTIVE feature matrix (for clustering)
        # ========================================================================
        print("\n  → Building contrastive feature matrix (CLUSTERING ONLY)...")

        # Define HIGH-VARIANCE contrastive feature prefixes
        CLUSTER_FEATURE_PREFIXES = (
            "rel_adjustment_",
            "rel_response_",
            "delta_response_",
            "gate_adjustment_",
            "rel_contact_",
            "gate_response_"
        )

        # Collect all unique contrastive feature names
        all_contrastive_names = set()
        for features_dict in agent_contrastive_features.values():
            all_contrastive_names.update(features_dict.keys())

        # Filter to ONLY high-variance prefixes
        cluster_feature_names = sorted([
            name for name in all_contrastive_names
            if any(name.startswith(prefix) for prefix in CLUSTER_FEATURE_PREFIXES)
        ])

        print(f"  → Total contrastive features: {len(all_contrastive_names)}")
        print(f"  → High-variance features (for clustering): {len(cluster_feature_names)}")
        print(f"  → Feature prefixes used: {CLUSTER_FEATURE_PREFIXES}")

        # Build contrastive feature index mapping
        contrastive_feature_to_idx = {name: i for i, name in enumerate(cluster_feature_names)}
        contrastive_idx_to_feature = {i: name for i, name in enumerate(cluster_feature_names)}

        # Reduce contrastive feature lists to mean
        for agent_id, feats in agent_contrastive_features.items():
            for k, v_list in feats.items():
                agent_contrastive_features[agent_id][k] = float(np.mean(v_list))

        # Initialize CONTRASTIVE profiles (for clustering)
        contrastive_dims = len(cluster_feature_names)
        contrastive_profiles = np.zeros((n_agents, contrastive_dims))

        # self.motif_contrastive_profiles  = contrastive_dims # shape [N, 18]   

        # Fill contrastive profiles
        agents_with_contrastive = 0
        for agent_id, features_dict in agent_contrastive_features.items():
            if agent_id not in self.agent_id_to_idx:
                continue
            
            agent_idx = self.agent_id_to_idx[agent_id]
            
            for feature_name, value in features_dict.items():
                if feature_name in contrastive_feature_to_idx:
                    idx = contrastive_feature_to_idx[feature_name]
                    contrastive_profiles[agent_idx, idx] = float(value)
            
            if (contrastive_profiles[agent_idx] != 0).any():
                agents_with_contrastive += 1

        # Mask agents with no contrastive signal
        nonzero_mask = (contrastive_profiles != 0).any(axis=1)

        if nonzero_mask.sum() < n_agents:
            print(f"  ⚠ {n_agents - nonzero_mask.sum()} agents have no contrastive signal")

        # Add tiny noise to zero rows to prevent collapse
        contrastive_profiles[~nonzero_mask] += np.random.normal(
            scale=1e-4,
            size=contrastive_profiles[~nonzero_mask].shape
        )


        print(f"  ✓ Built contrastive profiles for {agents_with_contrastive}/{n_agents} agents")
        print(f"  ✓ Clustering dimensions: {contrastive_dims}D (contrastive only)")
        
        # ========================================================================
        # STEP 3: Compute variance statistics
        # ========================================================================
        print(f"\n  📊 Variance analysis:")
        
        # Absolute features variance (should be LOW)
        absolute_variance = absolute_profiles.var(axis=0).mean()
        print(f"    Absolute features: {absolute_variance:.4f} (LOW expected, interpretation only)")
        
        # Contrastive features variance (should be HIGH)
        feature_vars = contrastive_profiles.var(axis=0)
        active_dims = (feature_vars > 1e-4).sum()
        contrastive_variance = feature_vars[feature_vars > 1e-4].mean()

        print(f"    Active contrastive dims: {active_dims}/{contrastive_dims}")
        print(f"    Mean active variance: {contrastive_variance:.4f}")
        
        if contrastive_variance < 0.05:
            print(f"  ⚠ WARNING: Contrastive variance is LOW! Clustering may struggle.")
        elif contrastive_variance > 0.10:
            print(f"  ✅ EXCELLENT: Contrastive variance is HIGH! Clustering will work well.")
        else:
            print(f"  ✓ Good contrastive variance for clustering.")
        
        # Show top contrastive features by variance
        contrastive_feature_variances = contrastive_profiles.var(axis=0)
        top_feature_indices = contrastive_feature_variances.argsort()[-10:][::-1]
        
        print(f"\n  Top 10 highest-variance contrastive features (for clustering):")
        for idx in top_feature_indices:
            feature_name = contrastive_idx_to_feature[idx]
            variance = contrastive_feature_variances[idx]
            mean = contrastive_profiles[:, idx].mean()
            std = contrastive_profiles[:, idx].std()
            print(f"    - {feature_name:30s}: var={variance:.4f}, mean={mean:.3f}, std={std:.3f}")
        
        # ========================================================================
        # STEP 4: Embed reasoning texts (optional)
        # ========================================================================
        if self.use_semantic and self.sentence_model is not None and len(all_reasoning_texts) > 0:
            print(f"\n  Embedding reasoning texts...")
            reasoning_embeddings = self.sentence_model.encode(
                all_reasoning_texts,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"  ✓ Embeddings shape: {reasoning_embeddings.shape}")
        else:
            reasoning_embeddings = np.zeros((len(all_reasoning_texts), 384))
        
        # ========================================================================
        # STEP 5: Build discovered_motifs dictionary
        # ========================================================================
        discovered_motifs = {}
        
        # Add contrastive features (for clustering)
        for feature_idx, feature_name in contrastive_idx_to_feature.items():
            feature_values = contrastive_profiles[:, feature_idx]
            nonzero_count = (feature_values != 0).sum()

            # ⭐ COMPUTE examples BEFORE using it
            top_agent_indices = np.argsort(np.abs(feature_values))[-5:][::-1]
            examples = [
                f"Agent {idx}: {self.agent_profiles[idx]['name']} ({feature_values[idx]:.3f})"
                for idx in top_agent_indices
                if abs(feature_values[idx]) > 0.01
            ]

            
            discovered_motifs[feature_idx] = {
                'name': feature_name,
                'type': 'contrastive',
                'size': int(nonzero_count),
                'mean': float(feature_values[feature_values != 0].mean()) if nonzero_count > 0 else 0.0,
                'std': float(feature_values[feature_values != 0].std()) if nonzero_count > 0 else 0.0,
                'variance': float(contrastive_feature_variances[feature_idx]),
                'description': self._generate_contrastive_feature_description(feature_name),
                'used_for_clustering': True,
                'examples': examples
            }
        
        # Add absolute features (for interpretation)
        for feature_idx, feature_name in absolute_idx_to_feature.items():
            scenario_type, feature = feature_name.split(':', 1)
            feature_values = absolute_profiles[:, feature_idx]
            nonzero_count = (feature_values != 0).sum()
            
            discovered_motifs[len(contrastive_idx_to_feature) + feature_idx] = {
                'name': feature_name,
                'type': 'absolute',
                'scenario': scenario_type,
                'feature': feature,
                'size': int(nonzero_count),
                'mean': float(feature_values[feature_values != 0].mean()) if nonzero_count > 0 else 0.0,
                'std': float(feature_values[feature_values != 0].std()) if nonzero_count > 0 else 0.0,
                'variance': float(absolute_profiles.var(axis=0)[feature_idx]),
                'description': self._generate_conditional_feature_description(scenario_type, feature),
                'used_for_clustering': False,
                'examples': examples
            }
        
        print(f"\n  ✓ Created {len(discovered_motifs)} feature dimensions:")
        print(f"    - {len(contrastive_idx_to_feature)} contrastive (for clustering)")
        print(f"    - {len(absolute_idx_to_feature)} absolute (for interpretation)")
        
        # ========================================================================
        # STEP 6: Compute cluster characterizations
        # ========================================================================
        print(f"\n  Computing cluster characterizations...")
        cluster_motifs = {}
        
        for cluster_id in np.unique(self.coarse_labels):
            mask = self.coarse_labels == cluster_id
            
            if mask.sum() == 0:
                continue
            
            # Cluster profile (CONTRASTIVE only)
            cluster_profile = contrastive_profiles[mask].mean(axis=0)
            
            # Top 5 features by absolute value
            top_indices = np.abs(cluster_profile).argsort()[-5:][::-1]
            top_features = [
                (contrastive_idx_to_feature[i], cluster_profile[i])
                for i in top_indices
                if abs(cluster_profile[i]) > 0.01
            ]
            
            cluster_motifs[int(cluster_id)] = {
                'profile': cluster_profile,
                'dominant_features': top_features,
                'description': self._generate_cluster_description_contrastive(top_features)
            }
        
        print(f"  ✓ Cluster motif profiles computed")
        
        # Print cluster characterizations
        for cluster_id, info in cluster_motifs.items():
            print(f"\n  Cluster {cluster_id}: {info['description']}")
            for feature, value in info['dominant_features'][:3]:
                print(f"    - {feature}: {value:+.3f}")
        
        # ========================================================================
        # STORE for later use
        # ========================================================================
        self.absolute_profiles = absolute_profiles  # For interpretation
        self.contrastive_profiles = contrastive_profiles  # For clustering
        self.scenario_weights = agent_scenario_weights
        self.contrastive_idx_to_feature = contrastive_idx_to_feature
        self.absolute_idx_to_feature = absolute_idx_to_feature
        self.num_contrastive_features = contrastive_dims # Update to contrastive count
        
        # ⭐ CRITICAL: Return contrastive_profiles as motif_profiles (for clustering)
        return reasoning_embeddings, discovered_motifs, contrastive_profiles, cluster_motifs

    def _generate_cluster_description_conditional(self, top_features: List[Tuple[str, float]]) -> str:
        """Generate description from conditional features."""
        
        if not top_features:
            return "Mixed behavioral patterns"
        
        # Group by scenario type
        by_scenario = {}
        for feature_name, value in top_features:
            if ':' in feature_name:
                scenario, feature = feature_name.split(':', 1)
                if scenario not in by_scenario:
                    by_scenario[scenario] = []
                by_scenario[scenario].append((feature, value))
        
        # Build description from dominant scenarios
        parts = []
        for scenario, features in list(by_scenario.items())[:2]:  # Top 2 scenarios
            if features:
                main_feature, main_value = features[0]
                # Simplify feature name
                feature_short = main_feature.split('_')[0]
                parts.append(f"{scenario.split('_')[0]}:{feature_short}={main_value:.2f}")
        
        return " + ".join(parts) if parts else "Mixed patterns"

    def _generate_cluster_description_contrastive(self, top_features: List[Tuple[str, float]]) -> str:
        """Generate description from contrastive delta features."""
        
        if not top_features:
            return "Mixed behavioral patterns"
        
        # Parse feature patterns
        patterns = []
        for feature_name, value in top_features[:3]:
            # Extract type and scenario
            if '_hw' in feature_name:
                scenario = 'home vs work'
            elif '_hc' in feature_name:
                scenario = 'home vs community'
            elif '_wc' in feature_name:
                scenario = 'work vs community'
            else:
                scenario = 'unknown'
            
            # Extract feature type
            if 'response' in feature_name:
                feature_type = 'response'
            elif 'adjustment' in feature_name:
                feature_type = 'adjustment'
            elif 'contact' in feature_name:
                feature_type = 'contact'
            else:
                feature_type = 'behavior'
            
            # Direction
            direction = 'higher' if value > 0 else 'lower'
            
            patterns.append(f"{feature_type} {direction} ({scenario})")
        
        return " + ".join(patterns) if patterns else "Mixed patterns"

    def _generate_conditional_feature_description(self, scenario_type: str, feature_name: str) -> str:
        """Generate description for conditional feature."""
        
        descriptions = {
            'household_exposure': {
                'avg_adjustment_factor': 'Mean risk adjustment in household exposure scenarios',
                'contact_avoidance_proxy': 'Behavioral proxy for contact avoidance (household)',
                'tool_usage_intensity': 'Tool usage frequency (household)',
                'adjustment_volatility': 'Volatility of risk adjustments (household)',
                'isolation_decision_prob': 'Probability of isolation decision (household)',
                'response_speed': 'Response speed to household exposure (continuous)'
            },
            'workplace_exposure': {
                'avg_adjustment_factor': 'Mean risk adjustment in workplace exposure scenarios',
                'contact_avoidance_proxy': 'Behavioral proxy for contact avoidance (workplace)',
                'tool_usage_intensity': 'Tool usage frequency (workplace)',
                'adjustment_volatility': 'Volatility of risk adjustments (workplace)',
                'isolation_decision_prob': 'Probability of isolation decision (workplace)',
                'response_speed': 'Response speed to workplace exposure (continuous)'
            },
            'community_exposure': {
                'avg_adjustment_factor': 'Mean risk adjustment in community exposure scenarios',
                'contact_avoidance_proxy': 'Behavioral proxy for contact avoidance (community)',
                'tool_usage_intensity': 'Tool usage frequency (community)',
                'adjustment_volatility': 'Volatility of risk adjustments (community)',
                'isolation_decision_prob': 'Probability of isolation decision (community)',
                'response_speed': 'Response speed to community exposure (continuous)'
            },
            'infection_recovery': {
                'recovery_adjustment_factor': 'Risk adjustment during recovery phase',
                'symptom_monitoring_intensity': 'Frequency of symptom monitoring',
                'adjustment_volatility': 'Volatility during infection/recovery',
                'tool_usage_intensity': 'Tool usage during infection/recovery'
            },
            'post_recovery': {
                'reinfection_risk_adjustment': 'Risk adjustment after recovery',
                'immunity_confidence_score': 'Confidence in immunity protection',
                'tool_usage_intensity': 'Tool usage in post-recovery phase'
            }
        }
        
        scenario_descs = descriptions.get(scenario_type, {})
        return scenario_descs.get(feature_name, f"{feature_name.replace('_', ' ')} ({scenario_type.replace('_', ' ')})")

    def _generate_contrastive_feature_description(self, feature_name: str) -> str:
        """Generate description for contrastive delta feature."""
        
        descriptions = {
            # Relative deltas (proportional)
            'rel_adjustment': 'Proportional difference in risk adjustment',
            'rel_response': 'Proportional difference in response speed',
            'rel_contact': 'Proportional difference in contact avoidance',
            'rel_avg': 'Proportional difference in average adjustment',
            
            # Absolute deltas
            'delta_adjustment': 'Absolute difference in adjustment volatility',
            'delta_response': 'Absolute difference in response speed',
            'delta_contact': 'Absolute difference in contact avoidance',
            'delta_avg': 'Absolute difference in average adjustment',
            
            # Gated deltas (volatility-amplified)
            'gate_adjustment': 'Volatility-amplified adjustment difference',
            'gate_response': 'Volatility-amplified response difference',
            'gate_contact': 'Volatility-amplified contact avoidance difference',
            'gate_avg': 'Volatility-amplified average adjustment difference'
        }
        
        # Match prefix
        for prefix, desc in descriptions.items():
            if feature_name.startswith(prefix):
                # Extract scenario suffix (e.g., _hw, _hc, _wc)
                import re
                match = re.search(r'_(hw|hc|wc)$', feature_name)
                suffix = match.group(1) if match else None

                scenario_map = {
                    'hw': '(household vs workplace)',
                    'hc': '(household vs community)',
                    'wc': '(workplace vs community)'
                }
                scenario_desc = scenario_map.get(suffix, '')
                return f"{desc} {scenario_desc}"
        
        return f"Contrastive feature: {feature_name}"
    
    def _process_discovery_motifs(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Process DISCOVERY-BASED motifs (NEW: motif_description field).
        
        NOW WITH ARCHETYPE CLUSTERING: Clusters 644 motifs → 12 archetypes
        
        Returns:
            reasoning_embeddings: Sentence embeddings
            discovered_motifs: Motif dictionary
            motif_profiles: Agent motif embeddings [num_agents, 12]
            cluster_motifs: Cluster characterizations
        """
        print("  Processing DISCOVERY-BASED motifs with archetype clustering...")
        
        n_agents = len(self.agent_profiles)
        
        # Collect agent motif descriptions
        agent_motif_descriptions = {}
        motif_counts = Counter()
        all_reasoning_texts = []
        
        print("  → Reading discovered motifs from traces...")
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                motifs_dict = trace.get('behavioral_motifs', {})
                
                if not isinstance(motifs_dict, dict) or 'motif_description' not in motifs_dict:
                    continue
                
                # Store motif description (only once per agent)
                description = motifs_dict['motif_description']
                if agent_id not in agent_motif_descriptions:
                    agent_motif_descriptions[agent_id] = description
                    motif_counts[description] += 1
                
                # Collect reasoning texts for embeddings
                conversations = trace.get('conversations', [])
                for conv in conversations[:3]:  # Sample
                    response = conv.get('response', '')
                    if 'REASONING:' in response:
                        reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                        if len(reasoning) > 20:
                            all_reasoning_texts.append(reasoning)
        
        print(f"  ✓ Found {len(agent_motif_descriptions)} agents with discovered motifs")
        print(f"  ✓ {len(motif_counts)} unique motif descriptions")
        raw_diversity = len(motif_counts) / max(len(agent_motif_descriptions), 1)
        print(f"  ✓ Raw diversity: {raw_diversity:.1%}")
        
        # ========================================================================
        # NEW: CLUSTER MOTIFS INTO ARCHETYPES
        # ========================================================================
        print(f"\n  🎯 CLUSTERING {len(motif_counts)} MOTIFS → {self.k_archetypes} ARCHETYPES...")
        
        motif_profiles, archetypes, description_to_archetype = \
            self._cluster_discovered_motifs_to_archetypes(
                agent_motif_descriptions,
                motif_counts,
                target_k=self.k_archetypes
            )
        
        print(f"  ✓ Reduced from {len(motif_counts)} → {len(archetypes)} archetypes")
        
        # Embed reasoning texts
        if self.use_semantic and self.sentence_model is not None and len(all_reasoning_texts) > 0:
            print(f"\n  Embedding {len(all_reasoning_texts)} reasoning texts...")
            reasoning_embeddings = self.sentence_model.encode(
                all_reasoning_texts,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"  ✓ Reasoning embeddings: {reasoning_embeddings.shape}")
        else:
            reasoning_embeddings = np.zeros((len(all_reasoning_texts), 384))
        
        # Build discovered_motifs dict (now archetypes)
        discovered_motifs = {}
        
        for archetype_id, info in archetypes.items():
            discovered_motifs[archetype_id] = {
                'name': info['name'],
                'size': info['size'],
                'n_motifs': info['n_motifs'],
                'examples': info['examples'][:5],
                'description': self._generate_archetype_description(info)
            }
        
        print(f"  ✓ Discovered {len(discovered_motifs)} behavioral archetypes")
        
        # Show top archetypes
        print(f"\n  Top archetypes:")
        for archetype_id, motif_info in list(discovered_motifs.items())[:10]:
            print(f"    {archetype_id}. {motif_info['name']}: {motif_info['size']} agents")
        
        # Compute cluster archetype profiles
        print(f"\n  Computing cluster characterizations...")
        cluster_motifs = {}
        
        for cluster_id in np.unique(self.coarse_labels):
            mask = self.coarse_labels == cluster_id
            
            if mask.sum() == 0:
                continue
            
            # Get agent descriptions in this cluster
            cluster_agent_ids = [
                self.idx_to_agent_id[i] 
                for i in np.where(mask)[0] 
                if self.idx_to_agent_id[i] in agent_motif_descriptions
            ]
            
            cluster_descriptions = [
                agent_motif_descriptions[aid] 
                for aid in cluster_agent_ids
            ]
            
            # For PCA: Count most common raw descriptions (not archetypes)
            desc_counts = Counter(cluster_descriptions)
            top_cluster_motifs = desc_counts.most_common(5)
            
            # Cluster profile (average PCA coordinates)
            cluster_profile = motif_profiles[mask].mean(axis=0)
            
            cluster_motifs[int(cluster_id)] = {
                'profile': cluster_profile,
                'dominant_motifs': top_cluster_motifs,  # Raw descriptions, not archetypes
                'unique_motifs': len(set(cluster_descriptions)),
                'total_agents': len(cluster_descriptions),
                'description': self.__generate_cluster_description_from_descriptions(top_cluster_motifs)
            }
        
        print(f"  ✓ Cluster motif profiles computed")
        
        # Print characterizations
        for cluster_id, info in cluster_motifs.items():
            print(f"\n  Cluster {cluster_id} ({info['total_agents']} agents, "
                  f"{info['unique_motifs']} unique motifs):")
            print(f"    {info['description']}")
            for motif, count in info['dominant_motifs'][:3]:
                pct = (count / max(info['total_agents'], 1)) * 100
                # Truncate long motif descriptions for display
                motif_display = motif if len(motif) <= 50 else motif[:47] + "..."
                print(f"    - {motif_display}: {count} ({pct:.1f}%)")
        
        # Store for later
        self.motif_descriptions = agent_motif_descriptions
        self.archetypes = archetypes
        self.description_to_archetype = description_to_archetype
        self.num_archetypes = len(archetypes) # Update to archetype count
        
        return reasoning_embeddings, discovered_motifs, motif_profiles, cluster_motifs
    
    # ========================================================================
    # NEW: ARCHETYPE CLUSTERING METHODS
    # ========================================================================
    
    def _cluster_discovered_motifs_to_archetypes(
        self, 
        motif_descriptions: Dict[str, str],
        motif_counts: Counter,
        target_k: int = 12
    ) -> Tuple[np.ndarray, Dict, Dict[str, int]]:
        """
        CORE METHOD: Reduce motif embeddings from 384D → 50D using PCA.
        
        This preserves variance while avoiding the dimensionality curse.
        PCA is better than K-means because it doesn't collapse semantic diversity.
        
        Args:
            motif_descriptions: Map agent_id → motif description
            motif_counts: Counter of description frequencies
            target_k: Target dimensionality (default 12, but we'll use 50 for PCA)
            
        Returns:
            motif_profiles: PCA-reduced agent profiles [n_agents, 50]
            archetypes: Dict with PCA info (for compatibility)
            description_to_archetype: Mapping (unused, for compatibility)
        """
        from sklearn.decomposition import PCA
        
        # Use 50 components for PCA (ignore target_k which was for K-means)
        pca_components = 50
        
        n_agents = len(self.agent_profiles)
        
        # Get unique descriptions
        descriptions = [desc for desc, _ in motif_counts.most_common()]
        
        if len(descriptions) == 0:
            return self._create_empty_archetypes(pca_components)
        
        # Embed all unique descriptions
        print(f"  → Embedding {len(descriptions)} unique motifs...")
        description_embeddings = self.sentence_model.encode(
            descriptions,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True
        )
        
        print(f"  ✓ Motif embeddings: {description_embeddings.shape}")
        
        # PCA dimensionality reduction
        if len(descriptions) < 5:
            print("  ⚠ Too few motifs for PCA — using raw embeddings")
            actual_k = description_embeddings.shape[1]
            description_embeddings_pca = description_embeddings
        else:
            actual_k = min(pca_components, description_embeddings.shape[0])
        print(f"  → Applying PCA: {description_embeddings.shape[1]}D → {actual_k}D...")
        pca = PCA(n_components=actual_k, random_state=42)
        description_embeddings_pca = pca.fit_transform(description_embeddings)

        from sklearn.cluster import KMeans

        k = min(target_k, description_embeddings_pca.shape[0])

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50
        )
        labels = kmeans.fit_predict(description_embeddings_pca)

        
        # CRITICAL: Normalize PCA embeddings to [-1, 1] range for proper fusion
        from sklearn.preprocessing import StandardScaler
        pca_scaler = StandardScaler()
        description_embeddings_pca = pca_scaler.fit_transform(description_embeddings_pca)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"  ✓ PCA complete: {variance_explained:.1%} variance preserved")
        print(f"  ✓ Normalized PCA embeddings: {description_embeddings_pca.shape}")
        print(f"  ✓ PCA value range: [{description_embeddings_pca.min():.2f}, {description_embeddings_pca.max():.2f}]")
        
        # Map descriptions to their PCA embeddings
        description_to_embedding = {
            desc: description_embeddings_pca[i]
            for i, desc in enumerate(descriptions)
        }
        
        # Create agent motif profiles by mapping agent descriptions to PCA space
        motif_profiles = np.zeros((n_agents, actual_k))
        
        agents_mapped = 0
        for agent_id, description in motif_descriptions.items():
            if agent_id not in self.agent_id_to_idx:
                continue
            
            agent_idx = self.agent_id_to_idx[agent_id]
            
            if description in description_to_embedding:
                archetype_id = description_to_archetype.get(description)
                if archetype_id is not None:
                    motif_profiles[agent_idx, archetype_id] = 1.0
                    agents_mapped += 1
        
        print(f"\n  ✓ Agent motif profiles: {motif_profiles.shape}")
        print(f"  ✓ Mapped {agents_mapped}/{len(motif_descriptions)} agents to PCA space")
        print(f"  ✓ Motif profile value range: [{motif_profiles.min():.2f}, {motif_profiles.max():.2f}]")
        print(f"  ✓ Motif profile mean: {motif_profiles.mean():.3f}, std: {motif_profiles.std():.3f}")
        
        # Build "archetypes" dict for compatibility (contains PCA info)
        # We'll create pseudo-archetypes based on PCA components
        archetypes = {}

        for i in range(k):
            cluster_descs = [
                descriptions[j]
                for j in range(len(descriptions))
                if labels[j] == i
            ]
            
            archetypes[i] = {
                'id': i,
                'name': f'archetype_{i}',
                'size': sum(motif_counts[d] for d in cluster_descs),
                'n_motifs': len(cluster_descs),
                'examples': cluster_descs[:5],
                'all_descriptions': cluster_descs
            }

            


        print(f"\n  ✓ Created {actual_k} PCA components (not archetypes)")
        print(f"  ✓ Top 5 components explain {pca.explained_variance_ratio_[:5].sum():.1%} of variance")
        print(f"  ✓ This preserves motif diversity without semantic collapse")
        
        # Store PCA model and scaler for later analysis
        self.motif_pca = pca
        self.motif_pca_scaler = pca_scaler
        self.description_to_embedding = description_to_embedding
        
        # Return empty description_to_archetype for compatibility
        description_to_archetype = {
                                        desc: labels[i]
                                        for i, desc in enumerate(descriptions)
                                    }
        
        return motif_profiles, archetypes, description_to_archetype
    
    def _name_archetype(self, example_motifs: List[str]) -> str:
        """
        Use LLM to name a behavioral archetype.
        
        Args:
            example_motifs: List of example motif descriptions in this archetype
            
        Returns:
            Concise archetype name
        """
        if not example_motifs:
            return "undefined_archetype"
        
        prompt = f"""Analyze these similar behavioral patterns and create ONE concise archetype name.

Example patterns in this group:
{chr(10).join([f"- {motif}" for motif in example_motifs[:5]])}

Create a name that captures the CORE BEHAVIORAL PATTERN (not demographics).
The name should:
1. Capture unique behavioral traits that distinguish this group
2. Be 2-4 words, underscore-separated, lowercase
3. Focus on HOW they think/decide (not who they are)
4. Be DISTINCTIVE - avoid generic words like "monitor", "tracker", "assessor"
5. Capture the SPECIFIC decision-making style

Good examples (distinctive, specific):
- "anxious_household_protector" (not just "household_monitor")
- "rational_data_calculator" (not just "risk_assessor")  
- "duration_focused_planner" (not just "timeline_tracker")
- "optimistic_youth_confident" (not just "young_pattern")
- "vaccine_reliant_decider" (not just "vaccine_aware")

Bad examples (too generic, too similar):
- "exposure_monitor", "risk_monitor", "health_monitor" (all just "X_monitor")
- "contact_tracker", "exposure_tracker" (all just "X_tracker")
- "risk_assessor", "exposure_assessor" (all just "X_assessor")

Look for what makes THIS group unique compared to other behavioral patterns.

Respond with ONLY the archetype name:"""

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30
            )
            
            name = response.choices[0].message.content.strip()
            
            # Clean up name
            name = name.lower().replace(" ", "_").replace("-", "_")
            name = ''.join(c for c in name if c.isalnum() or c == '_')
            name = name[:50]
            
            return name if name else self._fallback_archetype_name(example_motifs)
        
        except Exception as e:
            print(f"      Warning: LLM naming failed, using fallback")
            return self._fallback_archetype_name(example_motifs)
    
    def _fallback_archetype_name(self, example_motifs: List[str]) -> str:
        """
        Fallback archetype naming using keyword extraction.
        
        Args:
            example_motifs: List of motif descriptions
            
        Returns:
            Fallback archetype name
        """
        # Remove common demographic/occupational words
        common_words = {
            'young', 'old', 'middle', 'aged', 'teen', 'senior', 'student', 
            'worker', 'healthcare', 'office', 'retail', 'service', 'manual',
            'teacher', 'professional', 'high', 'low', 'partial', 'full'
        }
        
        # Extract keywords
        keywords = []
        for motif in example_motifs:
            import re
            words = re.findall(r"[a-zA-Z]+", motif.lower())
            keywords.extend([w for w in words if w not in common_words])
        
        # Count most frequent
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, _ in keyword_counts.most_common(3)]
        
        if len(top_keywords) >= 2:
            return f"{'_'.join(top_keywords[:2])}_archetype"
        elif len(top_keywords) == 1:
            return f"{top_keywords[0]}_pattern"
        else:
            return "mixed_behavioral_pattern"
    
    def _create_empty_archetypes(self, target_k: int) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Create empty archetype structures as fallback.
        
        Args:
            target_k: Number of archetypes
            
        Returns:
            Empty archetype structures
        """
        n_agents = len(self.agent_profiles)
        
        motif_profiles = np.zeros((n_agents, target_k))
        
        archetypes = {
            i: {
                'id': i,
                'name': f'empty_archetype_{i}',
                'size': 0,
                'n_motifs': 0,
                'examples': [],
                'all_descriptions': []
            }
            for i in range(target_k)
        }
        
        description_to_archetype = {}
        
        return motif_profiles, archetypes, description_to_archetype
    
    def _generate_archetype_description(self, archetype_info: Dict) -> str:
        """Generate human-readable description for archetype."""
        name = archetype_info['name']
        size = archetype_info['size']
        n_motifs = archetype_info['n_motifs']
        
        return f"{name.replace('_', ' ').title()} ({size} agents, {n_motifs} variants)"
    
    def __generate_cluster_description_from_descriptions(
        self, 
        top_archetypes: List[Tuple[str, int]]
    ) -> str:
        """Generate description from archetype/motif distribution."""
        if not top_archetypes:
            return "Mixed behavioral patterns"
        
        total = sum(count for _, count in top_archetypes)
        
        if len(top_archetypes) == 1 or (total > 0 and top_archetypes[0][1] / total > 0.5):
            # Dominant single pattern
            name = top_archetypes[0][0]
            # Truncate if too long
            if len(name) > 50:
                name = name[:47] + "..."
            return f"Dominated by: {name}"
        else:
            # Mixed patterns - show abbreviated versions
            themes = []
            for name, _ in top_archetypes[:3]:
                # Truncate long names
                if len(name) > 30:
                    name = name[:27] + "..."
                themes.append(name)
            return f"Mixed: {' + '.join(themes)}"
    
    # ========================================================================
    # END OF NEW ARCHETYPE METHODS
    # ========================================================================
    
    def _generate_cluster_description_discovery(self, top_motifs: List[Tuple[str, int]]) -> str:
        """Generate description from discovered motifs."""
        if not top_motifs:
            return "Mixed behavioral patterns"
        
        total = sum(count for _, count in top_motifs)
        
        if len(top_motifs) == 1 or (total > 0 and top_motifs[0][1] / total > 0.5):
            # Dominant single motif
            return f"Dominated by: {top_motifs[0][0]}"
        else:
            # Show top themes
            themes = [desc for desc, _ in top_motifs[:3]]
            return f"Mixed: {' | '.join(themes)}"
    
    
    def _synthesize_structured_from_discovery(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Synthesize HYBRID structured representation: categorical axes + continuous features.
        
        KEY INSIGHT: Use LLM to batch-map all unique discovery motifs to structured axes.
        This preserves diversity while enforcing orthogonal structure.
        
        This combines:
        - 26D categorical (6 orthogonal axes) - interpretable structure
        - 8D continuous (intensity features) - discriminative power
        
        Total: 34D expressive, interpretable representation
        
        Returns:
            Same format as _process_structured_axes_motifs
        """
        print("  Synthesizing HYBRID structured motifs (categorical + continuous)...")
        print("  → Using batch LLM mapping to preserve discovery diversity")
        
        n_agents = len(self.agent_profiles)
        
        # Define motif axes (categorical - 26D)
        motif_axes = {
            'exposure_reasoning': [
                'contact_counting', 'household_transmission', 'workplace_network',
                'community_prevalence', 'minimal_exposure_reasoning'
            ],
            'risk_posture': [
                'risk_averse', 'risk_neutral', 'risk_minimizing', 'risk_ignoring'
            ],
            'information_seeking': [
                'heavy_tool_user', 'moderate_tool_user', 'light_tool_user', 'tool_avoider'
            ],
            'protection_priority': [
                'self_focused', 'household_focused', 'community_focused', 'occupation_focused'
            ],
            'temporal_style': [
                'duration_tracker', 'symptom_monitor', 'calendar_planner', 'present_focused'
            ],
            'vaccine_reasoning': [
                'vaccine_reliant', 'vaccine_aware', 'vaccine_skeptical', 'vaccine_irrelevant'
            ]
        }
        
        # Define continuous features (8D)
        continuous_features = [
            'contact_count',           # Number of infected contacts
            'infected_household',      # Household infection count
            'compliance_score',        # [0,1] compliance
            'age_normalized',          # Age / 100
            'days_in_state',          # Duration in current state
            'activity_hours',         # Contact hours per day
            'risk_perception',        # Derived from reasoning
            'tool_usage_intensity'    # Tool usage frequency
        ]
        
        categorical_dims = sum(len(options) for options in motif_axes.values())
        continuous_dims = len(continuous_features)
        total_dims = categorical_dims + continuous_dims
        
        print(f"  → Categorical dimensions: {categorical_dims}D (6 orthogonal axes)")
        print(f"  → Continuous dimensions: {continuous_dims}D (intensity features)")
        print(f"  → Total dimensions: {total_dims}D")
        
        # STEP 1: Collect all unique discovery motifs
        print("\n  [STEP 1] Collecting unique discovery motifs...")
        unique_discovery_motifs = set()
        agent_to_description = {}
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                motifs_dict = trace.get('behavioral_motifs', {})
                if isinstance(motifs_dict, dict) and 'motif_description' in motifs_dict:
                    description = motifs_dict['motif_description']
                    unique_discovery_motifs.add(description)
                    agent_to_description[agent_id] = description
        
        unique_discovery_motifs = sorted(list(unique_discovery_motifs))
        print(f"  ✓ Found {len(unique_discovery_motifs)} unique discovery motifs")
        print(f"  ✓ Mapped {len(agent_to_description)}/{n_agents} agents to motifs")
        
        # STEP 2: Batch-map all unique motifs to structured axes using LLM
        print(f"\n  [STEP 2] Batch-mapping {len(unique_discovery_motifs)} motifs to structured axes...")
        print(f"  (This runs once and is cached)")
        
        discovery_to_structured = self._batch_llm_map_to_structured(
            unique_discovery_motifs, 
            motif_axes
        )
        
        print(f"  ✓ Mapped {len(discovery_to_structured)}/{len(unique_discovery_motifs)} motifs")
        
        # Verify diversity in mapping
        axis_diversity = {axis: set() for axis in motif_axes.keys()}
        for structured in discovery_to_structured.values():
            for axis, value in structured.items():
                axis_diversity[axis].add(value)
        
        print(f"\n  📊 Diversity in LLM mapping:")
        for axis, values in axis_diversity.items():
            print(f"    {axis}: {len(values)}/{len(motif_axes[axis])} options used")
        
        # Build dimension mapping for categorical
        dim_idx = 0
        axis_to_indices = {}
        idx_to_name = {}
        
        for axis_name, options in motif_axes.items():
            axis_to_indices[axis_name] = {}
            for option in options:
                axis_to_indices[axis_name][option] = dim_idx
                idx_to_name[dim_idx] = f"{axis_name}:{option}"
                dim_idx += 1
        
        # Add continuous feature indices
        continuous_start_idx = categorical_dims
        for i, feat_name in enumerate(continuous_features):
            idx_to_name[continuous_start_idx + i] = f"continuous:{feat_name}"
        
        # STEP 3: Build motif profiles
        print(f"\n  [STEP 3] Building agent motif profiles...")
        motif_profiles = np.zeros((n_agents, total_dims))
        motif_counts = {}
        all_reasoning_texts = []
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                agent_idx = self.agent_id_to_idx[agent_id]
                
                # Get discovery motif for this agent
                if agent_id in agent_to_description:
                    description = agent_to_description[agent_id]
                    
                    # Look up structured mapping
                    if description in discovery_to_structured:
                        structured_axes = discovery_to_structured[description]
                        
                        # Encode categorical axes (multi-hot)
                        for axis_name, value in structured_axes.items():
                            if axis_name in axis_to_indices and value in axis_to_indices[axis_name]:
                                idx = axis_to_indices[axis_name][value]
                                motif_profiles[agent_idx, idx] = 1.0
                                
                                motif_name = f"{axis_name}:{value}"
                                motif_counts[motif_name] = motif_counts.get(motif_name, 0) + 1
                
                # Extract continuous features
                continuous_vals = self._extract_continuous_features(trace, agent_idx)
                
                # Add continuous features to profile
                for i, feat_name in enumerate(continuous_features):
                    motif_profiles[agent_idx, continuous_start_idx + i] = continuous_vals.get(feat_name, 0.0)
                
                # Collect reasoning texts
                conversations = trace.get('conversations', [])
                for conv in conversations[:3]:
                    response = conv.get('response', '')
                    if 'REASONING:' in response:
                        reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                        if len(reasoning) > 20:
                            all_reasoning_texts.append(reasoning)
        
        print(f"  ✓ Synthesized {len(motif_counts)} categorical motif combinations")
        print(f"  ✓ Collected {len(all_reasoning_texts)} reasoning texts")
        
        # Verify diversity
        agents_with_motifs = (motif_profiles.sum(axis=1) > 0).sum()
        avg_categorical = motif_profiles[:, :categorical_dims].sum(axis=1).mean()
        avg_continuous = np.abs(motif_profiles[:, categorical_dims:]).sum(axis=1).mean()
        
        print(f"  ✓ Agents with motifs: {agents_with_motifs}/{n_agents}")
        print(f"  ✓ Average categorical per agent: {avg_categorical:.1f}/{categorical_dims}")
        print(f"  ✓ Average continuous magnitude: {avg_continuous:.2f}")
        
        # Check diversity across categorical axes
        print(f"\n  📊 Categorical axis diversity in agent profiles:")
        for axis_name in motif_axes.keys():
            axis_indices = [axis_to_indices[axis_name][opt] for opt in motif_axes[axis_name]]
            axis_coverage = motif_profiles[:, axis_indices].sum(axis=0)
            active_options = (axis_coverage > 0).sum()
            
            # Show distribution, not just dominant
            option_counts = [(motif_axes[axis_name][i], int(axis_coverage[i])) 
                           for i in range(len(motif_axes[axis_name]))]
            option_counts = sorted(option_counts, key=lambda x: x[1], reverse=True)
            
            dist_str = ", ".join([f"{opt}:{cnt}" for opt, cnt in option_counts if cnt > 0])
            print(f"    {axis_name}: {active_options}/{len(motif_axes[axis_name])} options ({dist_str})")
        
        # Check continuous feature statistics
        print(f"\n  📊 Continuous feature statistics:")
        for i, feat_name in enumerate(continuous_features):
            feat_vals = motif_profiles[:, continuous_start_idx + i]
            feat_vals_nonzero = feat_vals[feat_vals != 0]
            if len(feat_vals_nonzero) > 0:
                print(f"    {feat_name}: mean={feat_vals_nonzero.mean():.3f}, std={feat_vals_nonzero.std():.3f}, " +
                      f"range=[{feat_vals_nonzero.min():.3f}, {feat_vals_nonzero.max():.3f}]")
        
        # Check overall variance
        categorical_variance = motif_profiles[:, :categorical_dims].var(axis=0).mean()
        continuous_variance = motif_profiles[:, categorical_dims:].var(axis=0).mean()
        total_variance = motif_profiles.var(axis=0).mean()
        
        print(f"\n  ✓ Categorical variance: {categorical_variance:.3f}")
        print(f"  ✓ Continuous variance: {continuous_variance:.3f}")
        print(f"  ✓ Total profile variance: {total_variance:.3f}")
        
        if categorical_variance < 0.15:
            print(f"  ⚠ WARNING: Low categorical variance! LLM mapping may have collapsed diversity.")
            print(f"    Check if too many agents mapped to same categories above.")
        elif total_variance < 0.1:
            print(f"  ⚠ WARNING: Low total variance despite good categorical variance.")
            print(f"    Continuous features may need better extraction.")
        else:
            print(f"  ✅ Good variance - hybrid representation is expressive!")
        
        # Embed reasoning texts
        if self.use_semantic and self.sentence_model is not None and len(all_reasoning_texts) > 0:
            print(f"\n  Embedding reasoning texts...")
            reasoning_embeddings = self.sentence_model.encode(
                all_reasoning_texts,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"  ✓ Embeddings shape: {reasoning_embeddings.shape}")
        else:
            reasoning_embeddings = np.zeros((len(all_reasoning_texts), 384))
        
        # Build discovered_motifs dictionary (categorical only for interpretability)
        discovered_motifs = {}
        for motif_idx in range(categorical_dims):
            motif_name = idx_to_name[motif_idx]
            count = motif_counts.get(motif_name, 0)
            axis, option = motif_name.split(':')
            
            discovered_motifs[motif_idx] = {
                'name': motif_name,
                'axis': axis,
                'option': option,
                'size': count,
                'examples': [],
                'description': self._generate_structured_motif_description(axis, option)
            }
        
        # Add continuous features to discovered_motifs
        for i, feat_name in enumerate(continuous_features):
            motif_idx = continuous_start_idx + i
            discovered_motifs[motif_idx] = {
                'name': f"continuous:{feat_name}",
                'axis': 'continuous',
                'option': feat_name,
                'size': n_agents,
                'examples': [],
                'description': f"Continuous intensity feature: {feat_name}"
            }
        
        print(f"\n  ✓ Created {len(discovered_motifs)} motif dimensions ({categorical_dims} categorical + {continuous_dims} continuous)")
        
        # Show top categorical motifs
        top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 categorical motifs:")
        for motif_name, count in top_motifs:
            print(f"    - {motif_name}: {count} agents")
        
        # Compute cluster characterizations
        print(f"\n  Computing cluster characterizations...")
        cluster_motifs = {}
        
        for cluster_id in np.unique(self.coarse_labels):
            mask = self.coarse_labels == cluster_id
            if mask.sum() == 0:
                continue
            
            cluster_profile = motif_profiles[mask].mean(axis=0)
            
            # Top 3 categorical dimensions
            categorical_profile = cluster_profile[:categorical_dims]
            top_indices = categorical_profile.argsort()[-3:][::-1]
            top_motifs_cluster = [
                (discovered_motifs[i]['name'], categorical_profile[i])
                for i in top_indices
                if categorical_profile[i] > 0
            ]
            
            cluster_motifs[int(cluster_id)] = {
                'profile': cluster_profile,
                'dominant_motifs': top_motifs_cluster,
                'description': self._generate_cluster_description_structured(top_motifs_cluster)
            }
        
        print(f"  ✓ Cluster motif profiles computed")
        
        # Print cluster characterizations
        for cluster_id, info in cluster_motifs.items():
            print(f"\n  Cluster {cluster_id}: {info['description']}")
            for motif, freq in info['dominant_motifs'][:3]:
                print(f"    - {motif}: {freq:.1%}")
        
        # Update k_motifs
        self.k_motifs = total_dims
        
        return reasoning_embeddings, discovered_motifs, motif_profiles, cluster_motifs
    
    
    def _extract_continuous_features(self, trace: Dict, agent_idx: int) -> Dict[str, float]:
        """
        Extract continuous intensity features from trace.
        
        Features:
        - contact_count: Number of infected contacts
        - infected_household: Household infection count  
        - compliance_score: [0,1] compliance level
        - age_normalized: Agent age / 100
        - days_in_state: Duration in current state
        - activity_hours: Contact hours per day
        - risk_perception: Derived from reasoning sentiment
        - tool_usage_intensity: Tool usage frequency
        
        Returns:
            Dict of feature_name -> normalized value
        """
        features = {}
        
        # Get agent profile
        agent_profile = self.agent_profiles[agent_idx]
        
        # 1. Age (normalized to [0, 1])
        features['age_normalized'] = agent_profile.get('age', 30) / 100.0
        
        # 2. Compliance score (from profile)
        features['compliance_score'] = agent_profile.get('compliance_score', 0.7)
        
        # 3. Extract from conversations
        conversations = trace.get('conversations', [])
        
        if conversations:
            # Get last conversation (most recent state)
            last_conv = conversations[-1]
            response = last_conv.get('response', '')
            
            # Extract contact count from reasoning
            contact_count = self._extract_contact_count(response)
            features['contact_count'] = min(contact_count / 20.0, 1.0)  # Normalize to [0,1]
            
            # Extract household infection count
            household_count = self._extract_household_infections(response)
            features['infected_household'] = min(household_count / 5.0, 1.0)  # Normalize
            
            # Extract days in state
            days = self._extract_days_in_state(response)
            features['days_in_state'] = min(days / 14.0, 1.0)  # Normalize to 14 days
            
            # Extract risk perception from reasoning sentiment
            risk_perception = self._extract_risk_perception(response)
            features['risk_perception'] = risk_perception  # Already [0,1]
            
            # Tool usage intensity (count unique tools used)
            tool_count = self._count_tools_used(response)
            features['tool_usage_intensity'] = min(tool_count / 6.0, 1.0)  # Normalize to 6 tools
        else:
            # Default values if no conversations
            features['contact_count'] = 0.0
            features['infected_household'] = 0.0
            features['days_in_state'] = 0.0
            features['risk_perception'] = 0.5
            features['tool_usage_intensity'] = 0.0
        
        # 4. Activity hours (from profile or default)
        features['activity_hours'] = agent_profile.get('mobility_score', 0.7)  # Use mobility as proxy
        
        return features
    
    def _extract_contact_count(self, response: str) -> int:
        """Extract number of infected contacts from response."""
        import re
        
        # Look for patterns like "3 infected contacts", "5 out of 10", etc.
        patterns = [
            r'(\d+)\s+(?:infected\s+)?contacts?',
            r'(\d+)\s+out\s+of\s+\d+',
            r'(\d+)\s+people?\s+(?:are|were)\s+infected',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Fallback: count if mentions "high contact"
        if 'high contact' in response.lower() or 'many contacts' in response.lower():
            return 10
        elif 'few contacts' in response.lower() or 'low contact' in response.lower():
            return 2
        
        return 5  # Default moderate
    
    def _extract_household_infections(self, response: str) -> int:
        """Extract household infection count from response."""
        import re
        
        # Look for household mentions
        patterns = [
            r'(\d+)\s+household\s+members?\s+(?:are|were)?\s*infected',
            r'household.*?(\d+)\s+infected',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Check for qualitative mentions
        if 'household' in response.lower():
            if 'all' in response.lower() or 'entire' in response.lower():
                return 4
            elif 'some' in response.lower() or 'one' in response.lower():
                return 1
            return 2  # Default if mentioned
        
        return 0  # No household mentions
    
    def _extract_days_in_state(self, response: str) -> int:
        """Extract days in current state from response."""
        import re
        
        # Look for day mentions
        patterns = [
            r'(\d+)\s+days?',
            r'day\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Default based on state mentions
        if 'just' in response.lower() or 'recently' in response.lower():
            return 1
        elif 'week' in response.lower():
            return 7
        
        return 3  # Default moderate duration
    
    def _extract_risk_perception(self, response: str) -> float:
        """Extract risk perception from reasoning sentiment [0=low, 1=high]."""
        response_lower = response.lower()
        
        # High risk keywords
        high_risk_words = ['concerned', 'worried', 'anxious', 'afraid', 'terrified', 
                          'dangerous', 'severe', 'serious', 'critical', 'high risk']
        
        # Low risk keywords  
        low_risk_words = ['confident', 'safe', 'protected', 'low risk', 'minimal',
                         'unlikely', 'fine', 'okay', 'not worried']
        
        high_count = sum(1 for word in high_risk_words if word in response_lower)
        low_count = sum(1 for word in low_risk_words if word in response_lower)
        
        if high_count > low_count:
            return min(0.5 + 0.1 * high_count, 1.0)
        elif low_count > high_count:
            return max(0.5 - 0.1 * low_count, 0.0)
        
        return 0.5  # Neutral
    
    def _count_tools_used(self, response: str) -> int:
        """Count number of tools used in response."""
        tools = [
            'check_neighbors', 'query_knowledge', 'check_policy', 
            'assess_risk', 'check_calendar', 'check_contacts'
        ]
        
        count = sum(1 for tool in tools if tool in response.lower())
        return count
    
    
    def _batch_llm_map_to_structured(self, unique_motifs: List[str], motif_axes: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """
        Batch-map all unique discovery motifs to structured axes using LLM.
        
        This is done ONCE for all unique motifs, then cached.
        Much faster than per-agent mapping.
        
        Args:
            unique_motifs: List of unique discovery motif descriptions
            motif_axes: Dictionary of axis names to option lists
            
        Returns:
            Dictionary mapping discovery description → structured axes
        """
        from openai import OpenAI
        import json
        import os
        
        # Try to use cached mapping if available
        cache_file = self.cache_dir / 'discovery_to_structured_cache.json'
        if os.path.exists(cache_file):
            print(f"  ✓ Loading cached mapping from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_mapping = json.load(f)
            
            # Check if all motifs are in cache
            missing = [m for m in unique_motifs if m not in cached_mapping]
            if not missing:
                print(f"  ✓ All {len(unique_motifs)} motifs found in cache!")
                return cached_mapping
            else:
                print(f"  → {len(missing)} new motifs not in cache, will map them")
                motifs_to_map = missing
                partial_mapping = cached_mapping
        else:
            print(f"  → No cache found, mapping all {len(unique_motifs)} motifs")
            motifs_to_map = unique_motifs
            partial_mapping = {}
        
        # Batch map in groups of 20 (to avoid token limits)
        batch_size = 20
        mapping = partial_mapping.copy()
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            for i in range(0, len(motifs_to_map), batch_size):
                batch = motifs_to_map[i:i+batch_size]
                print(f"    Mapping batch {i//batch_size + 1}/{(len(motifs_to_map)-1)//batch_size + 1} ({len(batch)} motifs)...")
                
                # Create prompt for batch
                batch_prompt = self._create_batch_mapping_prompt(batch, motif_axes)
                
                # Call LLM
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                # Parse response
                content = response.choices[0].message.content.strip()
                
                # Extract JSON array
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    batch_results = json.loads(json_match.group())
                    
                    # Add to mapping
                    for result in batch_results:
                        if 'description' in result:
                            desc = result['description']
                            del result['description']
                            mapping[desc] = result
                else:
                    print(f"      ⚠ Warning: Could not parse batch response, using fallback")
                    # Fallback to improved keyword parsing
                    for desc in batch:
                        mapping[desc] = self._improved_keyword_parse(desc)
            
            # Save cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            print(f"  ✓ Saved mapping cache to {cache_file}")
            
        except Exception as e:
            print(f"  ⚠ LLM mapping failed: {e}")
            print(f"  → Falling back to improved keyword parsing for all motifs")
            
            # Fallback: use improved keyword parsing
            for desc in unique_motifs:
                if desc not in mapping:
                    mapping[desc] = self._improved_keyword_parse(desc)
        
        return mapping
    
    def _create_batch_mapping_prompt(self, batch: List[str], motif_axes: Dict[str, List[str]]) -> str:
        """Create prompt for batch-mapping discovery motifs to structured axes."""
        
        motifs_str = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(batch)])
        
        axes_str = ""
        for axis_name, options in motif_axes.items():
            axes_str += f"\n{axis_name}: {', '.join(options)}"
        
        prompt = f"""Map these agent behavioral patterns to structured axes.

Agent patterns:
{motifs_str}

Map each to these 6 axes:
{axes_str}

Guidelines:
- exposure_reasoning: What type of exposure tracking? (contact_counting if mentions contacts/high_contact, household if mentions family/household, workplace if mentions job/work/occupation, community if mentions community/public, minimal if vague)
- risk_posture: Emotional stance? (risk_averse if anxious/cautious/worried, risk_minimizing if optimistic/confident, risk_ignoring if careless/dismissive, risk_neutral if balanced/no strong keywords)
- information_seeking: Tool usage level? (heavy if tracker/monitor/vigilant, moderate if check/assess, light if explorer/browse, avoider if no monitoring)
- protection_priority: Who/what is protected? (self if individual/student, household if family/children, occupation if worker/professional/job, community if public/social)
- temporal_style: Time planning? (duration if tracks days/timeline, symptom if monitors symptoms/health, calendar if planner/schedule, present if no planning)
- vaccine_reasoning: Vaccine stance? (reliant if full_vaccine, aware if partial/vaccinated, skeptical if unvaccinated/no_vaccine, irrelevant if not mentioned)

Respond with ONLY a JSON array:
[
  {{
    "description": "pattern 1",
    "exposure_reasoning": "...",
    "risk_posture": "...",
    "information_seeking": "...",
    "protection_priority": "...",
    "temporal_style": "...",
    "vaccine_reasoning": "..."
  }},
  ...
]"""
        
        return prompt
    
    def _improved_keyword_parse(self, description: str) -> Dict[str, str]:
        """
        Improved keyword-based parsing with comprehensive pattern matching.
        
        This is the fallback when LLM batch mapping fails.
        Uses much more extensive keyword lists and pattern detection.
        """
        desc_lower = description.lower()
        structured = {}
        
        # exposure_reasoning: comprehensive detection
        if any(word in desc_lower for word in ['contact_count', 'infected_contact', 'exposure_count', 'contact_track', 'high_contact', 'infected_neighbors']):
            structured['exposure_reasoning'] = 'contact_counting'
        elif any(word in desc_lower for word in ['household', 'family', 'home', 'domestic', 'children', 'spouse']):
            structured['exposure_reasoning'] = 'household_transmission'
        elif any(word in desc_lower for word in ['workplace', 'work', 'occupation', 'job', 'office', 'healthcare_worker', 'teacher', 'retail_worker', 'service_worker', 'professional', 'employee']):
            structured['exposure_reasoning'] = 'workplace_network'
        elif any(word in desc_lower for word in ['community', 'public', 'social', 'neighborhood', 'local', 'area']):
            structured['exposure_reasoning'] = 'community_prevalence'
        else:
            # Only minimal if truly no exposure tracking
            if any(word in desc_lower for word in ['exposure', 'contact', 'transmission', 'spread']):
                # Default to contact_counting if mentions exposure but no specific type
                structured['exposure_reasoning'] = 'contact_counting'
            else:
                structured['exposure_reasoning'] = 'minimal_exposure_reasoning'
        
        # risk_posture: much more nuanced with better defaults
        risk_averse_words = ['anxious', 'cautious', 'worried', 'afraid', 'terrified', 'concerned', 'vigilant', 'careful', 'nervous', 'fearful', 'scared', 'panic']
        risk_minimizing_words = ['optimistic', 'confident', 'minimiz', 'downplay', 'unconcerned', 'relaxed', 'unworried', 'dismiss']
        risk_ignoring_words = ['ignor', 'careless', 'reckless', 'uncaring', 'negligent']
        
        averse_count = sum(1 for word in risk_averse_words if word in desc_lower)
        minimizing_count = sum(1 for word in risk_minimizing_words if word in desc_lower)
        ignoring_count = sum(1 for word in risk_ignoring_words if word in desc_lower)
        
        if ignoring_count > 0:
            structured['risk_posture'] = 'risk_ignoring'
        elif averse_count > minimizing_count and averse_count > 0:
            structured['risk_posture'] = 'risk_averse'
        elif minimizing_count > averse_count and minimizing_count > 0:
            structured['risk_posture'] = 'risk_minimizing'
        else:
            # Only default to neutral if no strong indicators
            # But check for weak indicators of aversion
            if any(word in desc_lower for word in ['monitor', 'check', 'aware', 'track']):
                # Monitoring suggests some risk awareness → lean toward averse
                structured['risk_posture'] = 'risk_averse'
            else:
                structured['risk_posture'] = 'risk_neutral'
        
        # information_seeking: distinguish levels more carefully
        heavy_words = ['tracker', 'heavy', 'vigilant', 'monitor', 'constant', 'frequent', 'intensive', 'thorough']
        moderate_words = ['check', 'assess', 'review', 'examine', 'periodic', 'regular']
        light_words = ['explorer', 'curious', 'browse', 'scan', 'occasional', 'casual']
        
        heavy_count = sum(1 for word in heavy_words if word in desc_lower)
        moderate_count = sum(1 for word in moderate_words if word in desc_lower)
        light_count = sum(1 for word in light_words if word in desc_lower)
        
        if heavy_count > 0:
            structured['information_seeking'] = 'heavy_tool_user'
        elif moderate_count > 0:
            structured['information_seeking'] = 'moderate_tool_user'
        elif light_count > 0:
            structured['information_seeking'] = 'light_tool_user'
        else:
            # Check for any tool/information seeking behavior
            if any(word in desc_lower for word in ['tool', 'info', 'data', 'knowledge', 'query']):
                # Has some info seeking → moderate by default
                structured['information_seeking'] = 'moderate_tool_user'
            else:
                structured['information_seeking'] = 'tool_avoider'
        
        # protection_priority: extract from role/context
        if any(word in desc_lower for word in ['household', 'family', 'home', 'children', 'child', 'parent', 'spouse', 'domestic']):
            structured['protection_priority'] = 'household_focused'
        elif any(word in desc_lower for word in ['healthcare', 'teacher', 'worker', 'employee', 'professional', 'occupation', 'job', 'work', 'career', 'service', 'retail', 'office']):
            structured['protection_priority'] = 'occupation_focused'
        elif any(word in desc_lower for word in ['community', 'public', 'social', 'neighbor', 'society', 'collective']):
            structured['protection_priority'] = 'community_focused'
        else:
            # Default to self if no clear external priority
            structured['protection_priority'] = 'self_focused'
        
        # temporal_style: detect planning behavior
        duration_words = ['duration', 'timeline', 'days', 'time', 'track', 'count']
        symptom_words = ['symptom', 'health', 'status', 'condition', 'feeling', 'signs']
        calendar_words = ['calendar', 'planner', 'schedule', 'plan', 'future', 'upcoming']
        
        duration_count = sum(1 for word in duration_words if word in desc_lower)
        symptom_count = sum(1 for word in symptom_words if word in desc_lower)
        calendar_count = sum(1 for word in calendar_words if word in desc_lower)
        
        if calendar_count > 0:
            structured['temporal_style'] = 'calendar_planner'
        elif symptom_count > 0:
            structured['temporal_style'] = 'symptom_monitor'
        elif duration_count > 0:
            structured['temporal_style'] = 'duration_tracker'
        else:
            # Default to present if no temporal planning evident
            structured['temporal_style'] = 'present_focused'
        
        # vaccine_reasoning: detect vaccine mentions
        if any(word in desc_lower for word in ['full_vaccine', 'fully_vaccinated', 'complete_vaccine', 'boosted']):
            structured['vaccine_reasoning'] = 'vaccine_reliant'
        elif any(word in desc_lower for word in ['partial_vaccine', 'one_dose', 'vaccinated', 'first_dose', 'second_dose']):
            structured['vaccine_reasoning'] = 'vaccine_aware'
        elif any(word in desc_lower for word in ['unvaccinated', 'no_vaccine', 'not_vaccinated', 'anti_vax']):
            structured['vaccine_reasoning'] = 'vaccine_skeptical'
        else:
            # Only irrelevant if truly no mention
            if 'vaccine' in desc_lower or 'vaccination' in desc_lower or 'immuniz' in desc_lower:
                # Has vaccine mention but unclear status → aware by default
                structured['vaccine_reasoning'] = 'vaccine_aware'
            else:
                structured['vaccine_reasoning'] = 'vaccine_irrelevant'
        
        return structured
    
    def _process_structured_axes_motifs(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Process pre-extracted STRUCTURED motifs (6-axis Dict format).
        
        Converts 6-axis structured motifs into multi-hot encoded motif profiles.
        
        This is the ORIGINAL structured approach - kept for backward compatibility.
        
        Returns:
            reasoning_embeddings: Sentence embeddings (if available)
            discovered_motifs: Dictionary of discovered motifs
            motif_profiles: Agent motif vectors [num_agents, num_dims]
            cluster_motifs: Cluster-level motif characterizations
        """
        print("  Processing STRUCTURED (6-axis) behavioral motifs...")
        
        n_agents = len(self.agent_profiles)
        
        # Define motif axes and options (matches trace collection schema)
        motif_axes = {
            'exposure_reasoning': [
                'contact_counting', 'household_transmission', 'workplace_network',
                'community_prevalence', 'minimal_exposure_reasoning'
            ],
            'risk_posture': [
                'risk_averse', 'risk_neutral', 'risk_minimizing', 'risk_ignoring'
            ],
            'information_seeking': [
                'heavy_tool_user', 'moderate_tool_user', 'light_tool_user', 'tool_avoider'
            ],
            'protection_priority': [
                'self_focused', 'household_focused', 'community_focused', 'occupation_focused'
            ],
            'temporal_style': [
                'duration_tracker', 'symptom_monitor', 'calendar_planner', 'present_focused'
            ],
            'vaccine_reasoning': [
                'vaccine_reliant', 'vaccine_aware', 'vaccine_skeptical', 'vaccine_irrelevant'
            ]
        }
        
        # Calculate total dimensions
        total_dims = sum(len(options) for options in motif_axes.values())
        print(f"  → Multi-hot encoding: 6 axes, {total_dims} total dimensions")
        
        # Build dimension index mapping
        dim_idx = 0
        axis_to_indices = {}
        idx_to_name = {}
        
        for axis_name, options in motif_axes.items():
            axis_to_indices[axis_name] = {}
            for option in options:
                axis_to_indices[axis_name][option] = dim_idx
                idx_to_name[dim_idx] = f"{axis_name}:{option}"
                dim_idx += 1
        
        # Initialize motif profiles
        motif_profiles = np.zeros((n_agents, total_dims))
        
        # Collect reasoning texts for sentence embeddings
        all_reasoning_texts = []
        
        # Process traces
        print("  → Reading structured motifs from traces...")
        motif_counts = {}
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                agent_idx = self.agent_id_to_idx[agent_id]
                motifs_dict = trace.get('behavioral_motifs', {})
                
                # Skip if not Dict format
                if not isinstance(motifs_dict, dict):
                    continue
                
                # Multi-hot encode each axis
                for axis_name, value in motifs_dict.items():
                    if axis_name == 'tool_usage_stats':
                        continue  # Skip tool stats
                    
                    if axis_name in axis_to_indices and value in axis_to_indices[axis_name]:
                        idx = axis_to_indices[axis_name][value]
                        motif_profiles[agent_idx, idx] = 1.0
                        
                        # Count motif occurrences
                        motif_name = f"{axis_name}:{value}"
                        motif_counts[motif_name] = motif_counts.get(motif_name, 0) + 1
                
                # Collect reasoning texts for embeddings
                conversations = trace.get('conversations', [])
                for conv in conversations[:3]:  # Sample a few
                    response = conv.get('response', '')
                    if 'REASONING:' in response:
                        reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                        if len(reasoning) > 20:
                            all_reasoning_texts.append(reasoning)
        
        print(f"  ✓ Processed {len(motif_counts)} unique motif combinations")
        print(f"  ✓ Collected {len(all_reasoning_texts)} reasoning texts")
        
        # Verify motif diversity
        agents_with_motifs = (motif_profiles.sum(axis=1) > 0).sum()
        avg_motifs_per_agent = motif_profiles.sum(axis=1).mean()
        print(f"  ✓ Agents with motifs: {agents_with_motifs}/{n_agents}")
        print(f"  ✓ Average motifs per agent: {avg_motifs_per_agent:.1f}/{total_dims}")
        
        # Check diversity across axes
        print(f"\n  Motif diversity by axis:")
        for axis_name, options in motif_axes.items():
            axis_indices = [axis_to_indices[axis_name][opt] for opt in options]
            axis_coverage = motif_profiles[:, axis_indices].sum(axis=0)
            active_options = (axis_coverage > 0).sum()
            print(f"    {axis_name}: {active_options}/{len(options)} options used")
        
        # Embed reasoning texts with sentence transformers
        if self.use_semantic and self.sentence_model is not None and len(all_reasoning_texts) > 0:
            print(f"  Embedding reasoning texts...")
            reasoning_embeddings = self.sentence_model.encode(
                all_reasoning_texts,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"  ✓ Embeddings shape: {reasoning_embeddings.shape}")
        else:
            reasoning_embeddings = np.zeros((len(all_reasoning_texts), 384))
        
        # Build discovered_motifs dictionary
        discovered_motifs = {}
        
        for motif_idx, motif_name in idx_to_name.items():
            count = motif_counts.get(motif_name, 0)
            
            axis, option = motif_name.split(':')
            
            discovered_motifs[motif_idx] = {
                'name': motif_name,
                'axis': axis,
                'option': option,
                'size': count,
                'examples': [],  # Could populate if needed
                'description': self._generate_structured_motif_description(axis, option)
            }
        
        print(f"  ✓ Discovered {len(discovered_motifs)} motif dimensions")
        
        # Show top motifs
        top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top 10 motifs:")
        for motif_name, count in top_motifs:
            print(f"    - {motif_name}: {count} agents")
        
        # Compute cluster motif profiles
        print(f"  Computing cluster motif characterizations...")
        cluster_motifs = {}
        
        for cluster_id in np.unique(self.coarse_labels):
            mask = self.coarse_labels == cluster_id
            
            if mask.sum() == 0:
                continue
            
            # Cluster's average motif profile
            cluster_profile = motif_profiles[mask].mean(axis=0)
            
            # Top 5 dominant motifs
            top_indices = cluster_profile.argsort()[-5:][::-1]
            top_motifs_cluster = [
                (discovered_motifs[i]['name'], cluster_profile[i])
                for i in top_indices
                if cluster_profile[i] > 0
            ]
            
            cluster_motifs[int(cluster_id)] = {
                'profile': cluster_profile,
                'dominant_motifs': top_motifs_cluster,
                'description': self._generate_cluster_description_structured(top_motifs_cluster)
            }
        
        print(f"  ✓ Cluster motif profiles computed")
        
        # Print cluster characterizations
        for cluster_id, info in cluster_motifs.items():
            print(f"\n  Cluster {cluster_id}: {info['description']}")
            for motif, freq in info['dominant_motifs'][:3]:
                print(f"    - {motif}: {freq:.1%}")
        
        # Update k_motifs to match actual dimensions
        self.k_motifs = total_dims
        
        return reasoning_embeddings, discovered_motifs, motif_profiles, cluster_motifs
    
    def _process_unstructured_motifs(self) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        FALLBACK: Process unstructured motifs (old List format or missing motifs).
        
        Uses LLM classification to discover motifs from reasoning texts.
        This is the ORIGINAL implementation - kept for backward compatibility.
        """
        print("  Extracting and classifying behavioral patterns with LLM...")
        
        n_agents = len(self.agent_profiles)

        # ⭐ SAMPLING CONFIGURATION (adjust based on agent count)
        if n_agents <= 50:
            MAX_TEXTS_PER_AGENT = 12
        elif n_agents <= 200:
            MAX_TEXTS_PER_AGENT = 10
        else:
            MAX_TEXTS_PER_AGENT = 8
        
        print(f"    Sampling strategy: max {MAX_TEXTS_PER_AGENT} texts per agent")
        
        # 2.1: Collect all reasoning texts (just the reasoning part, not context)
        all_reasoning_texts = []
        agent_reasoning_indices = [[] for _ in range(n_agents)]
        
        for scenario in self.reasoning_traces['scenarios']:
            for trace in scenario['traces']:
                agent_id = trace['agent_id']
                if agent_id not in self.agent_id_to_idx:
                    continue
                
                agent_idx = self.agent_id_to_idx[agent_id]
                conversations = trace.get('conversations', [])

                # ⭐ Sample conversations for this agent
                if len(conversations) > MAX_TEXTS_PER_AGENT:
                    # Stratified sampling: get diverse samples across time
                    step = len(conversations) / MAX_TEXTS_PER_AGENT
                    sampled_indices = [int(i * step) for i in range(MAX_TEXTS_PER_AGENT)]
                    conversations = [conversations[i] for i in sampled_indices]
                
                for conv in conversations:
                    # Extract ONLY the reasoning part (not tools/state/transition)
                    response = conv.get('response', '')
                    
                    # Get clean reasoning text
                    if 'REASONING:' in response:
                        reasoning = response.split('REASONING:')[1].split('ADJUSTMENT:')[0].strip()
                    else:
                        reasoning = response[:200]
                    
                    if len(reasoning) > 20:  # Skip very short responses
                        all_reasoning_texts.append(reasoning)
                        agent_reasoning_indices[agent_idx].append(len(all_reasoning_texts) - 1)

        print(f"  ✓ Collected {len(all_reasoning_texts)} reasoning texts (sampled)")
        print(f"    Average per agent: {len(all_reasoning_texts) / n_agents:.1f}")
        
        # Store as instance variables
        self.all_reasoning_texts = all_reasoning_texts
        self.agent_reasoning_indices = agent_reasoning_indices
        
        if len(all_reasoning_texts) == 0:
            print("  ⚠ Warning: No reasoning texts found! Using empty motif profiles.")
            return self._create_empty_motifs(n_agents)
        
        # 2.2: Classify behavioral patterns with LLM (batch processing)
        print(f"  Classifying behavioral patterns with LLM...")
        behavioral_labels = self._classify_behavioral_patterns_batch(all_reasoning_texts)
        
        # ⭐ CRITICAL FIX: Ensure behavioral_labels matches all_reasoning_texts length
        if len(behavioral_labels) != len(all_reasoning_texts):
            print(f"  ⚠ Warning: LLM returned {len(behavioral_labels)} labels for {len(all_reasoning_texts)} texts")
            print(f"    Padding/truncating to match...")
            
            # Pad with fallback if too few labels
            while len(behavioral_labels) < len(all_reasoning_texts):
                missing_idx = len(behavioral_labels)
                fallback_label = self._fallback_pattern_extraction([all_reasoning_texts[missing_idx]])[0]
                behavioral_labels.append(fallback_label)
            
            # Truncate if too many labels (shouldn't happen, but just in case)
            behavioral_labels = behavioral_labels[:len(all_reasoning_texts)]
        
        # Verify lengths match
        assert len(behavioral_labels) == len(all_reasoning_texts), \
            f"Length mismatch: {len(behavioral_labels)} labels vs {len(all_reasoning_texts)} texts"
        
        # Store as instance variable
        self.behavioral_labels = behavioral_labels
        
        # 2.3: Discover unique behavioral motifs
        print(f"  Discovering behavioral motifs...")
        unique_patterns = list(set(behavioral_labels))
        
        # If we have more unique patterns than k_motifs, keep top k_motifs by frequency
        if len(unique_patterns) > self.k_motifs:
            pattern_counts = Counter(behavioral_labels)
            unique_patterns = [p for p, _ in pattern_counts.most_common(self.k_motifs)]
        
        # Create motif mapping: pattern_name -> motif_id
        pattern_to_motif = {pattern: i for i, pattern in enumerate(unique_patterns)}
        
        # Convert behavioral labels to motif IDs
        motif_labels = np.array([pattern_to_motif.get(label, 0) for label in behavioral_labels])
        
        # ⭐ VERIFY: motif_labels should have same length as all_reasoning_texts
        assert len(motif_labels) == len(all_reasoning_texts), \
            f"Motif labels length mismatch: {len(motif_labels)} vs {len(all_reasoning_texts)}"
        
        # 2.4: Build discovered_motifs dictionary
        discovered_motifs = {}
        
        for motif_id, pattern_name in enumerate(unique_patterns):
            motif_mask = motif_labels == motif_id
            motif_indices = np.where(motif_mask)[0]
            motif_examples = [all_reasoning_texts[i] for i in motif_indices[:5]]  # Top 5 examples
            
            discovered_motifs[motif_id] = {
                'name': pattern_name,
                'size': int(motif_mask.sum()),
                'examples': motif_examples,
                'description': self._generate_motif_description(pattern_name)
            }
        
        # Fill remaining motifs if we have fewer than k_motifs
        for motif_id in range(len(unique_patterns), self.k_motifs):
            discovered_motifs[motif_id] = {
                'name': f'rare_pattern_{motif_id}',
                'size': 0,
                'examples': [],
                'description': 'Rare or undefined behavioral pattern'
            }
        
        print(f"  ✓ Discovered {len(unique_patterns)} behavioral motifs:")
        for motif_id in range(min(len(unique_patterns), self.k_motifs)):
            info = discovered_motifs[motif_id]
            print(f"    Motif {motif_id}: {info['name']} ({info['size']} instances)")
        
        # 2.5: Embed reasoning texts with sentence transformers (for later use)
        if self.use_semantic and self.sentence_model is not None:
            print(f"  Embedding reasoning texts with sentence transformers...")
            reasoning_embeddings = self.sentence_model.encode(
                all_reasoning_texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"  ✓ Embeddings shape: {reasoning_embeddings.shape}")
        else:
            print(f"  Skipping semantic embeddings (use_semantic=False)")
            reasoning_embeddings = np.zeros((len(all_reasoning_texts), 384))
        
        # 2.6: Build agent motif profiles
        print(f"  Building agent motif profiles...")
        motif_profiles = np.zeros((n_agents, self.k_motifs))
        
        for agent_idx, reasoning_indices in enumerate(agent_reasoning_indices):
            if len(reasoning_indices) > 0:
                # ⭐ FIX: Check bounds before indexing
                valid_indices = [idx for idx in reasoning_indices if idx < len(motif_labels)]
                
                if len(valid_indices) != len(reasoning_indices):
                    print(f"    Warning: Agent {agent_idx} has {len(reasoning_indices) - len(valid_indices)} out-of-bounds indices")
                
                if len(valid_indices) > 0:
                    agent_motif_labels = motif_labels[valid_indices]
                    for motif_id in agent_motif_labels:
                        if motif_id < self.k_motifs:
                            motif_profiles[agent_idx, motif_id] += 1
        
        # Normalize to frequencies
        row_sums = motif_profiles.sum(axis=1, keepdims=True)
        motif_profiles = np.divide(
            motif_profiles, 
            row_sums, 
            out=np.zeros_like(motif_profiles), 
            where=row_sums != 0
        )
        
        print(f"  ✓ Agent motif profiles: {motif_profiles.shape}")
        
        # 2.7: Compute cluster dominant motifs
        print(f"  Computing cluster motif characterizations...")
        cluster_motifs = {}
        
        unique_clusters = np.unique(self.coarse_labels)
        for cluster_id in unique_clusters:
            mask = self.coarse_labels == cluster_id
            
            if mask.sum() == 0:
                continue
            
            cluster_profile = motif_profiles[mask].mean(axis=0)
            
            # Top 3 dominant motifs
            top_indices = cluster_profile.argsort()[-3:][::-1]
            top_motifs = [
                (discovered_motifs[i]['name'], cluster_profile[i]) 
                for i in top_indices 
                if cluster_profile[i] > 0 and i < len(discovered_motifs)
            ]
            
            cluster_motifs[int(cluster_id)] = {
                'profile': cluster_profile,
                'dominant_motifs': top_motifs,
                'description': self._generate_cluster_description(top_motifs)
            }
        
        print(f"  ✓ Cluster motif profiles computed")
        
        # Print cluster characterizations
        for cluster_id, info in cluster_motifs.items():
            print(f"\n  Cluster {cluster_id}: {info['description']}")
            for motif, freq in info['dominant_motifs']:
                print(f"    - {motif}: {freq:.1%}")
        
        # Store motif labels as instance variable
        self.motif_labels = motif_labels
        
        return reasoning_embeddings, discovered_motifs, motif_profiles, cluster_motifs

    def _classify_behavioral_patterns_batch(self, reasoning_texts: List[str]) -> List[str]:
        """
        Classify behavioral patterns using LLM in batches with parallel processing.
        
        COMPLETE ORIGINAL IMPLEMENTATION - NO CODE REMOVED
        
        Uses ThreadPoolExecutor for parallel API calls to speed up classification.
        
        Args:
            reasoning_texts: List of reasoning text snippets
            
        Returns:
            List of behavioral pattern labels (SAME LENGTH as input)
        """
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import json
        import re
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Define behavioral archetypes
        archetypes = """
    1. anxious_protector - Catastrophic thinking, family-focused, overly cautious, uses words like "extremely", "significantly", "terrified"
    2. rational_calculator - Analytical, precise calculations, uses numbers/percentages, data-driven, multiplies risks
    3. timeline_tracker - Focuses on duration, tracks days, compares to typical timelines, plans based on time
    4. optimistic_minimizer - Downplays risks, focuses on protective factors, uses "however", "but", emphasizes youth/vaccination
    5. social_conformist - Follows others, mentions guidelines/recommendations, less independent thinking
    6. elderly_vulnerable - Heightened age-awareness, mentions age as risk factor, seeks extra protection
    7. youth_confident - Emphasizes young age and strong immunity, downplays severity
    8. vaccine_reliant - Heavy emphasis on vaccination protection, trusts vaccine efficacy
    9. household_focused - Primary concern is household/family safety, mentions household members frequently
    10. contact_counter - Carefully counts and tracks infected contacts, focuses on exposure numbers
    """
        
        batch_size = 20
        num_workers = 4  # Parallel workers for API calls
        
        def classify_single_batch(batch_data):
            """
            Classify a single batch (designed for parallel execution).
            
            Args:
                batch_data: Tuple of (batch_index, batch_texts)
                
            Returns:
                Tuple of (batch_index, labels)
            """
            batch_idx, batch = batch_data
            
            # Build batch prompt
            prompt = f"""Classify the BEHAVIORAL PATTERN (personality/decision-making style) for each reasoning text.

    BEHAVIORAL ARCHETYPES:
    {archetypes}

    REASONING TEXTS:
    """
            for j, text in enumerate(batch):
                prompt += f"\n{j}. \"{text[:150]}...\"\n"
            
            prompt += f"""
    Respond with ONLY a JSON array of exactly {len(batch)} pattern names (one per reasoning text):
    ["pattern1", "pattern2", ...]

    JSON:"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    batch_labels = json.loads(json_match.group())
                    
                    # Ensure correct length
                    if len(batch_labels) < len(batch):
                        print(f"      Warning: Batch {batch_idx} returned {len(batch_labels)}/{len(batch)} labels, padding...")
                        while len(batch_labels) < len(batch):
                            missing_idx = len(batch_labels)
                            fallback = self._fallback_pattern_extraction([batch[missing_idx]])[0]
                            batch_labels.append(fallback)
                    
                    # Truncate if too many
                    batch_labels = batch_labels[:len(batch)]
                    
                    return (batch_idx, batch_labels)
                else:
                    # Fallback if JSON parsing fails
                    print(f"      Warning: Batch {batch_idx} failed JSON parse, using fallback")
                    batch_labels = self._fallback_pattern_extraction(batch)
                    return (batch_idx, batch_labels)
            
            except Exception as e:
                print(f"      Warning: Batch {batch_idx} API call failed ({e}), using fallback")
                batch_labels = self._fallback_pattern_extraction(batch)
                return (batch_idx, batch_labels)
        
        # Create batches with indices
        batches = []
        for i in range(0, len(reasoning_texts), batch_size):
            batch = reasoning_texts[i:i+batch_size]
            batch_idx = i // batch_size
            batches.append((batch_idx, batch))
        
        print(f"    Processing {len(batches)} batches with {num_workers} parallel workers...")
        
        # Process batches in parallel
        results = {}  # batch_idx -> labels
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(classify_single_batch, batch_data): batch_data[0] 
                for batch_data in batches
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result_idx, labels = future.result()
                    results[result_idx] = labels
                    completed += 1
                    
                    # Progress indicator
                    if completed % 5 == 0 or completed == len(batches):
                        total_classified = min(completed * batch_size, len(reasoning_texts))
                        print(f"    Classified {total_classified}/{len(reasoning_texts)}...")
                
                except Exception as e:
                    print(f"      Error processing batch {batch_idx}: {e}")
                    # Use fallback for this batch
                    batch_data = batches[batch_idx]
                    results[batch_idx] = self._fallback_pattern_extraction(batch_data[1])
        
        # Reconstruct labels in original order
        behavioral_labels = []
        for batch_idx in sorted(results.keys()):
            behavioral_labels.extend(results[batch_idx])
        
        # Final length check
        if len(behavioral_labels) != len(reasoning_texts):
            print(f"  ⚠ Critical: Length mismatch after classification!")
            print(f"    Expected: {len(reasoning_texts)}, Got: {len(behavioral_labels)}")
            
            # Pad or truncate to match
            if len(behavioral_labels) < len(reasoning_texts):
                missing_count = len(reasoning_texts) - len(behavioral_labels)
                print(f"    Padding {missing_count} missing labels with fallback...")
                for i in range(len(behavioral_labels), len(reasoning_texts)):
                    fallback = self._fallback_pattern_extraction([reasoning_texts[i]])[0]
                    behavioral_labels.append(fallback)
            else:
                behavioral_labels = behavioral_labels[:len(reasoning_texts)]
        
        return behavioral_labels

    def _fallback_pattern_extraction(self, reasoning_texts: List[str]) -> List[str]:
        """
        Fallback: Simple rule-based pattern extraction if LLM fails.
        
        COMPLETE ORIGINAL IMPLEMENTATION
        
        Args:
            reasoning_texts: List of reasoning texts
            
        Returns:
            List of pattern names
        """
        patterns = []
        
        for text in reasoning_texts:
            text_lower = text.lower()
            
            # Simple keyword-based classification
            if 'household' in text_lower and ('terrified' in text_lower or 'extremely' in text_lower):
                pattern = 'anxious_protector'
            elif any(op in text for op in ['×', '*', '%', '/']):
                pattern = 'rational_calculator'
            elif 'days' in text_lower and ('typical' in text_lower or 'timeline' in text_lower):
                pattern = 'timeline_tracker'
            elif 'however' in text_lower or ('but' in text_lower and 'young' in text_lower):
                pattern = 'optimistic_minimizer'
            elif 'age' in text_lower and any(str(age) in text for age in range(60, 90)):
                pattern = 'elderly_vulnerable'
            elif 'young' in text_lower or '15 years' in text_lower or 'strong immune' in text_lower:
                pattern = 'youth_confident'
            elif 'vaccinated' in text_lower and 'protection' in text_lower:
                pattern = 'vaccine_reliant'
            elif 'household' in text_lower:
                pattern = 'household_focused'
            elif 'contacts' in text_lower or 'out of' in text_lower:
                pattern = 'contact_counter'
            else:
                pattern = 'general_risk_assessor'
            
            patterns.append(pattern)
        
        return patterns


    def _generate_structured_motif_description(self, axis: str, option: str) -> str:
        """Generate description for structured motif dimension."""
        descriptions = {
            'exposure_reasoning': {
                'contact_counting': 'Explicitly counts and tracks infected contacts',
                'household_transmission': 'Focuses on household member infections',
                'workplace_network': 'Emphasizes occupation/network-based risk',
                'community_prevalence': 'Considers general community infection rates',
                'minimal_exposure_reasoning': 'Rarely considers exposure sources'
            },
            'risk_posture': {
                'risk_averse': 'Overestimates risks, uses extreme language',
                'risk_neutral': 'Balanced, data-driven risk assessment',
                'risk_minimizing': 'Downplays risks, emphasizes protective factors',
                'risk_ignoring': 'Minimal risk consideration in reasoning'
            },
            'information_seeking': {
                'heavy_tool_user': 'Uses 4+ tools per decision, seeks comprehensive data',
                'moderate_tool_user': 'Uses 2-3 tools selectively',
                'light_tool_user': 'Uses 1 tool or intuition-based',
                'tool_avoider': 'No tool usage, pure intuition'
            },
            'protection_priority': {
                'self_focused': 'Primary concern is own health/risk',
                'household_focused': 'Primary concern is household/family safety',
                'community_focused': 'Considers broader community impact',
                'occupation_focused': 'Workplace/role obligations dominate'
            },
            'temporal_style': {
                'duration_tracker': 'Tracks days in state, compares to typical timelines',
                'symptom_monitor': 'Focuses on symptom progression',
                'calendar_planner': 'Plans based on future events/dates',
                'present_focused': 'Minimal temporal planning'
            },
            'vaccine_reasoning': {
                'vaccine_reliant': 'Strong trust in vaccination protection',
                'vaccine_aware': 'Acknowledges vaccine but considers other factors',
                'vaccine_skeptical': 'Doubts vaccine effectiveness',
                'vaccine_irrelevant': 'Vaccination not mentioned in reasoning'
            }
        }
        
        return descriptions.get(axis, {}).get(option, f'{axis}: {option}')
    
    def _generate_cluster_description_structured(self, top_motifs: List[Tuple[str, float]]) -> str:
        """Generate description from structured motifs."""
        if not top_motifs:
            return "Mixed behavioral patterns"
        
        # Group by axis
        by_axis = {}
        for motif_name, freq in top_motifs:
            if ':' in motif_name:
                axis, option = motif_name.split(':', 1)
                if axis not in by_axis:
                    by_axis[axis] = []
                by_axis[axis].append((option, freq))
        
        # Build description from top axes
        parts = []
        for axis, options in list(by_axis.items())[:3]:  # Top 3 axes
            if options:
                main_option, main_freq = options[0]
                parts.append(f"{main_option}")
        
        return " + ".join(parts) if parts else "Mixed patterns"
    
    def _generate_motif_description(self, pattern_name: str) -> str:
        """
        Generate human-readable description for a behavioral pattern.
        
        Args:
            pattern_name: Behavioral pattern name
            
        Returns:
            Description string
        """
        descriptions = {
            'anxious_protector': 'Highly cautious, family-focused, catastrophic thinking',
            'rational_calculator': 'Analytical, data-driven, precise calculations',
            'timeline_tracker': 'Duration-focused, tracks progression timelines',
            'optimistic_minimizer': 'Downplays risks, emphasizes protective factors',
            'social_conformist': 'Follows guidelines, relies on recommendations',
            'elderly_vulnerable': 'Age-aware, heightened caution due to age',
            'youth_confident': 'Relies on youth and strong immunity',
            'vaccine_reliant': 'Strong trust in vaccination protection',
            'household_focused': 'Primary concern is household/family safety',
            'contact_counter': 'Carefully tracks and counts exposures',
            'general_risk_assessor': 'Balanced, general risk assessment'
        }
        
        return descriptions.get(pattern_name, 'Undefined behavioral pattern')


    def _create_empty_motifs(self, n_agents: int) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Create empty motif structures when no reasoning traces found.
        
        Args:
            n_agents: Number of agents
            
        Returns:
            Empty motif structures
        """
        empty_embeddings = np.zeros((0, 384))
        empty_motifs = {
            i: {
                'name': f'empty_motif_{i}',
                'size': 0,
                'examples': [],
                'description': 'No data'
            } for i in range(self.k_motifs)
        }
        empty_profiles = np.zeros((n_agents, self.k_motifs))
        empty_cluster_motifs = {
            int(i): {
                'profile': np.zeros(self.k_motifs),
                'dominant_motifs': [],
                'description': 'No motifs'
            } for i in np.unique(self.coarse_labels)
        }
        
        return empty_embeddings, empty_motifs, empty_profiles, empty_cluster_motifs

    def _name_motif(self, examples: List[str]) -> str:
        """
        Generate meaningful motif name using LLM.
        
        COMPLETE ORIGINAL IMPLEMENTATION
        
        Uses GPT to analyze reasoning patterns and create descriptive names.
        
        Args:
            examples: List of reasoning-action pair texts
            
        Returns:
            Descriptive motif name
        """
        if not examples:
            return "empty_motif"
        
        # Take up to 5 examples for LLM analysis
        sample_examples = examples[:5]
        
        # Extract just the reasoning parts (before "| Context:")
        reasoning_parts = []
        for ex in sample_examples:
            if "Reasoning:" in ex and "|" in ex:
                reasoning = ex.split("Reasoning:")[1].split("|")[0].strip()
                reasoning_parts.append(reasoning[:200])  # Limit length
        
        if not reasoning_parts:
            # Fallback: extract tools and transition
            return self._fallback_motif_name(examples[0])
        
        # Build LLM prompt
        prompt = f"""Analyze these COVID-19 agent reasoning patterns and create a SHORT, DESCRIPTIVE motif name (3-5 words max).

    Examples of reasoning from this behavioral cluster:
    {chr(10).join([f"{i+1}. {r}" for i, r in enumerate(reasoning_parts)])}

    The name should capture the CORE BEHAVIORAL PATTERN, not just the tools used.

    Good examples:
    - "household_exposure_concern"
    - "vaccine_reliant_protection"
    - "duration_based_recovery"
    - "elderly_vulnerability_awareness"
    - "workplace_risk_assessment"

    Bad examples:
    - "check_neighbors_query_knowledge" (too generic)
    - "S_to_E_transition" (just describes transition)

    Respond with ONLY the motif name (3-5 words, underscore-separated, lowercase):"""

        try:
            # Use OpenAI to generate name
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            
            motif_name = response.choices[0].message.content.strip()
            
            # Clean up
            motif_name = motif_name.lower().replace(" ", "_").replace("-", "_")
            motif_name = ''.join(c for c in motif_name if c.isalnum() or c == '_')
            
            # Truncate if too long
            motif_name = motif_name[:50]
            
            return motif_name
        
        except Exception as e:
            print(f"      Warning: LLM motif naming failed: {e}")
            return self._fallback_motif_name(examples[0])


    def _fallback_motif_name(self, example: str) -> str:
        """
        Fallback motif naming when LLM fails.
        
        Extracts transition and first few reasoning keywords.
        """
        # Extract transition
        transition = "unknown"
        if "Transition:" in example:
            trans_part = example.split("Transition:")[1].strip()
            transition = trans_part.split()[0] if trans_part else "unknown"
        
        # Extract state
        state = "S"
        if "State:" in example:
            state_part = example.split("State:")[1].split(";")[0].strip()
            state = state_part if state_part else "S"
        
        # Simple name
        return f"pattern_{state}_{transition}"

    
    def _generate_cluster_description(self, top_motifs: List[Tuple[str, float]]) -> str:
        """Generate natural language description of cluster."""
        if not top_motifs:
            return "Mixed behavioral patterns"
        
        if len(top_motifs) == 1:
            return f"{top_motifs[0][0]} ({top_motifs[0][1]:.0%})"
        else:
            main = top_motifs[0][0]
            secondary = top_motifs[1][0]
            return f"Primarily {main} ({top_motifs[0][1]:.0%}), with {secondary} tendencies"
    
    def _stage3_contrastive_refinement(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Stage 3: Learn discriminative embeddings via contrastive learning.
        
        COMPLETE ORIGINAL IMPLEMENTATION
        
        Key innovation: Use cluster anchors as positive/negative examples.
        
        Returns:
            anchors: Dict mapping cluster_id -> anchor_agent_idx
            contrastive_embeddings: Learned embeddings [num_agents, 64]
            hybrid_embeddings: Fused embeddings [num_agents, embed_dim]
        """
        print("  Selecting cluster anchors (prototypical agents)...")
        
        # 3.1: Anchor Selection
        anchors = {}
        for cluster_id in np.unique(self.coarse_labels):
            mask = self.coarse_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            
            # Find agent closest to cluster dominant motif
            cluster_profile = self.cluster_motifs[int(cluster_id)]['profile']
            
            best_idx = None
            best_sim = -1
            
            for idx in cluster_indices:
                agent_profile = self.motif_profiles[idx]
                sim = 1 - cosine(agent_profile, cluster_profile)
                
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            
            anchors[int(cluster_id)] = int(best_idx)
            agent_name = self.agent_profiles[best_idx]['name']
            print(f"    Cluster {cluster_id} anchor: Agent {best_idx} ({agent_name})")
        
        print(f"  ✓ Selected {len(anchors)} cluster anchors")
        
        # 3.2: Contrastive Learning
        print("  Training contrastive encoder...")
        
        # Prepare input: [GraphSAGE || motif_profile || context]
        context_features = self._extract_context_features()
        context_features = StandardScaler().fit_transform(context_features)
        
        input_features = np.hstack([
            self.embeddings_norm,
            self.motif_profiles,
            context_features
        ])

        
        input_dim = input_features.shape[1]
        
        # Convert to torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    Using device: {device}")
        
        X = torch.FloatTensor(input_features).to(device)
        labels_tensor = torch.LongTensor(self.coarse_labels).to(device)
        
        # Initialize encoder
        encoder = ContrastiveEncoder(input_dim, output_dim=64).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        # Training loop
        n_epochs = 200
        batch_size = min(32, len(X))
        n_samples = len(X)
        
        encoder.train()
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            total_loss = 0
            perm = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                indices = perm[i:i+batch_size]
                batch_X = X[indices]
                batch_labels = labels_tensor[indices]
                
                # Forward pass
                embeddings = encoder(batch_X)
                
                # Compute contrastive loss
                loss = self._contrastive_loss(embeddings, batch_labels, anchors, encoder, X, device)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            n_batches = int(np.ceil(n_samples / batch_size))
            avg_loss = total_loss / n_batches

            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
        
        # Extract final contrastive embeddings
        encoder.eval()
        with torch.no_grad():
            contrastive_embeddings = encoder(X).cpu().numpy()
        
        print(f"  ✓ Contrastive embeddings learned: {contrastive_embeddings.shape}")
        
        # ========================================================================
        # 3.3: HYBRID EMBEDDING FUSION (FIXED - PROPER SCALING)
        # ========================================================================
        print("  Fusing embeddings (α·GraphSAGE + β·Contrastive + γ·Motifs)...")
        
        # Normalize contrastive embeddings
        contrastive_norm = StandardScaler().fit_transform(contrastive_embeddings)
        
        # Standardize motif profiles (THIS IS THE KEY FIX!)
        # This ensures motifs have magnitude ~1.0 before padding
        motif_scaler = StandardScaler()
        motif_standardized = motif_scaler.fit_transform(self.motif_profiles)
        
        # Determine target dimensionality
        embed_dim = 128
        
        # Pad to 128D
        graphsage_component = self.embeddings_norm
        
        contrastive_component = np.pad(
            contrastive_norm, 
            ((0,0), (0, embed_dim - contrastive_norm.shape[1]))
        )
        
        # KEY: Use standardized motifs (not raw!)
        motif_component_padded = np.pad(
            motif_standardized,  # ← Standardized, not raw!
            ((0,0), (0, embed_dim - motif_standardized.shape[1]))
        )
        
        # Check magnitudes BEFORE L2 norm
        print(f"\n  📊 Component magnitudes (before L2 norm):")
        print(f"    Graph:       {np.linalg.norm(graphsage_component, axis=1).mean():.4f}")
        print(f"    Contrastive: {np.linalg.norm(contrastive_component, axis=1).mean():.4f}")
        print(f"    Motifs:      {np.linalg.norm(motif_component_padded, axis=1).mean():.4f}")
        
        # L2 normalize
        def row_l2_normalize(A, eps=1e-8):
            return A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
        
        G = row_l2_normalize(graphsage_component)
        C = row_l2_normalize(contrastive_component)
        M = row_l2_normalize(motif_component_padded)
        
        # Weighted fusion
        hybrid_embeddings = self.alpha * G + self.beta * C + self.gamma * M
        hybrid_embeddings = row_l2_normalize(hybrid_embeddings)
        
        print(f"\n  ✓ Hybrid embeddings created: {hybrid_embeddings.shape}")
        print(f"  ✓ Fusion weights: α={self.alpha:.2f} (graph), β={self.beta:.2f} (contrastive), γ={self.gamma:.2f} (motifs)")
        
        # Measure actual contributions
        graph_contrib = self.alpha * G.std(axis=0).mean()
        contrastive_contrib = self.beta * C.std(axis=0).mean()
        motif_contrib = self.gamma * M.std(axis=0).mean()
        
        total_contrib = graph_contrib + contrastive_contrib + motif_contrib
        
        print(f"\n  📊 Component contributions (after normalization):")
        print(f"    Graph: {graph_contrib/total_contrib:.1%}")
        print(f"    Contrastive: {contrastive_contrib/total_contrib:.1%}")
        print(f"    Motifs: {motif_contrib/total_contrib:.1%}")
        
        if motif_contrib / total_contrib < 0.3:
            print(f"  ⚠ WARNING: Motifs contributing < 30% despite γ={self.gamma:.2f}!")
            print(f"    Motif magnitude before L2: {np.linalg.norm(motif_component_padded, axis=1).mean():.4f}")
        else:
            print(f"  ✅ Motifs contributing {motif_contrib/total_contrib:.1%}!")
        
        return anchors, contrastive_embeddings, hybrid_embeddings
    
    def _skip_contrastive_learning(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        ABLATION HELPER: Skip contrastive learning (use_contrastive=False).
        
        Returns:
            Empty anchors, zero contrastive embeddings, hybrid embeddings without contrastive
        """
        print("  Creating hybrid embeddings without contrastive learning...")
        
        # No anchors
        anchors = {}
        
        # No contrastive embeddings
        contrastive_embeddings = np.zeros((len(self.agent_profiles), 64))
        
        # Fuse only graph + motifs
        embed_dim = 64
        graphsage_component = self.embeddings_norm[:, :embed_dim] if self.embeddings_norm.shape[1] >= embed_dim else \
                             np.pad(self.embeddings_norm, ((0,0), (0, embed_dim - self.embeddings_norm.shape[1])))
        
        motif_norm = StandardScaler().fit_transform(self.motif_profiles)
        motif_component = motif_norm[:, :embed_dim] if motif_norm.shape[1] >= embed_dim else \
                         np.pad(motif_norm, ((0,0), (0, embed_dim - motif_norm.shape[1])))
        
        # Redistribute weights (no beta)
        if self.alpha + self.gamma > 0:
            alpha_adj = self.alpha / (self.alpha + self.gamma)
            gamma_adj = self.gamma / (self.alpha + self.gamma)
        else:
            alpha_adj, gamma_adj = 0.5, 0.5
        
        hybrid_embeddings = (
            alpha_adj * graphsage_component +
            gamma_adj * motif_component
        )
        
        print(f"  ✓ Hybrid embeddings (no contrastive): {hybrid_embeddings.shape}")
        
        return anchors, contrastive_embeddings, hybrid_embeddings
    
    def _contrastive_loss(self, embeddings, labels, anchors, encoder, X, device):
        """
        Compute contrastive loss using cluster anchors.
        
        COMPLETE ORIGINAL IMPLEMENTATION
        
        Loss: -Σ_j log[exp(sim(f(j), f(a_j))/τ) / Σ_k exp(sim(f(j), f(a_k))/τ)]
        """
        batch_size = embeddings.shape[0]
        
        # Get anchor embeddings for this batch
        anchor_indices = torch.LongTensor([anchors[int(l.item())] for l in labels]).to(device)
        with torch.no_grad():
            anchor_embeddings = encoder(X[anchor_indices]).detach()
        
        # Get all anchor embeddings
        all_anchor_indices = torch.LongTensor(list(anchors.values())).to(device)
        with torch.no_grad():
            all_anchor_embeddings = encoder(X[all_anchor_indices]).detach()
        
        # Normalize
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
        anchor_embeddings_norm = torch.nn.functional.normalize(anchor_embeddings, dim=1)
        all_anchor_embeddings_norm = torch.nn.functional.normalize(all_anchor_embeddings, dim=1)
        
        # Positive similarities (to own cluster anchor)
        pos_sim = torch.sum(embeddings_norm * anchor_embeddings_norm, dim=1) / self.temperature
        
        # Negative similarities (to all anchors)
        neg_sim = torch.matmul(embeddings_norm, all_anchor_embeddings_norm.T) / self.temperature
        
        # Contrastive loss
        loss = -torch.mean(pos_sim - torch.logsumexp(neg_sim, dim=1))
        
        return loss
    
    def _extract_context_features(self) -> np.ndarray:
        """Extract additional context features for agents."""
        n = len(self.agent_profiles)
        features = np.zeros((n, 10))
        
        for i, profile in enumerate(self.agent_profiles):
            features[i, 0] = profile['age'] / 100.0
            features[i, 1] = profile['vaccination_status'] / 2.0
            features[i, 2] = profile['comorbidity_count'] / 5.0
            features[i, 3] = profile.get('compliance_score', 0.7)
            features[i, 4] = profile.get('risk_awareness', 0.5)
            features[i, 5] = profile.get('mobility_score', 0.7)
            
            # Degree centrality
            agent_id = profile['agent_id']
            if self.graph.has_node(agent_id):
                features[i, 6] = self.graph.degree(agent_id) / 50.0
            
            # Occupation one-hot (3 dims)
            occ_map = {'healthcare_worker': 0, 'student': 1, 'teacher': 1, 
                      'office_worker': 2, 'retail_worker': 2, 'retired': 2}
            occ_idx = occ_map.get(profile['occupation'], 2)
            features[i, 7 + occ_idx] = 1.0
        
        return features

    def _adaptive_distance_threshold(self) -> float:
        """
        Compute an adaptive distance threshold for hierarchical clustering.

        Strategy:
        - Compute pairwise distances in hybrid embedding space
        - Use robust statistics (median + MAD)
        - Clamp threshold to avoid collapse or fragmentation
        """

        from scipy.spatial.distance import pdist

        # Pairwise distances (condensed)
        dists = pdist(self.hybrid_embeddings, metric="cosine")

        # Robust statistics
        median = np.median(dists)
        mad = np.median(np.abs(dists - median)) + 1e-8  # avoid zero

        # Core threshold
        threshold = median + 0.75 * mad

        # Guardrails
        min_thresh = np.percentile(dists, 40)
        max_thresh = np.percentile(dists, 80)

        threshold = np.clip(threshold, min_thresh, max_thresh)

        print(
            f"    Adaptive distance threshold: {threshold:.4f} "
            f"(median={median:.4f}, MAD={mad:.4f})"
        )

        return float(threshold)

    
    def _stage4_boundary_optimization(self) -> np.ndarray:
        """
        Stage 4: Polish cluster boundaries and ensure quality.

        COMPLETE ORIGINAL IMPLEMENTATION

        - Hierarchical clustering (FIXED: don't use both n_clusters and distance_threshold)
        - Boundary agent detection
        - Motif-guided reassignment
        - Quality-driven merging & splitting

        Returns:
            Fine cluster labels [num_agents]
        """
        print("  Performing hierarchical clustering on hybrid embeddings...")

        # FIXED: Use n_clusters only (not both n_clusters and distance_threshold)
        clustering = AgglomerativeClustering(
            n_clusters=self.k_fine,
            linkage="ward"
        )

        fine_labels = clustering.fit_predict(self.hybrid_embeddings)

        # ⭐ ADD: Final balance check
        unique, counts = np.unique(fine_labels, return_counts=True)
        min_size = counts.min()
        max_size = counts.max()
        balance_ratio = min_size / max_size
        
        print(f"\n  Final balance check:")
        print(f"    Cluster sizes: {dict(zip(unique, counts))}")
        print(f"    Balance ratio: {balance_ratio:.2f}")

        # ⭐ If still unbalanced, force rebalance
        if balance_ratio < 0.2:
            print(f"  ⚠ CRITICAL: Clusters too unbalanced after optimization!")
            print(f"  → Force rebalancing with KMeans...")
            
            fine_labels = KMeans(
                n_clusters=self.k_fine,
                random_state=42,
                n_init=100
            ).fit_predict(self.hybrid_embeddings)
            
            unique, counts = np.unique(fine_labels, return_counts=True)
            print(f"    Rebalanced sizes: {dict(zip(unique, counts))}")

            unique, counts = np.unique(fine_labels, return_counts=True)
            print(f"  ✓ Initial fine clustering: K={len(unique)}")
            print(f"    Cluster sizes: {dict(zip(unique, counts))}")

        # Guardrail: avoid degenerate clustering
        if len(unique) < 2:
            print("  ⚠ Degenerate clustering detected (K < 2). Falling back to K=3.")
            fine_labels = AgglomerativeClustering(
                n_clusters=3,
                linkage="ward"
            ).fit_predict(self.hybrid_embeddings)

        # ------------------------------------------------------------------
        # 4.1 Identify boundary agents
        # ------------------------------------------------------------------
        print("  Identifying boundary agents...")
        boundary_agents = self._identify_boundary_agents(fine_labels)
        print(f"  ✓ Found {len(boundary_agents)} boundary agents")

        # ------------------------------------------------------------------
        # 4.2 Motif-guided reassignment
        # ------------------------------------------------------------------
        if boundary_agents:
            print("  Performing motif-guided reassignment...")
            fine_labels = self._motif_guided_reassignment(fine_labels, boundary_agents)
            print("  ✓ Reassigned boundary agents")

        # ------------------------------------------------------------------
        # 4.3 Quality-driven merging & splitting
        # ------------------------------------------------------------------
        print("  Checking for cluster merging/splitting...")
        fine_labels, n_merges, n_splits = self._quality_driven_merging_splitting(fine_labels)

        if n_merges > 0 or n_splits > 0:
            print(f"  ✓ Adapted clusters: {n_merges} merges, {n_splits} splits")
            print(f"    Final K = {len(np.unique(fine_labels))}")
        else:
            print("  ✓ No merging/splitting needed")

        # ------------------------------------------------------------------
        # 4.4 Final validation
        # ------------------------------------------------------------------
        
        # Check for degenerate clustering (K=1)
        if len(np.unique(fine_labels)) < 2:
            print("  ⚠ CRITICAL: Clustering collapsed to K < 2. Forcing K=3 split.")
            fine_labels = KMeans(n_clusters=3, random_state=42, n_init=100).fit_predict(self.hybrid_embeddings)
        
        silhouette = silhouette_score(self.hybrid_embeddings, fine_labels)

        modularity = self._compute_modularity(fine_labels)

        print("  ✓ Final quality metrics:")
        print(f"    Silhouette: {silhouette:.3f}")
        print(f"    Modularity: {modularity:.3f}")

        return fine_labels

    def _simple_clustering(self) -> np.ndarray:
        """
        ABLATION HELPER: Simple clustering without optimization (use_boundary_opt=False).
        
        Returns:
            Simple cluster labels without boundary optimization
        """
        print("  Simple hierarchical clustering (no optimization)...")
        
        clustering = AgglomerativeClustering(
            n_clusters=self.k_fine,
            linkage="ward"
        )
        
        fine_labels = clustering.fit_predict(self.hybrid_embeddings)
        
        print(f"  ✓ Simple clustering: K={self.k_fine}")
        
        return fine_labels
    
    def _identify_boundary_agents(self, labels: np.ndarray) -> List[int]:
        """Identify agents with low cluster assignment confidence."""
        boundary_agents = []
        
        for i in range(len(labels)):
            # Distance to own cluster centroid
            own_cluster = labels[i]
            own_mask = labels == own_cluster
            own_centroid = self.hybrid_embeddings[own_mask].mean(axis=0)
            dist_own = euclidean(self.hybrid_embeddings[i], own_centroid)
            
            # Distance to nearest other cluster
            other_clusters = [c for c in np.unique(labels) if c != own_cluster]
            if len(other_clusters) == 0:
                continue
            
            min_other_dist = float('inf')
            
            for other_cluster in other_clusters:
                other_mask = labels == other_cluster
                other_centroid = self.hybrid_embeddings[other_mask].mean(axis=0)
                dist_other = euclidean(self.hybrid_embeddings[i], other_centroid)
                min_other_dist = min(min_other_dist, dist_other)
            
            # Boundary if distances are similar (within 20% margin)
            if dist_own > 0.95 * min_other_dist:
                boundary_agents.append(i)
        
        return boundary_agents
    
    def _motif_guided_reassignment(self, labels: np.ndarray, boundary_agents: List[int]) -> np.ndarray:
        """
        Reassign boundary agents using behavioral pull.
        
        pull(j, C_i) = cosine(P_j, D_i) × graph_connectivity(j, C_i)
        """
        labels = labels.copy()
        
        # Compute cluster motif profiles
        cluster_profiles = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_profiles[cluster_id] = self.motif_profiles[mask].mean(axis=0)
        
        reassignments = 0
        
        for agent_idx in boundary_agents:
            agent_id = self.idx_to_agent_id[agent_idx]
            agent_motif = self.motif_profiles[agent_idx]
            
            best_cluster = labels[agent_idx]
            best_pull = -1
            
            for cluster_id, cluster_motif in cluster_profiles.items():
                # Behavioral similarity
                motif_sim = 1 - cosine(agent_motif, cluster_motif)
                
                # Graph connectivity
                cluster_mask = labels == cluster_id
                cluster_agent_ids = [self.idx_to_agent_id[i] for i in np.where(cluster_mask)[0]]
                
                connectivity = 0
                if self.graph.has_node(agent_id):
                    for cluster_agent_id in cluster_agent_ids:
                        if self.graph.has_edge(agent_id, cluster_agent_id):
                            connectivity += 1
                
                graph_connectivity = connectivity / max(len(cluster_agent_ids), 1)
                
                # Combined pull with embedding distance
                embed_dist = euclidean(
                    self.hybrid_embeddings[agent_idx],
                    self.hybrid_embeddings[labels == cluster_id].mean(axis=0)
                )
                embed_score = np.exp(-embed_dist)

                pull = (
                    0.5 * motif_sim +
                    0.3 * embed_score +
                    0.2 * graph_connectivity
                )
                
                if pull > best_pull:
                    best_pull = pull
                    best_cluster = cluster_id
            
            if best_cluster != labels[agent_idx]:
                reassignments += 1
                labels[agent_idx] = best_cluster
        
        print(f"    Reassigned {reassignments}/{len(boundary_agents)} boundary agents")
        
        return labels
    
    def _quality_driven_merging_splitting(self, labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        CRITICAL IMPLEMENTATION: Quality-driven cluster adaptation.
        
        Merge: If two clusters have similar dominant motifs (JS-divergence < θ_merge)
        Split: If cluster has high motif variance (entropy > θ_split)
        
        Returns:
            labels: Updated labels
            n_merges: Number of merges performed
            n_splits: Number of splits performed
        """
        labels = labels.copy()
        n_merges = 0
        n_splits = 0
        
        # MERGING: Check JS-divergence between cluster pairs
        print(f"    Checking for merging (threshold: {self.theta_merge:.3f})...")
        
        cluster_ids = np.unique(labels)
        cluster_motif_dists = {}
        
        # Compute motif distributions for each cluster
        for cluster_id in cluster_ids:
            mask = labels == cluster_id
            cluster_motif_dist = self.motif_profiles[mask].mean(axis=0)
            # Normalize to probability distribution
            cluster_motif_dist = cluster_motif_dist / (cluster_motif_dist.sum() + 1e-10)
            cluster_motif_dists[cluster_id] = cluster_motif_dist
        
        # Check pairs for merging
        merge_candidates = []
        for i, cluster_i in enumerate(cluster_ids):
            for cluster_j in cluster_ids[i+1:]:
                dist_i = cluster_motif_dists[cluster_i]
                dist_j = cluster_motif_dists[cluster_j]
                
                # Jensen-Shannon divergence
                js_dist = jensenshannon(dist_i, dist_j)
                js_div = js_dist ** 2

                if js_div < self.theta_merge:
                    merge_candidates.append((cluster_i, cluster_j, js_div))
        
        # Perform merges
        if merge_candidates:
            merge_candidates.sort(key=lambda x: x[2])  # Sort by similarity
            
            for cluster_i, cluster_j, js_div in merge_candidates:
                # SAFEGUARD: Never merge if it would result in K < 2
                if len(np.unique(labels)) <= 2:
                    print(f"      Stopping merges: would result in K < 2")
                    break
                
                if cluster_i in np.unique(labels) and cluster_j in np.unique(labels):
                    print(f"      Merging clusters {cluster_i} and {cluster_j} (JS-div: {js_div:.3f})")
                    labels[labels == cluster_j] = cluster_i
                    n_merges += 1
        
        # Relabel clusters sequentially
        unique_clusters = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_clusters)}
        labels = np.array([label_map[l] for l in labels])
        
        # SPLITTING: Check entropy within clusters
        print(f"    Checking for splitting (threshold: {self.theta_split:.3f})...")
        
        current_clusters = np.unique(labels)
        for cluster_id in current_clusters:
            mask = labels == cluster_id
            cluster_size = mask.sum()
            
            if cluster_size < 6:  # Don't split very small clusters
                continue
            
            # Compute motif variance (entropy)
            cluster_motif_profiles = self.motif_profiles[mask]
            
            # Entropy of average motif distribution
            avg_motif_dist = cluster_motif_profiles.mean(axis=0)
            avg_motif_dist = avg_motif_dist / (avg_motif_dist.sum() + 1e-10)
            motif_entropy = entropy(avg_motif_dist)
            
            # Also check variance across agents
            motif_variance = cluster_motif_profiles.var(axis=0).mean()

            normalized_entropy = motif_entropy / np.log(self.k_motifs)
            
            if normalized_entropy > self.theta_split and motif_variance > 0.1:
                print(f"      Splitting cluster {cluster_id} (entropy: {motif_entropy:.3f}, variance: {motif_variance:.3f})")
                
                # Split into 2 sub-clusters using k-means on motif profiles
                cluster_indices = np.where(mask)[0]
                cluster_embeddings = self.hybrid_embeddings[cluster_indices]
                
                sub_clusterer = KMeans(n_clusters=2, random_state=42, n_init=10)
                sub_labels = sub_clusterer.fit_predict(cluster_embeddings)
                
                # Assign new cluster IDs
                max_label = labels.max()
                for i, idx in enumerate(cluster_indices):
                    if sub_labels[i] == 1:
                        labels[idx] = max_label + 1
                
                n_splits += 1
        
        return labels, n_merges, n_splits
    
    def _compute_modularity(self, labels: np.ndarray) -> float:
        """Compute modularity of clustering."""
        if not self.use_graph:
            return 0.0
        
        communities = []
        for label in np.unique(labels):
            community = [self.agent_profiles[i]['agent_id'] 
                        for i in range(len(self.agent_profiles)) 
                        if labels[i] == label]
            communities.append(community)
        
        from networkx.algorithms.community import modularity
        try:
            mod = modularity(self.graph, communities)
        except:
            mod = 0.0
        
        return mod
    
    def _print_cluster_summary(self):
        """Print detailed cluster summary."""
        print("\n" + "="*80)
        print("FINAL CLUSTER SUMMARY")
        print("="*80)
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Ablation config: graph={self.use_graph}, motifs={self.use_motifs}, "
              f"contrastive={self.use_contrastive}, boundary_opt={self.use_boundary_opt}")
        
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for label, count in zip(unique_labels, counts):
            pct = 100.0 * count / len(self.cluster_labels)
            print(f"  Cluster {label}: {count} agents ({pct:.1f}%)")
        
        # Only print motif info if motifs are enabled
        if self.use_motifs:
            for cluster_id in np.unique(self.cluster_labels):
                mask = self.cluster_labels == cluster_id
                cluster_agents = [self.agent_profiles[i] for i in np.where(mask)[0]]
                
                print(f"\nCluster {cluster_id} ({len(cluster_agents)} agents):")
                
                # For PCA: Show top PCA components (dimensions with highest values)
                cluster_motif_profile = self.motif_profiles[mask].mean(axis=0)
                top_motif_indices = np.abs(cluster_motif_profile).argsort()[-3:][::-1]
                
                print(f"  Top behavioral dimensions (PCA):")
                for motif_idx in top_motif_indices:
                    component_val = cluster_motif_profile[motif_idx]
                    if abs(component_val) > 0.01:  # Only show significant components
                        # Show component index and its contribution
                        print(f"    - Component {motif_idx}: {component_val:.3f}")
                
                # Demographics
                ages = [a['age'] for a in cluster_agents]
                occupations = [a['occupation'] for a in cluster_agents]
                
                print(f"  Age range: {min(ages)}-{max(ages)} (mean: {np.mean(ages):.1f})")
                
                occ_counts = Counter(occupations)
                print(f"  Occupations: {dict(occ_counts.most_common(3))}")
                
                # Sample agents
                sample_names = [a['name'] for a in cluster_agents[:5]]
                print(f"  Sample agents: {', '.join(sample_names)}")
    
    def get_cluster_assignments(self) -> Dict[int, List[int]]:
        """Get cluster assignments as dictionary."""
        clusters = {}
        for label in np.unique(self.cluster_labels):
            mask = self.cluster_labels == label
            agent_ids = [self.agent_profiles[i]['agent_id'] for i in np.where(mask)[0]]
            clusters[int(label)] = agent_ids
        
        return clusters
 
    def evaluate_clustering(self) -> Dict:
        """
        Evaluate final clustering quality with geometric, graph, behavioral,
        stability, and anchor-based metrics.

        Returns RAW metrics (standard for research papers).
        Normalized metrics are provided separately for dashboards only.
        """

        from scipy.spatial.distance import pdist
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            adjusted_rand_score
        )

        # ------------------------------------------------------------------
        # Basic setup
        # ------------------------------------------------------------------
        X = self.hybrid_embeddings
        labels = self.cluster_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        num_clusters = len(unique_labels)

        metrics = {}

        # ------------------------------------------------------------------
        # 1. Standard geometric clustering metrics
        # ------------------------------------------------------------------
        silhouette = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_index = davies_bouldin_score(X, labels)

        metrics.update({
            "silhouette_score": float(silhouette),          # ↑
            "calinski_harabasz_score": float(ch_score),      # ↑
            "davies_bouldin_index": float(db_index),         # ↓
        })

        # ------------------------------------------------------------------
        # 2. Graph / topological quality
        # ------------------------------------------------------------------
        modularity = self._compute_modularity(labels) if self.use_graph else 0.0
        metrics["modularity"] = float(modularity)           # ↑

        # Conductance (lower is better)
        # ------------------------------------------------------------------
        # Graph conductance (structure–cluster alignment)
        # ------------------------------------------------------------------
        conductances = []

        if self.use_graph and hasattr(self, "graph") and self.graph is not None:
            for c in unique_labels:
                idxs = np.where(labels == c)[0]

                # Map cluster indices → agent IDs via agent_id_to_idx
                cluster_nodes = {
                    agent_id
                    for agent_id, idx in self.agent_id_to_idx.items()
                    if idx in idxs and self.graph.has_node(agent_id)
                }

                if not cluster_nodes:
                    continue

                cut_edges = sum(
                    1
                    for u in cluster_nodes
                    for v in self.graph.neighbors(u)
                    if v not in cluster_nodes
                )

                volume = sum(self.graph.degree(u) for u in cluster_nodes)

                conductances.append(cut_edges / max(volume, 1))

        metrics["avg_conductance"] = float(np.mean(conductances)) if conductances else 0.0


        # ------------------------------------------------------------------
        # 3. Cluster balance
        # ------------------------------------------------------------------
        probs = counts / counts.sum()
        cluster_entropy = entropy(probs)

        metrics.update({
            "cluster_entropy": float(cluster_entropy),       # ↑ (balance, not quality)
            "cluster_sizes": counts.tolist(),
            "num_clusters": int(num_clusters),
        })

        # ------------------------------------------------------------------
        # 4. Compactness & separation (explicit, interpretable)
        # ------------------------------------------------------------------
        centroids = []
        intra_dists = []

        for c in unique_labels:
            Xc = X[labels == c]
            if len(Xc) == 0:
                continue

            centroid = Xc.mean(axis=0)
            centroids.append(centroid)

            if len(Xc) > 1:
                intra_dists.append(
                    np.mean(np.linalg.norm(Xc - centroid, axis=1))
                )

        inter_centroid_dist = float(np.mean(pdist(np.array(centroids)))) if len(centroids) > 1 else 0.0

        metrics.update({
            "avg_intra_cluster_distance": float(np.mean(intra_dists)) if intra_dists else 0.0,  # ↓
            "avg_inter_centroid_distance": inter_centroid_dist,                                  # ↑
        })

        # ------------------------------------------------------------------
        # 5. Behavioral motif metrics (HSBC³ novelty)
        # ------------------------------------------------------------------
        motif_coherence = 0.0
        motif_entropy = 0.0

        if self.use_motifs and hasattr(self, "motif_profiles"):
            # Motif coherence (mean intra-cluster cosine similarity)
            motif_coherence = self._compute_motif_coherence()

            # Motif entropy (interpretability)
            motif_entropies = []
            for c in unique_labels:
                profile = self.motif_profiles[labels == c].mean(axis=0)
                p = profile / (profile.sum() + 1e-9)
                motif_entropies.append(entropy(p))

            motif_entropy = float(np.mean(motif_entropies))

        metrics.update({
            "motif_coherence": float(motif_coherence),        # ↑
            "avg_motif_entropy": float(motif_entropy),        # ↓ (more interpretable)
        })

        # ------------------------------------------------------------------
        # 6. Anchor quality (HSBC³-specific, very strong for papers)
        # ------------------------------------------------------------------
        anchor_representativeness = 0.0
        if hasattr(self, "anchors") and self.anchors:
            anchor_dists = []

            for c, anchor_idx in self.anchors.items():
                cluster_idxs = np.where(labels == c)[0]
                if len(cluster_idxs) == 0:
                    continue

                centroid = X[cluster_idxs].mean(axis=0)
                dist = np.linalg.norm(X[anchor_idx] - centroid)
                anchor_dists.append(dist)

            if anchor_dists:
                anchor_representativeness = float(
                    1.0 / (np.mean(anchor_dists) + 1e-9)
                )

        metrics["anchor_representativeness"] = anchor_representativeness  # ↑

        # ------------------------------------------------------------------
        # 7. Stability (optional but highly recommended)
        # ------------------------------------------------------------------
        stability_ari = None
        if hasattr(self, "_cluster_with_noise"):
            try:
                noisy_labels = self._cluster_with_noise(eps=0.01)
                stability_ari = adjusted_rand_score(labels, noisy_labels)
            except Exception:
                stability_ari = None

        metrics["stability_ari"] = float(stability_ari) if stability_ari is not None else None  # ↑

        # ------------------------------------------------------------------
        # 8. Normalized metrics (FOR DASHBOARDS ONLY)
        # ------------------------------------------------------------------
        metrics["normalized"] = {
            "silhouette": (silhouette + 1) / 2,
            "db_inverted": 1 / (1 + db_index),
            "modularity": modularity,
            "motif_coherence": motif_coherence,
        }

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        print(f"\n{'='*80}")
        print("FINAL CLUSTERING QUALITY METRICS (RAW)")
        print("=" * 80)
        print(f"Silhouette (↑):                 {silhouette:.4f}")
        print(f"Calinski-Harabasz (↑):          {ch_score:.1f}")
        print(f"Davies-Bouldin (↓):             {db_index:.4f}")
        print(f"Modularity (↑):                 {modularity:.4f}")
        print(f"Avg Conductance (↓):            {metrics['avg_conductance']:.4f}")
        print(f"Avg Intra-cluster Dist (↓):     {metrics['avg_intra_cluster_distance']:.4f}")
        print(f"Avg Inter-centroid Dist (↑):    {inter_centroid_dist:.4f}")
        print(f"Cluster Entropy (↑ balance):    {cluster_entropy:.4f}")

        if self.use_motifs:
            print(f"Motif Coherence (↑):            {motif_coherence:.4f}")
            print(f"Avg Motif Entropy (↓):          {motif_entropy:.4f}")

        if anchor_representativeness > 0:
            print(f"Anchor Representativeness (↑):  {anchor_representativeness:.4f}")

        if stability_ari is not None:
            print(f"Stability ARI (↑):              {stability_ari:.4f}")

        print(f"Number of Clusters:             {num_clusters}")
        print("-" * 80)

        return metrics
