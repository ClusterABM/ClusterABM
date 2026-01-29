"""
Build graph from agent profiles and edges.
Singapore COVID-19 data with day_0 states as Initial States.
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict
import matplotlib.pyplot as plt


def status_to_state(status: int) -> str:
    """Convert disease_status number to SEIRD state letter."""
    return {0: 'S', 1: 'E', 2: 'I', 3: 'R', 4: 'D'}.get(status, 'S')


class GraphBuilder:
    """Build and manage agent graph from Singapore COVID-19 data."""
    
    def __init__(self, profiles: List[Dict], edges: List[Dict]):
        """
        Initialize graph builder.
        
        Args:
            profiles: List of agent profile dictionaries (from Singapore extraction)
            edges: List of edge dictionaries with agent_1, agent_2, edge_type, contact_rate
        
        Note:
            Profiles should contain disease_status field (0-4) from Kaggle day_0 column.
            This represents actual epidemic state on January 23, 2020.
        """
        self.profiles = profiles
        self.edges = edges
        self.graph = None
        self.pyg_data = None
        
        print(f"GraphBuilder initialized: {len(profiles)} agents, {len(edges)} edges")
        print(f"Dataset: Singapore COVID-19 (January 23, 2020)")
        
    def build_networkx_graph(self) -> nx.Graph:
        """
        Build NetworkX graph with Singapore COVID-19 specific attributes.
        
        Returns:
            NetworkX graph with agents as nodes
        """
        print("\nBuilding NetworkX graph...")
        
        G = nx.Graph()
        
        # Add nodes with Singapore-specific attributes
        for profile in self.profiles:
            # Get state from disease_status (Kaggle day_0)
            if 'disease_status' in profile:
                state = status_to_state(profile['disease_status'])
            elif 'initial_state' in profile:
                state = profile['initial_state']
            else:
                state = 'S'
            
            G.add_node(
                profile['agent_id'],
                # Basic info
                name=profile['name'],
                age=profile['age'],
                gender=profile.get('gender', 'unknown'),
                nationality=profile.get('nationality', 'unknown'),
                occupation=profile['occupation'],
                
                # Social structure
                household_id=profile['household_id'],
                cluster=profile.get('cluster', ''),
                
                # Epidemic state (from Kaggle day_0)
                state=state,
                disease_status=profile.get('disease_status', 0),
                
                # Singapore-specific
                is_imported=profile.get('is_imported', False),
                days_since_start=profile.get('days_since_start', 0),
                
                # Health attributes
                vaccination=profile['vaccination_status'],
                comorbidities=profile['comorbidity_count'],
                
                # Behavioral attributes
                mobility=profile.get('mobility_score', 0.5),
                compliance=profile.get('compliance_score', 0.5),
                risk_awareness=profile.get('risk_awareness', 0.5)
            )
        
        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge['agent_1'],
                edge['agent_2'],
                weight=edge['contact_rate'],
                edge_type=edge['edge_type']
            )
        
        self.graph = G
        
        # Print statistics
        print(f"✓ Graph created:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Avg degree: {np.mean([d for n, d in G.degree()]):.2f}")
        print(f"  Density: {nx.density(G):.3f}")
        print(f"  Connected: {nx.is_connected(G)}")
        
        # Print state distribution
        from collections import Counter
        states = [G.nodes[n]['state'] for n in G.nodes()]
        state_counts = Counter(states)
        print(f"\n  State distribution (from Kaggle day_0):")
        for state in ['S', 'E', 'I', 'R', 'D']:
            count = state_counts.get(state, 0)
            pct = 100 * count / G.number_of_nodes()
            print(f"    {state}: {count:4d} ({pct:5.1f}%)")
        
        return G
    
    def extract_node_features(self) -> np.ndarray:
        """
        Extract node features for GraphSAGE with Singapore COVID-19 specific attributes.
        
        Returns:
            Feature matrix [num_nodes, num_features]
        
        Features (total: 25 dimensions):
            - Age (1)
            - Occupation one-hot (8)
            - Vaccination status one-hot (3)
            - Comorbidities (1)
            - Mobility, compliance, risk awareness (3)
            - Degree centrality (1)
            - Gender one-hot (3)
            - Is imported (1)
            - Days since start (normalized) (1)
            - Initial disease state one-hot (5) - from Kaggle day_0
        """
        print("\nExtracting node features...")
        
        if self.graph is None:
            self.build_networkx_graph()
        
        features = []
        
        for profile in sorted(self.profiles, key=lambda x: x['agent_id']):
            agent_id = profile['agent_id']
            
            feature_vec = []
            
            # 1. Age (normalized) - 1 feature
            feature_vec.append(profile['age'] / 100.0)
            
            # 2. Occupation (one-hot, 8 categories) - 8 features
            occupations = ['student', 'healthcare_worker', 'teacher', 'office_worker', 
                          'retail_worker', 'service_worker', 'manual_worker', 'retired']
            occ = profile['occupation']
            occ_vec = [1.0 if occ == o else 0.0 for o in occupations]
            feature_vec.extend(occ_vec)
            
            # 3. Vaccination status (one-hot, 3 categories) - 3 features
            vacc_vec = [1.0 if profile['vaccination_status'] == i else 0.0 for i in range(3)]
            feature_vec.extend(vacc_vec)
            
            # 4. Comorbidities (normalized) - 1 feature
            feature_vec.append(profile['comorbidity_count'] / 5.0)
            
            # 5. Mobility score - 1 feature
            feature_vec.append(profile.get('mobility_score', 0.5))
            
            # 6. Compliance score - 1 feature
            feature_vec.append(profile.get('compliance_score', 0.5))
            
            # 7. Risk awareness - 1 feature
            feature_vec.append(profile.get('risk_awareness', 0.5))
            
            # 8. Degree centrality (normalized) - 1 feature
            degree = self.graph.degree(agent_id)
            feature_vec.append(degree / 50.0)
            
            # 9. Gender (one-hot, 3 categories: m, f, unknown) - 3 features
            gender = profile.get('gender', 'unknown')
            gender_vec = [
                1.0 if gender == 'm' else 0.0,
                1.0 if gender == 'f' else 0.0,
                1.0 if gender not in ['m', 'f'] else 0.0
            ]
            feature_vec.extend(gender_vec)
            
            # 10. Is imported case - 1 feature
            feature_vec.append(1.0 if profile.get('is_imported', False) else 0.0)
            
            # 11. Days since start (normalized) - 1 feature
            days = profile.get('days_since_start', 0)
            feature_vec.append(days / 100.0)  # Normalize by ~100 days
            
            # 12. Initial disease state (one-hot, 5 categories) - 5 features
            # This comes from Kaggle day_0 column
            disease_status = profile.get('disease_status', 0)
            state_vec = [1.0 if disease_status == i else 0.0 for i in range(5)]
            feature_vec.extend(state_vec)
            
            features.append(feature_vec)
        
        features = np.array(features, dtype=np.float32)
        
        print(f"✓ Features extracted: {features.shape}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Components:")
        print(f"    - Age: 1")
        print(f"    - Occupation: 8")
        print(f"    - Vaccination: 3")
        print(f"    - Comorbidities: 1")
        print(f"    - Behavioral: 3 (mobility, compliance, risk)")
        print(f"    - Network: 1 (degree)")
        print(f"    - Demographics: 3 (gender)")
        print(f"    - Singapore-specific: 2 (imported, days_since_start)")
        print(f"    - Disease state (Kaggle day_0): 5")
        
        return features
    
    def to_pyg_data(self) -> Data:
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            PyG Data object for GraphSAGE training
        """
        print("\nConverting to PyTorch Geometric format...")
        
        if self.graph is None:
            self.build_networkx_graph()
        
        # Extract node features
        node_features = self.extract_node_features()
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create agent_id to index mapping
        agent_to_idx = {p['agent_id']: i for i, p in enumerate(sorted(self.profiles, key=lambda x: x['agent_id']))}
        
        # Edge indices
        edge_index = []
        edge_weights = []
        
        for edge in self.edges:
            src = agent_to_idx[edge['agent_1']]
            dst = agent_to_idx[edge['agent_2']]
            weight = edge['contact_rate']
            
            # Add both directions (undirected graph)
            edge_index.append([src, dst])
            edge_index.append([dst, src])
            
            edge_weights.append(weight)
            edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.profiles)
        )
        
        self.pyg_data = data
        
        print(f"✓ PyG Data created:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge features: {data.edge_attr.shape}")
        
        return data
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get weighted adjacency matrix."""
        if self.graph is None:
            self.build_networkx_graph()
        
        return nx.to_numpy_array(self.graph, weight='weight')
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics for Singapore COVID-19 data."""
        if self.graph is None:
            self.build_networkx_graph()
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'transitivity': nx.transitivity(self.graph)
        }
        
        # Degree distribution
        degrees = [d for n, d in self.graph.degree()]
        stats['degree_distribution'] = {
            'min': int(min(degrees)),
            'max': int(max(degrees)),
            'median': float(np.median(degrees)),
            'std': float(np.std(degrees))
        }
        
        # Edge type distribution
        edge_types = nx.get_edge_attributes(self.graph, 'edge_type')
        type_counts = {}
        for edge_type in edge_types.values():
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        stats['edge_types'] = type_counts
        
        # State distribution (from Kaggle day_0)
        from collections import Counter
        states = [self.graph.nodes[n]['state'] for n in self.graph.nodes()]
        state_counts = Counter(states)
        stats['state_distribution'] = dict(state_counts)
        
        # Singapore-specific statistics
        imported_count = sum(1 for n in self.graph.nodes() 
                            if self.graph.nodes[n].get('is_imported', False))
        stats['imported_cases'] = {
            'count': imported_count,
            'percentage': 100 * imported_count / self.graph.number_of_nodes()
        }
        
        # Cluster information
        clusters = [self.graph.nodes[n].get('cluster', '') for n in self.graph.nodes() 
                   if self.graph.nodes[n].get('cluster', '')]
        stats['unique_clusters'] = len(set(clusters))
        stats['agents_in_clusters'] = len(clusters)
        
        return stats
    
    def visualize_graph(self, save_path: str = None, figsize=(14, 12)):
        """
        Visualize the Singapore COVID-19 agent graph with SEIRD states from Kaggle day_0.
        
        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if self.graph is None:
            self.build_networkx_graph()
        
        print("\nVisualizing Singapore COVID-19 agent network...")
        
        plt.figure(figsize=figsize)
        
        # Layout - spring layout for better visualization
        print("  Computing layout...")
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=50)
        
        # Node colors by SEIRD state (from Kaggle day_0)
        state_colors = {
            'S': '#2ecc71',  # Green - Susceptible
            'E': '#f39c12',  # Orange - Exposed
            'I': '#e74c3c',  # Red - Infected
            'R': '#3498db',  # Blue - Recovered
            'D': '#34495e'   # Dark gray - Dead
        }
        node_colors = [state_colors.get(self.graph.nodes[n]['state'], '#95a5a6') 
                      for n in self.graph.nodes()]
        
        # Node sizes by degree
        node_sizes = [300 + 100 * self.graph.degree(n) for n in self.graph.nodes()]
        
        # Draw edges by type
        print("  Drawing edges...")
        edge_types = set(nx.get_edge_attributes(self.graph, 'edge_type').values())
        edge_colors_map = {
            'household': '#FF6B6B',
            'work': '#4ECDC4',
            'school': '#95E1D3',
            'cluster': '#F3A683',
            'contact': '#AA96DA',
            'community': '#FCBAD3'
        }
        
        for edge_type in edge_types:
            edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                     if d.get('edge_type') == edge_type]
            if edges:
                nx.draw_networkx_edges(
                    self.graph, pos, edgelist=edges,
                    edge_color=edge_colors_map.get(edge_type, 'gray'),
                    width=1.5, alpha=0.5
                )
        
        # Draw nodes
        print("  Drawing nodes...")
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1.5
        )
        
        # Labels (conditional on graph size)
        if len(self.graph.nodes()) <= 50:
            labels = {n: f"{n}\n{self.graph.nodes[n]['name'].split()[0]}" 
                     for n in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=7)
        elif len(self.graph.nodes()) <= 200:
            # Medium size - just show IDs
            labels = {n: str(n) for n in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=5)
        else:
            # Large graph - no labels
            pass
        
        # Legend
        from matplotlib.patches import Patch
        from collections import Counter
        
        # Get state counts for legend
        states = [self.graph.nodes[n]['state'] for n in self.graph.nodes()]
        state_counts = Counter(states)
        
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='States (Kaggle day_0):'),
            Patch(facecolor='#2ecc71', label=f'Susceptible (S={state_counts.get("S", 0)})'),
            Patch(facecolor='#f39c12', label=f'Exposed (E={state_counts.get("E", 0)})'),
            Patch(facecolor='#e74c3c', label=f'Infected (I={state_counts.get("I", 0)})'),
            Patch(facecolor='#3498db', label=f'Recovered (R={state_counts.get("R", 0)})'),
            Patch(facecolor='#34495e', label=f'Dead (D={state_counts.get("D", 0)})'),
            Patch(facecolor='white', edgecolor='white', label=''),
            Patch(facecolor='white', edgecolor='black', label='Edge Types:'),
        ]
        
        # Add edge types to legend
        for edge_type in sorted(edge_types):
            color = edge_colors_map.get(edge_type, 'gray')
            legend_elements.append(Patch(facecolor=color, label=edge_type.capitalize()))
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=9, 
                  framealpha=0.95, edgecolor='black')
        
        plt.title("Singapore COVID-19 Agent Network\n"
                 "Initial States from Kaggle (January 23, 2020)", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            print(f"  Saving to {save_path}...")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graph visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_imported_cases_subgraph(self) -> nx.Graph:
        """
        Get subgraph containing only imported cases and their contacts.
        
        Returns:
            Subgraph with imported cases
        """
        if self.graph is None:
            self.build_networkx_graph()
        
        # Get imported case nodes
        imported_nodes = [n for n in self.graph.nodes() 
                         if self.graph.nodes[n].get('is_imported', False)]
        
        # Get their neighbors
        all_nodes = set(imported_nodes)
        for node in imported_nodes:
            all_nodes.update(self.graph.neighbors(node))
        
        subgraph = self.graph.subgraph(all_nodes).copy()
        
        print(f"Imported cases subgraph:")
        print(f"  Imported cases: {len(imported_nodes)}")
        print(f"  Total nodes (with contacts): {len(all_nodes)}")
        print(f"  Edges: {subgraph.number_of_edges()}")
        
        return subgraph
    
    def get_cluster_subgraph(self, cluster_name: str) -> nx.Graph:
        """
        Get subgraph for a specific cluster.
        
        Args:
            cluster_name: Name of the cluster
            
        Returns:
            Subgraph containing agents in the specified cluster
        """
        if self.graph is None:
            self.build_networkx_graph()
        
        cluster_nodes = [n for n in self.graph.nodes() 
                        if self.graph.nodes[n].get('cluster', '') == cluster_name]
        
        if not cluster_nodes:
            print(f"No agents found in cluster: {cluster_name}")
            return nx.Graph()
        
        subgraph = self.graph.subgraph(cluster_nodes).copy()
        
        print(f"Cluster '{cluster_name}' subgraph:")
        print(f"  Nodes: {len(cluster_nodes)}")
        print(f"  Edges: {subgraph.number_of_edges()}")
        
        return subgraph
