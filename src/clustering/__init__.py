# """
# Clustering module: Graph construction, GraphSAGE training, and clustering algorithms.
# """

# from .graph_builder import GraphBuilder
# from .graphsage_trainer import GraphSAGE, GraphSAGETrainer

# __all__ = ['GraphBuilder', 'GraphSAGE', 'GraphSAGETrainer']

"""
Clustering module: Graph construction, GraphSAGE training, HSBC³ clustering, and visualization.

Components:
- GraphBuilder: Construct agent contact networks
- GraphSAGE: Graph neural network for node embeddings
- GraphSAGETrainer: Training pipeline for GraphSAGE
- HSBC3Clustering: Novel hierarchical semantic-behavioral contrastive clustering
- ClusterVisualizer: Publication-quality visualization suite
"""

from .graph_builder import GraphBuilder
from .graphsage_trainer import GraphSAGE, GraphSAGETrainer
from .hsbc_clustering import HSBC3Clustering
from .cluster_visualizer import ClusterVisualizer, quick_visualize, generate_publication_figures

__all__ = [
    'GraphBuilder',
    'GraphSAGE', 
    'GraphSAGETrainer',
    'HSBC3Clustering',
    'ClusterVisualizer',
    'quick_visualize',
    'generate_publication_figures'
]
