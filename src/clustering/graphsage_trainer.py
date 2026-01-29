"""
GraphSAGE training for agent embeddings.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from typing import Tuple
from pathlib import Path


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model for node embeddings."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            dropout: Dropout rate
        """
        super(GraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        return x


class GraphSAGETrainer:
    """Train GraphSAGE model on agent graph."""
    
    def __init__(
        self,
        data: Data,
        hidden_channels: int = 64,
        out_channels: int = 128,
        learning_rate: float = 0.01,
        dropout: float = 0.1,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            data: PyG Data object
            hidden_channels: Hidden dimension
            out_channels: Output embedding dimension
            learning_rate: Learning rate
            dropout: Dropout rate
            device: Device (cpu/cuda)
        """
        self.data = data
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move data to device
        self.data = self.data.to(self.device)
        
        # Initialize model
        self.model = GraphSAGE(
            in_channels=data.num_node_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=5e-4
        )
        
        print(f"\n✓ GraphSAGE initialized on {self.device}")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"  Input dim: {data.num_node_features}")
        print(f"  Hidden dim: {hidden_channels}")
        print(f"  Output dim: {out_channels}")
    
    def train_unsupervised(
        self,
        epochs: int = 200,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """
        Train GraphSAGE using unsupervised link prediction.
        
        Args:
            epochs: Number of training epochs
            verbose: Print progress
            
        Returns:
            embeddings: Final node embeddings
            losses: Training losses
        """
        print(f"\nTraining GraphSAGE for {epochs} epochs...")
        
        self.model.train()
        losses = []
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(self.data.x, self.data.edge_index)
            
            # Unsupervised loss: maximize similarity for connected nodes
            loss = self._compute_unsupervised_loss(embeddings)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 50 == 0:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"\n✓ Training complete. Final loss: {losses[-1]:.4f}")
        
        # Get final embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.data.x, self.data.edge_index)
        
        return embeddings, losses
    
    def _compute_unsupervised_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute unsupervised loss (link prediction).
        
        Args:
            embeddings: Node embeddings
            
        Returns:
            Loss value
        """
        # Positive samples: actual edges
        edge_index = self.data.edge_index
        pos_src = embeddings[edge_index[0]]
        pos_dst = embeddings[edge_index[1]]
        pos_score = (pos_src * pos_dst).sum(dim=1)
        
        # Negative samples: random node pairs
        num_neg = edge_index.size(1)
        neg_src_idx = torch.randint(0, embeddings.size(0), (num_neg,), device=self.device)
        neg_dst_idx = torch.randint(0, embeddings.size(0), (num_neg,), device=self.device)
        
        neg_src = embeddings[neg_src_idx]
        neg_dst = embeddings[neg_dst_idx]
        neg_score = (neg_src * neg_dst).sum(dim=1)
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def get_embeddings(self) -> np.ndarray:
        """
        Get current node embeddings.
        
        Returns:
            Embeddings as numpy array [num_nodes, embedding_dim]
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.data.x, self.data.edge_index)
        
        return embeddings.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Model loaded from {path}")
