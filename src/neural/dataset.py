"""
PyTorch dataset for cluster-level multimodal transition prediction.
SINGAPORE COVID-19 DATA.

MULTIMODAL STRUCTURE:
- Tabular: [N, 33] cluster features (SINGAPORE: 30 → 33)
- Temporal: [N, 8, 10] cluster time series (7-day lookback + current)
- Graph: [N, 128] cluster graph embeddings
- Labels: [N, 5] transition rates [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]

SINGAPORE ADDITIONS (+3 tabular features):
- Imported case rate
- Quarantine rate
- Dormitory cluster indicator
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional


class TransitionDataset(Dataset):
    """
    Multimodal dataset for cluster-level transition prediction.
    
    Handles normalization and batching for all three modalities.
    UPDATED: Supports both 30-dim (generic) and 33-dim (Singapore) tabular features.
    """
    
    def __init__(
        self,
        tabular: np.ndarray,
        temporal: np.ndarray,
        graph: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            tabular: [N, 30 or 33] cluster features
            temporal: [N, 8, 10] cluster time series
            graph: [N, 128] cluster graph embeddings
            labels: [N, 5] transition rates
            normalize: Whether to normalize features
        """
        self.tabular = torch.FloatTensor(tabular)
        self.temporal = torch.FloatTensor(temporal)
        self.graph = torch.FloatTensor(graph)
        self.labels = torch.FloatTensor(labels)
        
        # Detect dataset type
        self.dataset_type = 'singapore' if self.tabular.shape[1] == 33 else 'generic'
        
        # Normalize each modality separately
        if normalize:
            self._normalize()
        else:
            self.tabular_mean = None
            self.tabular_std = None
            self.temporal_mean = None
            self.temporal_std = None
            self.graph_mean = None
            self.graph_std = None
        
        print(f"✓ TransitionDataset initialized ({self.dataset_type.upper()})")
        print(f"  Samples: {len(self)}")
        print(f"  Tabular: {self.tabular.shape} ({self.tabular.shape[1]} features)")
        print(f"  Temporal: {self.temporal.shape}")
        print(f"  Graph: {self.graph.shape}")
        print(f"  Labels: {self.labels.shape}")
        
        # Print label statistics
        print(f"\n  Label statistics (transition rates):")
        transition_names = ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']
        for i, name in enumerate(transition_names):
            mean = self.labels[:, i].mean().item()
            std = self.labels[:, i].std().item()
            min_val = self.labels[:, i].min().item()
            max_val = self.labels[:, i].max().item()
            print(f"    {name}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    def _normalize(self):
        """Normalize each modality using z-score normalization."""
        
        # === Tabular normalization ===
        self.tabular_mean = self.tabular.mean(dim=0)
        self.tabular_std = self.tabular.std(dim=0)
        
        # ⭐ Handle constant features (std = 0)
        # Don't normalize constant features (keep them as-is)
        constant_mask = self.tabular_std < 1e-6
        if constant_mask.any():
            print(f"  ⚠️ Warning: {constant_mask.sum()} constant tabular features (not normalized)")
            self.tabular_std[constant_mask] = 1.0  # Set to 1.0 to avoid division issues
        
        self.tabular = (self.tabular - self.tabular_mean) / self.tabular_std
        
        # === Temporal normalization ===
        temporal_flat = self.temporal.reshape(-1, self.temporal.shape[-1])
        self.temporal_mean = temporal_flat.mean(dim=0)
        self.temporal_std = temporal_flat.std(dim=0)
        
        constant_mask = self.temporal_std < 1e-6
        if constant_mask.any():
            print(f"  ⚠️ Warning: {constant_mask.sum()} constant temporal features (not normalized)")
            self.temporal_std[constant_mask] = 1.0
        
        self.temporal = (self.temporal - self.temporal_mean.unsqueeze(0).unsqueeze(0)) / \
                        self.temporal_std.unsqueeze(0).unsqueeze(0)
        
        # === Graph normalization ===
        self.graph_mean = self.graph.mean(dim=0)
        self.graph_std = self.graph.std(dim=0)
        
        # ⭐ CRITICAL: Graph embeddings might be constant across clusters
        constant_mask = self.graph_std < 1e-6
        if constant_mask.any():
            print(f"  ⚠️ Warning: {constant_mask.sum()} constant graph features")
            # If ALL graph features are constant, skip normalization entirely
            if constant_mask.all():
                print(f"  ⚠️ ALL graph features constant - skipping graph normalization")
                self.graph_std = torch.ones_like(self.graph_std)
            else:
                self.graph_std[constant_mask] = 1.0
        
        self.graph = (self.graph - self.graph_mean) / self.graph_std
        
        print(f"\n  ✓ Normalized all modalities")
        print(f"    Constant tabular features: {(self.tabular_std == 1.0).sum()}/{len(self.tabular_std)}")
        print(f"    Constant temporal features: {(self.temporal_std == 1.0).sum()}/{len(self.temporal_std)}")
        print(f"    Constant graph features: {(self.graph_std == 1.0).sum()}/{len(self.graph_std)}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - 'tabular': [30 or 33] cluster features
                - 'temporal': [8, 10] time series
                - 'graph': [128] graph embedding
                - 'label': [5] transition rates
        """
        return {
            'tabular': self.tabular[idx],
            'temporal': self.temporal[idx],
            'graph': self.graph[idx],
            'label': self.labels[idx]
        }
    
    def get_transition_statistics(self) -> dict:
        """
        Get statistics for each transition type.
        
        Returns:
            Dictionary with mean, std, min, max for each transition
        """
        transition_names = ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']
        stats = {}
        
        for i, name in enumerate(transition_names):
            stats[name] = {
                'mean': self.labels[:, i].mean().item(),
                'std': self.labels[:, i].std().item(),
                'min': self.labels[:, i].min().item(),
                'max': self.labels[:, i].max().item(),
                'median': self.labels[:, i].median().item()
            }
        
        return stats


def create_dataloaders(
    dataset: TransitionDataset,
    train_split: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset: TransitionDataset instance
        train_split: Fraction of data for training (default 0.8)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


def load_saved_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load saved cluster-level multimodal data.
    
    Args:
        data_dir: Directory containing saved .npy files
        
    Returns:
        (tabular, temporal, graph, labels) tuple
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    tabular = np.load(data_dir / 'cluster_tabular.npy')
    temporal = np.load(data_dir / 'cluster_temporal.npy')
    graph = np.load(data_dir / 'cluster_graph.npy')
    labels = np.load(data_dir / 'cluster_labels.npy')
    
    print(f"✓ Loaded data from {data_dir}")
    print(f"  Tabular: {tabular.shape}")
    print(f"  Temporal: {temporal.shape}")
    print(f"  Graph: {graph.shape}")
    print(f"  Labels: {labels.shape}")
    
    return tabular, temporal, graph, labels


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_class_weights(labels: np.ndarray, method: str = 'inverse_freq') -> torch.Tensor:
    """
    Compute weights for each transition type based on label distribution.
    
    Useful for weighting rare transitions (like I->D) more heavily.
    
    Args:
        labels: [N, 5] transition rates
        method: 'inverse_freq' or 'balanced'
        
    Returns:
        weights: [5,] tensor with weights for each transition
    """
    # Calculate mean transition rate for each type
    mean_rates = labels.mean(axis=0)
    
    if method == 'inverse_freq':
        # Weight inversely proportional to frequency
        # Higher weight for rare transitions
        weights = 1.0 / (mean_rates + 1e-8)
        weights = weights / weights.sum() * 5  # Normalize to sum to 5
        
    elif method == 'balanced':
        # Balanced weighting
        max_rate = mean_rates.max()
        weights = max_rate / (mean_rates + 1e-8)
        weights = weights / weights.sum() * 5
        
    else:
        weights = np.ones(5)
    
    print(f"\n✓ Computed transition weights ({method}):")
    transition_names = ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']
    for i, name in enumerate(transition_names):
        print(f"    {name}: {weights[i]:.3f} (mean rate: {mean_rates[i]:.4f})")
    
    return torch.FloatTensor(weights)


def analyze_dataset(dataset: TransitionDataset):
    """
    Print comprehensive statistics about the dataset.
    
    Args:
        dataset: TransitionDataset instance
    """
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Dataset type: {dataset.dataset_type.upper()}")
    
    # Tabular statistics
    print(f"\nTabular features ({dataset.tabular.shape[1]} dims):")
    print(f"  Mean range: [{dataset.tabular.mean(dim=0).min():.3f}, {dataset.tabular.mean(dim=0).max():.3f}]")
    print(f"  Std range: [{dataset.tabular.std(dim=0).min():.3f}, {dataset.tabular.std(dim=0).max():.3f}]")
    
    # Temporal statistics
    print(f"\nTemporal features ({dataset.temporal.shape[1]} timesteps × {dataset.temporal.shape[2]} dims):")
    print(f"  Mean range: [{dataset.temporal.mean():.3f}]")
    print(f"  Std range: [{dataset.temporal.std():.3f}]")
    
    # Graph statistics
    print(f"\nGraph embeddings ({dataset.graph.shape[1]} dims):")
    print(f"  Mean: {dataset.graph.mean():.3f}")
    print(f"  Std: {dataset.graph.std():.3f}")
    
    # Label statistics
    print(f"\nTransition rates:")
    stats = dataset.get_transition_statistics()
    for name, stat in stats.items():
        print(f"  {name}:")
        print(f"    Mean: {stat['mean']:.4f}, Median: {stat['median']:.4f}")
        print(f"    Std: {stat['std']:.4f}, Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
    
    # Check for any issues
    print(f"\nData quality checks:")
    
    # NaN check
    has_nan_tabular = torch.isnan(dataset.tabular).any().item()
    has_nan_temporal = torch.isnan(dataset.temporal).any().item()
    has_nan_graph = torch.isnan(dataset.graph).any().item()
    has_nan_labels = torch.isnan(dataset.labels).any().item()
    
    if any([has_nan_tabular, has_nan_temporal, has_nan_graph, has_nan_labels]):
        print(f"  ⚠ WARNING: NaN values detected!")
        print(f"    Tabular: {has_nan_tabular}")
        print(f"    Temporal: {has_nan_temporal}")
        print(f"    Graph: {has_nan_graph}")
        print(f"    Labels: {has_nan_labels}")
    else:
        print(f"  ✓ No NaN values")
    
    # Inf check
    has_inf_tabular = torch.isinf(dataset.tabular).any().item()
    has_inf_temporal = torch.isinf(dataset.temporal).any().item()
    has_inf_graph = torch.isinf(dataset.graph).any().item()
    has_inf_labels = torch.isinf(dataset.labels).any().item()
    
    if any([has_inf_tabular, has_inf_temporal, has_inf_graph, has_inf_labels]):
        print(f"  ⚠ WARNING: Inf values detected!")
        print(f"    Tabular: {has_inf_tabular}")
        print(f"    Temporal: {has_inf_temporal}")
        print(f"    Graph: {has_inf_graph}")
        print(f"    Labels: {has_inf_labels}")
    else:
        print(f"  ✓ No Inf values")
    
    # Label range check (should be [0, 1] for probabilities)
    labels_in_range = ((dataset.labels >= 0) & (dataset.labels <= 1)).all().item()
    if labels_in_range:
        print(f"  ✓ All labels in valid range [0, 1]")
    else:
        print(f"  ⚠ WARNING: Some labels outside [0, 1] range!")
        print(f"    Min: {dataset.labels.min():.4f}, Max: {dataset.labels.max():.4f}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the dataset module.
    """
    from pathlib import Path
    
    # Load data
    data_dir = Path("data/processed/neural_training_data")
    
    if data_dir.exists():
        tabular, temporal, graph, labels = load_saved_data(data_dir)
        
        # Create dataset
        dataset = TransitionDataset(
            tabular, temporal, graph, labels,
            normalize=True
        )
        
        # Analyze dataset
        analyze_dataset(dataset)
        
        # Compute class weights
        weights = compute_class_weights(labels, method='inverse_freq')
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset,
            train_split=0.8,
            batch_size=256
        )
        
        # Test batch loading
        print("\nTesting batch loading:")
        batch = next(iter(train_loader))
        print(f"  Batch tabular: {batch['tabular'].shape}")
        print(f"  Batch temporal: {batch['temporal'].shape}")
        print(f"  Batch graph: {batch['graph'].shape}")
        print(f"  Batch labels: {batch['label'].shape}")
        
    else:
        print(f"Data directory not found: {data_dir}")
        print("Run train_neural_pathway.py first to generate data.")
