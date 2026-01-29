"""
Training loop for multimodal neural pathway.

CLUSTER-LEVEL MULTIMODAL MULTI-OUTPUT REGRESSION:
- Input: Tabular, Temporal, Graph modalities (cluster-level)
- Output: 5 transition rates [P(S→E), P(E→I), P(I→R), P(I→D), P(R→S)]
- Loss: Weighted MSE with rare event oversampling
- Metrics: Per-transition MAE, RMSE, R²

HANDLES SPARSE TRANSITIONS: Most rates are 0, so we oversample non-zero examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for handling sparse transition rates.
    Focuses learning on non-zero transitions (rare events).
    """
    
    def __init__(self, transition_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.transition_weights = transition_weights
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch, 5] predicted transition rates
            targets: [batch, 5] actual transition rates
        Returns:
            scalar loss
        """
        # MSE per sample per transition
        mse = (predictions - targets) ** 2  # [batch, 5]
        
        # Focal weight: focus on non-zero targets (rare events)
        # For targets near 0, focal_weight ≈ 1 (normal weight)
        # For targets > 0, focal_weight increases
        focal_weight = 1.0 + self.gamma * targets  # [batch, 5]
        
        # Apply focal weighting
        focal_mse = mse * focal_weight
        
        # Weight by transition importance
        weighted_mse = focal_mse * self.transition_weights.unsqueeze(0)
        
        return weighted_mse.mean()


def create_balanced_sampler(labels: np.ndarray, oversample_factor: float = 3.0):
    """
    Create sampler that oversamples timesteps with transitions.
    
    Args:
        labels: [N, 5] transition rate array
        oversample_factor: How much to oversample non-zero samples (3.0 = 3x)
    
    Returns:
        WeightedRandomSampler
    """
    # Calculate "transition score" for each sample
    # Higher score = more transitions happening
    transition_scores = labels.sum(axis=1)  # [N,]
    
    # Create sample weights
    weights = np.ones(len(labels))
    
    # Oversample samples with any transitions
    has_transition = transition_scores > 0.001
    weights[has_transition] *= oversample_factor
    
    # Extra boost for samples with multiple transitions
    multiple_transitions = transition_scores > 0.01
    weights[multiple_transitions] *= 1.5
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    print(f"  Created balanced sampler:")
    print(f"    Samples with transitions: {has_transition.sum()} ({has_transition.mean()*100:.1f}%)")
    print(f"    Oversample factor: {oversample_factor}x")
    
    return sampler


class MultimodalTrainer:
    """
    Trainer for multimodal cluster-level multi-output regression.
    
    Features:
    - Multi-output regression (5 transition rates)
    - Focal loss for sparse transitions
    - Balanced sampling (oversample non-zero transitions)
    - Early stopping based on validation loss
    - Learning rate scheduling
    - Gradient clipping
    - Per-transition metrics (MAE, RMSE, R²)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        transition_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: MultimodalEncoder instance
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            transition_weights: Optional weights for each transition [5,]
            use_focal_loss: Use focal loss to handle sparse transitions
            focal_gamma: Focal loss gamma parameter (higher = more focus on non-zero)
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        if transition_weights is not None:
            self.transition_weights = transition_weights.to(device)
        else:
            # Default: emphasize S->E and I->D
            self.transition_weights = torch.tensor([5.0, 2.0, 1.5, 3.0, 1.0]).to(device)
        
        if use_focal_loss:
            self.criterion = FocalMSELoss(self.transition_weights, gamma=focal_gamma)
            print(f"  Using Focal MSE Loss (gamma={focal_gamma})")
        else:
            self.criterion = WeightedMSELoss(self.transition_weights)
            print(f"  Using Weighted MSE Loss")
        
        print(f"  Transition weights: {self.transition_weights.tolist()}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Transition names
        self.transition_names = ['S->E', 'E->I', 'I->R', 'I->D', 'R->S']
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Per-transition metrics
        for name in self.transition_names:
            self.history[f'train_mae_{name}'] = []
            self.history[f'val_mae_{name}'] = []
            self.history[f'val_rmse_{name}'] = []
            self.history[f'val_r2_{name}'] = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print(f"✓ Trainer initialized (Transition Rate Regression)")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Task: Predict 5 transition RATES (continuous values [0,1])")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tabular = batch['tabular'].to(self.device)
            temporal = batch['temporal'].to(self.device)
            graph = batch['graph'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(tabular, temporal, graph)  # [batch, 5]
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Monitor gradients on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                print(f"\n    Gradient analysis (first batch):")
                if total_norm < 1e-7:
                    print(f"      ⚠️  WARNING: Vanishing gradients (norm={total_norm:.2e})")
                elif total_norm > 100:
                    print(f"      ⚠️  WARNING: Exploding gradients (norm={total_norm:.2e})")
                else:
                    print(f"      ✓ Gradient norm: {total_norm:.4f}")
                
                # Check prediction range
                pred_min, pred_max = predictions.min().item(), predictions.max().item()
                print(f"      Prediction range: [{pred_min:.4f}, {pred_max:.4f}]")
                print(f"      Target range: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * len(labels)
            
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Calculate epoch metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        avg_loss = total_loss / len(all_targets)
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Per-transition MAE
        per_transition_mae = {}
        for i, name in enumerate(self.transition_names):
            per_transition_mae[name] = mean_absolute_error(
                all_targets[:, i], all_predictions[:, i]
            )
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'per_transition_mae': per_transition_mae
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                tabular = batch['tabular'].to(self.device)
                temporal = batch['temporal'].to(self.device)
                graph = batch['graph'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(tabular, temporal, graph)
                
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item() * len(labels)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        avg_loss = total_loss / len(all_targets)
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Per-transition metrics
        per_transition_metrics = {}
        for i, name in enumerate(self.transition_names):
            y_true = all_targets[:, i]
            y_pred = all_predictions[:, i]
            
            mae_i = mean_absolute_error(y_true, y_pred)
            rmse_i = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # R² can be undefined if all targets are identical
            try:
                if y_true.std() > 1e-10:
                    r2_i = r2_score(y_true, y_pred)
                else:
                    r2_i = -999.0  # Undefined
            except:
                r2_i = -999.0
            
            per_transition_metrics[name] = {
                'mae': mae_i,
                'rmse': rmse_i,
                'r2': r2_i
            }
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'per_transition_metrics': per_transition_metrics
        }
    
    def analyze_predictions(self, val_loader: DataLoader) -> Dict:
        """Analyze prediction distributions to detect issues."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular'].to(self.device)
                temporal = batch['temporal'].to(self.device)
                graph = batch['graph'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(tabular, temporal, graph)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        analysis = {}
        
        for i, name in enumerate(self.transition_names):
            preds = all_predictions[:, i]
            targets = all_targets[:, i]
            
            analysis[name] = {
                'pred_mean': float(preds.mean()),
                'pred_std': float(preds.std()),
                'pred_min': float(preds.min()),
                'pred_max': float(preds.max()),
                'target_mean': float(targets.mean()),
                'target_std': float(targets.std()),
                'target_min': float(targets.min()),
                'target_max': float(targets.max()),
                'target_nonzero_frac': float((targets > 0.001).mean()),
                'target_variance': float(targets.var())
            }
        
        return analysis
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: Optional[Path] = None
    ):
        """Full training loop with early stopping."""
        print(f"\n{'='*80}")
        print("STARTING TRAINING (Cluster-Level Transition Rate Regression)")
        print(f"{'='*80}\n")
        
        # Sanity check: verify training data has non-zero targets
        print("Checking training data...")
        sample_batch = next(iter(train_loader))
        sample_labels = sample_batch['label']
        
        print(f"  Shapes:")
        print(f"    Tabular: {sample_batch['tabular'].shape}")
        print(f"    Temporal: {sample_batch['temporal'].shape}")
        print(f"    Graph: {sample_batch['graph'].shape}")
        print(f"    Labels: {sample_labels.shape}")
        
        print(f"\n  Target transition rate statistics:")
        print(f"  {'Transition':<10} {'Mean':<12} {'Std':<12} {'NonZero%':<12} {'Max':<12}")
        print(f"  {'-'*60}")
        
        all_zero = True
        for i, name in enumerate(self.transition_names):
            trans_labels = sample_labels[:, i]
            nonzero = (trans_labels > 0.001).sum().item()
            nonzero_pct = nonzero / len(trans_labels) * 100
            
            print(f"  {name:<10} {trans_labels.mean():<12.6f} {trans_labels.std():<12.6f} "
                  f"{nonzero_pct:<12.1f} {trans_labels.max():<12.6f}")
            
            if trans_labels.sum() > 0.001:
                all_zero = False
            
            if trans_labels.sum() < 0.0001:
                print(f"             ⚠️  WARNING: {name} targets are all ZERO!")
        
        if all_zero:
            print(f"\n  ❌ CRITICAL ERROR: ALL TRANSITION RATES ARE ZERO!")
            print(f"     Your simulation is not generating any state transitions.")
            print(f"     Check _run_singapore_seird_simulation() in cluster_data_generator.py")
            return
        
        print()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch=epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Debug: Analyze predictions on first 3 epochs
            if epoch < 3:
                print(f"\n  PREDICTION ANALYSIS (Epoch {epoch+1}):")
                pred_analysis = self.analyze_predictions(val_loader)
                
                print(f"  {'Transition':<10} {'Pred μ':<12} {'Pred σ':<12} {'Target μ':<12} {'Status':<20}")
                print(f"  {'-'*75}")
                
                for name in self.transition_names:
                    stats = pred_analysis[name]
                    
                    status = "✓"
                    if stats['pred_std'] < 0.0001:
                        status = "⚠️ Constant preds"
                    elif stats['target_variance'] < 1e-10:
                        status = "⚠️ No target variance"
                    elif abs(stats['pred_mean'] - stats['target_mean']) > 0.2:
                        status = "⚠️ Mean mismatch"
                    
                    print(f"  {name:<10} {stats['pred_mean']:<12.6f} {stats['pred_std']:<12.6f} "
                          f"{stats['target_mean']:<12.6f} {status:<20}")
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rate'].append(current_lr)
            
            # Per-transition metrics
            for name in self.transition_names:
                self.history[f'train_mae_{name}'].append(
                    train_metrics['per_transition_mae'][name]
                )
                self.history[f'val_mae_{name}'].append(
                    val_metrics['per_transition_metrics'][name]['mae']
                )
                self.history[f'val_rmse_{name}'].append(
                    val_metrics['per_transition_metrics'][name]['rmse']
                )
                self.history[f'val_r2_{name}'].append(
                    val_metrics['per_transition_metrics'][name]['r2']
                )
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.6f} | MAE: {train_metrics['mae']:.6f}")
            print(f"  Val   Loss: {val_metrics['loss']:.6f} | MAE: {val_metrics['mae']:.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            print(f"\n  Per-Transition Validation Metrics:")
            print(f"  {'Transition':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
            print(f"  {'-'*50}")
            for name in self.transition_names:
                metrics = val_metrics['per_transition_metrics'][name]
                r2_str = f"{metrics['r2']:.4f}" if metrics['r2'] > -100 else "undefined"
                print(f"  {name:<10} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} {r2_str:<12}")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                
                if save_dir:
                    self.save_checkpoint(save_dir / 'best_model.pt', epoch, val_metrics)
                    print(f"  ✓ New best model saved (Loss={val_metrics['loss']:.6f})!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"{'='*80}")
                break
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_dir / 'final_model.pt', epochs, val_metrics)
            self.save_history(save_dir / 'training_history.json')
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        metrics_to_save = {'loss': metrics['loss'], 'mae': metrics['mae']}
        
        for name in self.transition_names:
            metrics_to_save[f'{name}_mae'] = metrics['per_transition_metrics'][name]['mae']
            metrics_to_save[f'{name}_rmse'] = metrics['per_transition_metrics'][name]['rmse']
            metrics_to_save[f'{name}_r2'] = metrics['per_transition_metrics'][name]['r2']
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics_to_save,
            'history': self.history,
            'transition_weights': self.transition_weights.tolist()
        }, path)
    
    def save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', {})
        
        print(f"✓ Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['metrics']['loss']:.6f}")
        print(f"  Val MAE: {checkpoint['metrics']['mae']:.6f}")


# Keep old class for backwards compatibility
class WeightedMSELoss(nn.Module):
    """MSE Loss with per-sample and per-transition weighting."""
    
    def __init__(self, transition_weights: torch.Tensor):
        super().__init__()
        self.transition_weights = transition_weights
    
    def forward(self, predictions, targets):
        mse = (predictions - targets) ** 2
        weighted_mse = mse * self.transition_weights.unsqueeze(0)
        return weighted_mse.mean()
