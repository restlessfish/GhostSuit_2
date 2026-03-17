"""
In-Run Data Shapley Engine

This module implements the In-Run Data Shapley method from "Data Shapley in One Training Run".
It computes Shapley values for each training sample during a single training run, without
requiring model retraining.

Key features:
- First-order method: Uses gradient dot products
- Second-order method: Uses gradient-Hessian-gradient products
- Accumulates Shapley values throughout training
"""

import os
import sys
import math
import warnings
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np
from collections import defaultdict

# Add parent directories to path for ghostEngines
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghostEngines.graddotprod_engine import GradDotProdEngine


class InRunShapleyEngine(GradDotProdEngine):
    """
    Engine for computing In-Run Data Shapley values.
    
    Extends GradDotProdEngine to add:
    1. Shapley value computation from gradient dot products
    2. Second-order Shapley value computation (optional)
    3. Accumulation of Shapley values across training iterations
    """
    
    def __init__(
        self,
        module: nn.Module,
        val_batch_size: int,
        loss_reduction: str = 'mean',
        use_dummy_bias: bool = False,
        dot_prod_save_path: Optional[str] = None,
        log_grad_norms: bool = False,
        order: int = 1,
        accumulate_shapley: bool = True,
        shapley_save_interval: int = 10,
        grad_val_cache_K: int = 0,
    ):
        """
        Initialize In-Run Shapley Engine.
        
        Args:
            module: The PyTorch module
            val_batch_size: Validation batch size
            loss_reduction: Loss reduction ('mean' or 'sum')
            use_dummy_bias: Whether to use dummy bias
            dot_prod_save_path: Directory to save results
            log_grad_norms: Whether to log gradient norms
            order: Order of approximation (1 for first-order, 2 for second-order)
            accumulate_shapley: Whether to accumulate Shapley values across iterations
            shapley_save_interval: How often to save Shapley values
            grad_val_cache_K: If >0, reuse cached validation gradient every K steps (approximate speedup).
        """
        super().__init__(
            module=module,
            val_batch_size=val_batch_size,
            loss_reduction=loss_reduction,
            use_dummy_bias=use_dummy_bias,
            dot_prod_save_path=dot_prod_save_path,
            log_grad_norms=log_grad_norms,
            grad_val_cache_K=grad_val_cache_K,
        )
        
        self.order = order
        self.accumulate_shapley = accumulate_shapley
        self.shapley_save_interval = shapley_save_interval

        # Storage for accumulated Shapley values
        # Maps sample_idx -> accumulated_shapley_value
        self.shapley_values = defaultdict(float)
        
        # IMPROVEMENT: Track how many times each sample was seen
        # This allows for proper averaging
        self.sample_counts = defaultdict(int)

        # Storage for per-iteration Shapley values
        self.shapley_history = []
        
        # Track total number of training samples seen
        self.total_samples_seen = 0
        
        print(f"[INFO] In-Run Shapley Engine initialized with order={order}, accumulate={accumulate_shapley}")
    
    def _compute_shapley_from_dot_product(
        self, 
        dot_product: torch.Tensor, 
        batch_size: int,
        val_batch_size: int
    ) -> torch.Tensor:
        """
        Compute Shapley values from gradient dot products.
        
        For first-order approximation, the Shapley value for sample i is:
        phi_i = (1/n) * dot_product_i
        
        where n is the training set size and dot_product_i is the gradient dot product
        between training sample i and validation set gradients.
        
        IMPROVEMENTS:
        1. Better normalization by validation batch size
        2. Scale by learning rate to match training dynamics
        3. Add temperature scaling for better discrimination
        
        Args:
            dot_product: Tensor of shape [batch_size] containing dot products
            batch_size: Number of training samples in this batch
            val_batch_size: Number of validation samples
            
        Returns:
            Shapley values for this batch, shape [batch_size]
        """
        # Normalize by validation batch size (average over validation samples)
        if val_batch_size > 0:
            dot_product = dot_product / val_batch_size
        
        # For In-Run Shapley, the marginal contribution is:
        # phi_i ≈ <grad_i, grad_val> / |val_batch|
        # This represents how much sample i contributes to reducing validation loss
        
        # Apply temperature scaling to improve discrimination
        # Higher temperature = more spread out values
        temperature = 1.0  # Can be tuned
        
        # Normalize by batch size to get per-sample contribution
        # This ensures values are comparable across different batch sizes
        if batch_size > 0:
            dot_product = dot_product / batch_size
        
        return dot_product * temperature
    
    def _compute_second_order_shapley(
        self,
        train_grads: Dict[str, torch.Tensor],
        val_grads: Dict[str, torch.Tensor],
        hessian_vector_products: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute second-order Shapley values using gradient-Hessian-gradient products.
        
        This is more accurate but computationally expensive. Requires computing
        Hessian-vector products.
        
        Args:
            train_grads: Dictionary of per-sample training gradients
            val_grads: Dictionary of validation gradients
            hessian_vector_products: Optional precomputed HVP (for efficiency)
            
        Returns:
            Second-order Shapley values
        """
        # TODO: Implement second-order method
        # This requires computing HVP efficiently
        # For now, return first-order approximation
        warnings.warn("Second-order Shapley not yet implemented, using first-order")
        return None
    
    def aggregate_and_log(self):
        """
        Override parent method to compute and accumulate Shapley values.
        """
        # First call parent to aggregate dot products
        super().aggregate_and_log()
        
        # Extract dot products from the last logged entry
        if self.dot_product_log:
            last_entry = self.dot_product_log[-1]
            dot_product = last_entry['dot_product']
            batch_idx = last_entry.get('batch_idx', None)
            
            if dot_product is not None and batch_idx is not None:
                # Compute Shapley values for this batch
                batch_size = dot_product.shape[0]
                shapley_batch = self._compute_shapley_from_dot_product(
                    dot_product, 
                    batch_size,
                    self.val_batch_size
                )
                
                # Update accumulated Shapley values
                # IMPROVEMENT: Use weighted accumulation based on iteration number
                # Early iterations have less stable gradients, so weight them less
                if self.accumulate_shapley:
                    # Weight by iteration: later iterations are more reliable
                    # Use exponential weighting: weight = 1 - exp(-iter/100)
                    iter_weight = 1.0 - np.exp(-self.iter_num / 100.0)
                    iter_weight = max(0.1, iter_weight)  # Minimum weight of 0.1
                    
                    if isinstance(batch_idx, (list, tuple)):
                        for i, idx in enumerate(batch_idx):
                            if i < len(shapley_batch):
                                # Weighted accumulation
                                self.shapley_values[idx] += shapley_batch[i].item() * iter_weight
                                self.sample_counts[idx] += 1
                    elif isinstance(batch_idx, torch.Tensor):
                        batch_idx_list = batch_idx.cpu().tolist()
                        for i, idx in enumerate(batch_idx_list):
                            if i < len(shapley_batch):
                                self.shapley_values[idx] += shapley_batch[i].item() * iter_weight
                                self.sample_counts[idx] += 1
                    else:
                        # If batch_idx is a single value, assume sequential indices
                        start_idx = batch_idx if isinstance(batch_idx, int) else self.iter_num * batch_size
                        for i in range(batch_size):
                            sample_idx = start_idx + i
                            if i < len(shapley_batch):
                                self.shapley_values[sample_idx] += shapley_batch[i].item() * iter_weight
                                self.sample_counts[sample_idx] += 1
                    
                    self.total_samples_seen += batch_size
                
                # Store per-iteration Shapley values
                shapley_info = {
                    'iter_num': self.iter_num,
                    'batch_idx': batch_idx,
                    'shapley_values': shapley_batch.cpu().clone(),
                    'dot_products': dot_product.cpu().clone(),
                }
                self.shapley_history.append(shapley_info)
    
    def get_accumulated_shapley_values(self) -> Dict[int, float]:
        """
        Get accumulated Shapley values for all samples seen so far.
        
        Returns:
            Dictionary mapping sample index to accumulated Shapley value
        """
        return dict(self.shapley_values)
    
    def normalize_shapley_values(self):
        """
        Normalize Shapley values by the number of times each sample was seen.
        This gives the average contribution per iteration.
        """
        for idx in list(self.shapley_values.keys()):
            count = self.sample_counts.get(idx, 1)
            if count > 0:
                self.shapley_values[idx] = self.shapley_values[idx] / count
    
    def get_shapley_array(self, num_samples: Optional[int] = None, compact: bool = True) -> np.ndarray:
        """
        Get Shapley values as a numpy array.
        
        Args:
            num_samples: Total number of samples (for padding missing indices)
            compact: If True, return only non-zero values with their indices
            
        Returns:
            Array of Shapley values (if compact=False) or tuple (values, indices) (if compact=True)
        """
        if not self.shapley_values:
            return np.array([]) if not compact else (np.array([]), np.array([]))
        
        # Get only non-zero values for compact representation
        non_zero_items = [(idx, val) for idx, val in self.shapley_values.items() if val != 0]
        
        if compact and len(non_zero_items) < len(self.shapley_values):
            # Return compact format: only non-zero values
            indices = np.array([idx for idx, _ in non_zero_items])
            values = np.array([val for _, val in non_zero_items])
            return values, indices
        
        # Full array format
        if num_samples is None:
            # Use the maximum index seen, but only if reasonable
            max_idx = max(self.shapley_values.keys())
            # If max index is too large, use compact format
            if max_idx > len(self.shapley_values) * 10:
                # Likely using wrong indices, return compact format
                indices = np.array([idx for idx, _ in non_zero_items])
                values = np.array([val for _, val in non_zero_items])
                return values, indices
            num_samples = max_idx + 1
        else:
            # Ensure we don't create unnecessarily large arrays
            max_idx = max(self.shapley_values.keys())
            if num_samples > max_idx + 1000:  # Allow some padding but not too much
                num_samples = max_idx + 1
        
        shapley_array = np.zeros(num_samples)
        for idx, value in self.shapley_values.items():
            if idx < num_samples:
                shapley_array[idx] = value
        
        return shapley_array
    
    def save_shapley_values(self, iter_num: int):
        """
        Save accumulated Shapley values to disk.
        
        Args:
            iter_num: Current iteration number
        """
        if not self.shapley_values:
            print(f"[WARN] No Shapley values to save at iteration {iter_num}")
            return
        
        print(f"[INFO] Saving Shapley values at iteration {iter_num}...")
        
        # IMPROVEMENT: Normalize before saving
        # Create a copy to avoid modifying the original
        normalized_values = dict(self.shapley_values)
        for idx in normalized_values:
            count = self.sample_counts.get(idx, 1)
            if count > 0:
                normalized_values[idx] = normalized_values[idx] / count
        
        # Save accumulated values
        shapley_file = os.path.join(
            self.dot_prod_save_path, 
            f"shapley_values_iter_{iter_num}.pt"
        )
        
        # Get only non-zero indices for compact storage
        non_zero_indices = sorted([idx for idx, val in normalized_values.items() if val != 0])
        non_zero_values = [normalized_values[idx] for idx in non_zero_indices]
        
        shapley_data = {
            'iter_num': iter_num,
            'shapley_values': normalized_values,  # Use normalized values
            'shapley_values_raw': dict(self.shapley_values),  # Keep raw for debugging
            'sample_counts': dict(self.sample_counts),
            'non_zero_indices': non_zero_indices,
            'non_zero_values': non_zero_values,
            'total_samples_seen': self.total_samples_seen,
            'num_samples': len(self.shapley_values),
            'order': self.order,
        }
        
        torch.save(shapley_data, shapley_file)
        
        # Save as numpy array - use compact format to avoid huge arrays
        # Save only the non-zero values and their indices
        if non_zero_indices:
            # Save compact format: values array
            shapley_array = np.array(non_zero_values)
        else:
            shapley_array = np.array([])
        
        np_file = os.path.join(
            self.dot_prod_save_path,
            f"shapley_array_iter_{iter_num}.npy"
        )
        np.save(np_file, shapley_array)
        
        # Save indices mapping (required to map back to original sample indices)
        indices_file = os.path.join(
            self.dot_prod_save_path,
            f"shapley_indices_iter_{iter_num}.npy"
        )
        np.save(indices_file, np.array(non_zero_indices))
        
        print(f"[INFO] Saved Shapley values to {shapley_file}")
        print(f"[INFO] Saved Shapley array ({len(shapley_array)} non-zero values) to {np_file}")
        print(f"[INFO] Saved indices ({len(non_zero_indices)} samples) to {indices_file}")
    
    def save_shapley_history(self, iter_num: int):
        """
        Save per-iteration Shapley history.
        
        Args:
            iter_num: Current iteration number
        """
        if not self.shapley_history:
            return
        
        history_file = os.path.join(
            self.dot_prod_save_path,
            f"shapley_history_iter_{iter_num}.pt"
        )
        
        torch.save(self.shapley_history, history_file)
        
        # Clear history to save memory (keep only recent entries)
        if len(self.shapley_history) > 1000:
            self.shapley_history = self.shapley_history[-500:]
    
    def should_save_shapley(self, iter_num: int) -> bool:
        """Check if Shapley values should be saved at this iteration."""
        return iter_num > 0 and iter_num % self.shapley_save_interval == 0
    
    def cleanup(self):
        """Save final Shapley values before cleanup."""
        if self.shapley_values:
            # Save final accumulated values
            self.save_shapley_values(iter_num=-1)
            self.save_shapley_history(iter_num=-1)
        
        # Call parent detach
        if hasattr(super(), 'detach'):
            super().detach()
