#!/usr/bin/env python3
"""
K-Value AUROC Evaluation Script.

This script evaluates mislabeled data detection performance (AUROC) for different
gradient cache K values: K=2, K=5, K=10.

The K parameter controls how often the validation gradient is recomputed:
- K=0: Always recompute (no caching)
- K>0: Cache validation gradient for K steps

Usage:
    python evaluate_auroc_by_k.py --result_dir RESULT_DIR [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, List, Optional

import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_shapley_values(result_dir: str, iteration: Optional[int] = None) -> tuple:
    """
    Load Shapley values from result directory.
    
    Returns:
        Tuple of (shapley_values, indices) or (shapley_array, None)
    """
    grad_dir = os.path.join(result_dir, 'grad_dotprods')
    
    if not os.path.exists(grad_dir):
        raise FileNotFoundError(f"grad_dotprods directory not found: {grad_dir}")
    
    # Find shapley files
    shapley_files = [f for f in os.listdir(grad_dir) if f.startswith('shapley_array_iter_')]
    
    if not shapley_files:
        raise FileNotFoundError(f"No shapley_array_iter_*.npy files found in {grad_dir}")
    
    # Get specific iteration or latest
    if iteration is not None:
        shapley_file = os.path.join(grad_dir, f'shapley_array_iter_{iteration}.npy')
        if not os.path.exists(shapley_file):
            raise FileNotFoundError(f"File not found: {shapley_file}")
    else:
        # Find latest
        iterations = [int(f.split('_')[-1].split('.')[0]) for f in shapley_files]
        latest_iter = max(iterations)
        shapley_file = os.path.join(grad_dir, f'shapley_array_iter_{latest_iter}.npy')
        iteration = latest_iter
    
    shapley_array = np.load(shapley_file)
    
    # Try to load indices
    indices_file = os.path.join(grad_dir, f'shapley_indices_iter_{iteration}.npy')
    indices = None
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    
    return shapley_array, indices, iteration


def load_mislabeled_indices(result_dir: str) -> np.ndarray:
    """Load mislabeled sample indices."""
    file_path = os.path.join(result_dir, 'mislabeled_indices.npy')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mislabeled indices not found: {file_path}")
    
    return np.load(file_path)


def mislabeled_data_detection(shapley_values: np.ndarray, 
                              mislabeled_indices: np.ndarray,
                              num_samples: int = 50000) -> Dict:
    """
    Evaluate mislabeled data detection using Shapley values.
    
    Args:
        shapley_values: Array of Shapley values (can be sparse)
        mislabeled_indices: Indices of mislabeled samples
        num_samples: Total number of samples (for dense array reconstruction)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Reconstruct dense array if sparse
    if len(shapley_values) < num_samples:
        # Sparse representation
        dense = np.zeros(num_samples, dtype=shapley_values.dtype)
        if hasattr(shapley_values, 'shape') and len(shapley_values.shape) == 1:
            # Already a values array - need indices
            pass
        else:
            dense = shapley_values
    else:
        dense = shapley_values
    
    # Create binary labels: 1 for mislabeled, 0 for correctly labeled
    binary_labels = np.zeros(len(dense))
    binary_labels[mislabeled_indices] = 1
    
    # Lower Shapley values indicate potentially mislabeled data
    # So we use negative Shapley values as scores
    scores = -dense
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(binary_labels, scores)
    except ValueError as e:
        print(f"Warning: AUROC computation failed: {e}")
        auroc = 0.5
    
    # Compute PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(binary_labels, scores)
        pr_auc = auc(recall, precision)
    except ValueError as e:
        print(f"Warning: PR-AUC computation failed: {e}")
        pr_auc = 0.0
    
    # Compute accuracy at optimal threshold
    threshold = np.median(scores)
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == binary_labels)
    
    # Top-K recall
    sorted_indices = np.argsort(scores)
    num_mislabeled = len(mislabeled_indices)
    
    top_k_recalls = {}
    for k in [10, 20, 50, 100]:
        top_k_indices = sorted_indices[:k]
        detected = len(set(top_k_indices) & set(mislabeled_indices))
        recall = detected / num_mislabeled if num_mislabeled > 0 else 0
        top_k_recalls[f'top_{k}'] = {
            'detected': detected,
            'total': k,
            'recall': recall
        }
    
    return {
        'auroc': float(auroc),
        'pr_auc': float(pr_auc),
        'accuracy': float(accuracy),
        'threshold': float(threshold),
        'num_samples': len(dense),
        'num_mislabeled': int(num_mislabeled),
        'top_k_recalls': top_k_recalls,
    }


def cluster_smoothing(shapley_values: np.ndarray, 
                     gradients: Optional[np.ndarray] = None,
                     alpha: float = 0.2,
                     k_neighbors: int = 10) -> np.ndarray:
    """
    Apply cluster smoothing post-processing.
    
    Args:
        shapley_values: Raw Shapley values
        gradients: Gradient features for similarity computation
        alpha: Smoothing parameter (0-1)
        k_neighbors: Number of nearest neighbors
        
    Returns:
        Smoothed Shapley values
    """
    # For this simplified version, we use a moving average approximation
    # In practice, you would use gradient cosine similarity for neighbor weighting
    
    n = len(shapley_values)
    smoothed = np.zeros_like(shapley_values)
    
    # Simple implementation: weighted local average
    for i in range(n):
        # Get local window
        start = max(0, i - k_neighbors)
        end = min(n, i + k_neighbors + 1)
        
        local_vals = shapley_values[start:end]
        local_weights = np.exp(-np.abs(np.arange(start, end) - i) / k_neighbors)
        
        smoothed[i] = (1 - alpha) * shapley_values[i] + alpha * np.average(local_vals, weights=local_weights)
    
    return smoothed


def evaluate_with_smoothing(shapley_values: np.ndarray,
                           mislabeled_indices: np.ndarray,
                           num_samples: int = 50000) -> Dict:
    """Evaluate with and without cluster smoothing."""
    # Raw evaluation
    raw_results = mislabeled_data_detection(shapley_values, mislabeled_indices, num_samples)
    
    # Smoothed evaluation (with alpha=0.2 as in paper)
    smoothed_values = cluster_smoothing(shapley_values, alpha=0.2)
    smoothed_results = mislabeled_data_detection(smoothed_values, mislabeled_indices, num_samples)
    
    return {
        'raw': raw_results,
        'smoothed': smoothed_results,
    }


def compare_k_values(result_dirs: Dict[int, str], 
                     num_samples: int = 50000) -> Dict:
    """
    Compare AUROC across different K values.
    
    Args:
        result_dirs: Dictionary mapping K values to result directories
        num_samples: Total number of samples
        
    Returns:
        Comparison results
    """
    results = {}
    
    for k, result_dir in result_dirs.items():
        print(f"\nEvaluating K={k}...")
        print(f"  Directory: {result_dir}")
        
        try:
            shapley_values, indices, iteration = load_shapley_values(result_dir)
            mislabeled = load_mislabeled_indices(result_dir)
            
            print(f"  Iteration: {iteration}")
            print(f"  Shapley values shape: {shapley_values.shape}")
            print(f"  Mislabeled samples: {len(mislabeled)}")
            
            # Evaluate
            eval_results = evaluate_with_smoothing(shapley_values, mislabeled, num_samples)
            
            results[f'K_{k}'] = {
                'k_value': k,
                'result_dir': result_dir,
                'iteration': iteration,
                'auroc_raw': eval_results['raw']['auroc'],
                'auroc_smoothed': eval_results['smoothed']['auroc'],
                'pr_auc_raw': eval_results['raw']['pr_auc'],
                'pr_auc_smoothed': eval_results['smoothed']['pr_auc'],
                'accuracy_raw': eval_results['raw']['accuracy'],
                'accuracy_smoothed': eval_results['smoothed']['accuracy'],
            }
            
            print(f"  AUROC (raw): {eval_results['raw']['auroc']:.4f}")
            print(f"  AUROC (smoothed): {eval_results['smoothed']['auroc']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f'K_{k}'] = {
                'k_value': k,
                'error': str(e),
            }
    
    return results


def print_comparison_table(results: Dict):
    """Print comparison table for K values."""
    print("\n" + "=" * 80)
    print("K-Value Comparison: AUROC Results")
    print("=" * 80)
    print(f"{'K Value':<10} {'AUROC (raw)':<15} {'AUROC (smoothed)':<20} {'PR-AUC (smoothed)':<20}")
    print("-" * 80)
    
    for k_label, result in sorted(results.items(), key=lambda x: x[1].get('k_value', -1)):
        if 'error' not in result:
            print(f"K={result['k_value']:<5} {result['auroc_raw']:<15.4f} "
                  f"{result['auroc_smoothed']:<20.4f} {result['pr_auc_smoothed']:<20.4f}")
        else:
            print(f"{k_label:<10} ERROR: {result['error']}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AUROC by K-Value')
    parser.add_argument('--result_dir', type=str, default=None,
                       help='Single result directory (for one K value)')
    parser.add_argument('--k2_dir', type=str, default=None,
                       help='Result directory for K=2')
    parser.add_argument('--k5_dir', type=str, default=None,
                       help='Result directory for K=5')
    parser.add_argument('--k10_dir', type=str, default=None,
                       help='Result directory for K=10')
    parser.add_argument('--output', type=str, default='./auroc_results',
                       help='Output directory')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Specific iteration to evaluate')
    parser.add_argument('--num_samples', type=int, default=50000,
                       help='Total number of training samples')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Cluster smoothing alpha parameter')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 80)
    print("AUROC Evaluation by K-Value")
    print("=" * 80)
    
    # Prepare result directories
    result_dirs = {}
    
    if args.result_dir:
        # Single result directory
        result_dirs[0] = args.result_dir
    else:
        # Multiple K values
        if args.k2_dir:
            result_dirs[2] = args.k2_dir
        if args.k5_dir:
            result_dirs[5] = args.k5_dir
        if args.k10_dir:
            result_dirs[10] = args.k10_dir
    
    if not result_dirs:
        print("Error: Please provide either --result_dir or K-specific directories")
        sys.exit(1)
    
    # Run evaluation
    results = compare_k_values(result_dirs, args.num_samples)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    output_file = os.path.join(args.output, 'k_value_auroc_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
