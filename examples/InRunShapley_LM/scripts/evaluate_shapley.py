"""
Evaluation script for In-Run Data Shapley values.

This script provides evaluation functions for:
1. Mislabeled data detection
2. Data selection based on Shapley values
3. Analysis of Shapley value distributions
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import argparse
import json
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_shapley_values(shapley_file: str) -> Dict:
    """
    Load Shapley values from saved file.
    
    Args:
        shapley_file: Path to saved Shapley values file
        
    Returns:
        Dictionary containing Shapley values and metadata
    """
    data = torch.load(shapley_file, map_location='cpu')
    return data


def load_shapley_array(npy_file: str) -> np.ndarray:
    """
    Load Shapley values as numpy array.
    
    Args:
        npy_file: Path to .npy file
        
    Returns:
        Array of Shapley values
    """
    return np.load(npy_file)


def mislabeled_data_detection(
    shapley_values: np.ndarray,
    true_labels: np.ndarray,
    mislabeled_indices: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate mislabeled data detection using Shapley values.
    
    Args:
        shapley_values: Array of Shapley values for each sample
        true_labels: True labels (0 for correct, 1 for mislabeled)
        mislabeled_indices: Indices of mislabeled samples
        
    Returns:
        Dictionary with evaluation metrics (AUROC, etc.)
    """
    # Create binary labels: 1 for mislabeled, 0 for correctly labeled
    binary_labels = np.zeros(len(shapley_values))
    binary_labels[mislabeled_indices] = 1
    
    # Lower Shapley values indicate potentially mislabeled data
    # So we use negative Shapley values as scores
    scores = -shapley_values
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(binary_labels, scores)
    except ValueError:
        # Handle case where all labels are the same
        auroc = 0.5
    
    # Compute accuracy at optimal threshold
    threshold = np.median(scores)
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(binary_labels, predictions)
    
    return {
        'auroc': float(auroc),
        'accuracy': float(accuracy),
        'threshold': float(threshold)
    }


def data_selection_evaluation(
    shapley_values: np.ndarray,
    test_labels: np.ndarray,
    selection_budgets: List[float] = [0.2, 0.4, 0.6, 0.8]
) -> Dict[str, float]:
    """
    Evaluate data selection based on Shapley values.
    
    Args:
        shapley_values: Array of Shapley values
        test_labels: Test labels for evaluation
        selection_budgets: List of selection budgets (fractions)
        
    Returns:
        Dictionary with test accuracies for each budget
    """
    results = {}
    
    # Sort indices by Shapley values (descending)
    sorted_indices = np.argsort(shapley_values)[::-1]
    
    for budget in selection_budgets:
        num_select = int(len(shapley_values) * budget)
        selected_indices = sorted_indices[:num_select]
        
        # For now, return the selection indices
        # In practice, you would train a model on selected data and evaluate
        results[f'selected_indices_budget_{budget}'] = selected_indices.tolist()
        results[f'num_selected_budget_{budget}'] = num_select
    
    return results


def analyze_shapley_distribution(shapley_values: np.ndarray) -> Dict[str, float]:
    """
    Analyze the distribution of Shapley values.
    
    Args:
        shapley_values: Array of Shapley values
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(shapley_values)),
        'std': float(np.std(shapley_values)),
        'min': float(np.min(shapley_values)),
        'max': float(np.max(shapley_values)),
        'median': float(np.median(shapley_values)),
        'q25': float(np.percentile(shapley_values, 25)),
        'q75': float(np.percentile(shapley_values, 75)),
        'num_positive': int(np.sum(shapley_values > 0)),
        'num_negative': int(np.sum(shapley_values < 0)),
        'num_zero': int(np.sum(shapley_values == 0)),
    }


def evaluate_shapley_results(
    shapley_dir: str,
    true_labels: Optional[np.ndarray] = None,
    mislabeled_indices: Optional[np.ndarray] = None,
    output_file: Optional[str] = None
):
    """
    Comprehensive evaluation of Shapley values.
    
    Args:
        shapley_dir: Directory containing Shapley value files
        true_labels: Optional true labels for mislabeled detection
        mislabeled_indices: Optional indices of mislabeled samples
        output_file: Optional path to save evaluation results
    """
    shapley_dir = Path(shapley_dir)
    
    # Find latest Shapley values file
    shapley_files = list(shapley_dir.glob("shapley_array_iter_*.npy"))
    if not shapley_files:
        print(f"No Shapley array files found in {shapley_dir}")
        return
    
    # Use the latest iteration
    latest_file = max(shapley_files, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"Loading Shapley values from {latest_file}")
    
    shapley_values = load_shapley_array(str(latest_file))
    
    # Analyze distribution
    print("\n=== Shapley Value Distribution ===")
    distribution_stats = analyze_shapley_distribution(shapley_values)
    for key, value in distribution_stats.items():
        print(f"{key}: {value}")
    
    results = {
        'distribution': distribution_stats,
    }
    
    # Mislabeled data detection if labels provided
    if mislabeled_indices is not None:
        print("\n=== Mislabeled Data Detection ===")
        mislabeled_results = mislabeled_data_detection(
            shapley_values, None, mislabeled_indices
        )
        for key, value in mislabeled_results.items():
            print(f"{key}: {value:.4f}")
        results['mislabeled_detection'] = mislabeled_results
    
    # Data selection evaluation
    print("\n=== Data Selection ===")
    selection_results = data_selection_evaluation(shapley_values, None)
    for key, value in selection_results.items():
        if 'indices' not in key:
            print(f"{key}: {value}")
    results['data_selection'] = selection_results
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate In-Run Data Shapley values')
    parser.add_argument('--shapley_dir', type=str, required=True,
                       help='Directory containing Shapley value files')
    parser.add_argument('--mislabeled_indices', type=str, default=None,
                       help='Path to file with mislabeled indices (numpy array)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load mislabeled indices if provided
    mislabeled_indices = None
    if args.mislabeled_indices:
        mislabeled_indices = np.load(args.mislabeled_indices)
    
    # Run evaluation
    evaluate_shapley_results(
        shapley_dir=args.shapley_dir,
        mislabeled_indices=mislabeled_indices,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
