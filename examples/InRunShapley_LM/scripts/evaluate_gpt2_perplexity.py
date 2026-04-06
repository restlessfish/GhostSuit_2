#!/usr/bin/env python3
"""
GPT-2 Perplexity Evaluation Script.

This script evaluates the impact of Shapley value-based data filtering
on GPT-2 language model training. It measures perplexity on validation set
for models trained with:
1. Full training data
2. Top-K% Shapley filtered data (high-value samples only)
3. Bottom-K% Shapley filtered data (low-value samples only)
4. Random K% data selection (baseline)

Usage:
    python evaluate_gpt2_perplexity.py --data_dir DATA_DIR --shapley_dir SHAPLEY_DIR --output OUTPUT_DIR
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TokenDataset(Dataset):
    """Simple token dataset for GPT-2."""
    
    def __init__(self, tokens: np.ndarray, block_size: int = 128):
        self.tokens = torch.from_numpy(tokens).long()
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y


def load_data(data_dir: str, block_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data.
    
    Returns:
        Tuple of (train_tokens, val_tokens, test_tokens)
    """
    train_file = os.path.join(data_dir, 'train.bin')
    val_file = os.path.join(data_dir, 'val.bin')
    test_file = os.path.join(data_dir, 'test.bin')
    
    # Load or generate data
    if os.path.exists(train_file):
        train_tokens = np.fromfile(train_file, dtype=np.uint16)
    else:
        print("Generating synthetic training data...")
        train_tokens = np.random.randint(0, 50257, size=2048000, dtype=np.uint16)
        train_tokens.tofile(train_file)
    
    if os.path.exists(val_file):
        val_tokens = np.fromfile(val_file, dtype=np.uint16)
    else:
        print("Generating synthetic validation data...")
        val_tokens = np.random.randint(0, 50257, size=102400, dtype=np.uint16)
        val_tokens.tofile(val_file)
    
    if os.path.exists(test_file):
        test_tokens = np.fromfile(test_file, dtype=np.uint16)
    else:
        print("Generating synthetic test data...")
        test_tokens = np.random.randint(0, 50257, size=102400, dtype=np.uint16)
        test_tokens.tofile(test_file)
    
    return train_tokens, val_tokens, test_tokens


def load_shapley_values(shapley_dir: str) -> np.ndarray:
    """Load Shapley values from directory."""
    import glob
    
    shapley_files = glob.glob(os.path.join(shapley_dir, 'shapley_array_iter_*.npy'))
    if not shapley_files:
        raise FileNotFoundError(f"No shapley files found in {shapley_dir}")
    
    # Get latest iteration
    latest = max(shapley_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f"Loading Shapley values from: {latest}")
    
    shapley_values = np.load(latest)
    
    # Load indices if available
    iter_num = latest.split('_')[-1].split('.')[0]
    indices_file = os.path.join(shapley_dir, f'shapley_indices_iter_{iter_num}.npy')
    
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        # Reconstruct full array
        num_samples = len(shapley_values) if indices is None else max(indices) + 1
        if len(shapley_values) < num_samples:
            full_values = np.zeros(num_samples)
            full_values[indices] = shapley_values
            shapley_values = full_values
    
    return shapley_values


def select_data_by_shapley(shapley_values: np.ndarray, 
                          selection_ratio: float,
                          selection_type: str = 'top') -> np.ndarray:
    """
    Select training samples based on Shapley values.
    
    Args:
        shapley_values: Shapley values for each sample
        selection_ratio: Fraction of data to select (0-1)
        selection_type: 'top', 'bottom', or 'random'
        
    Returns:
        Selected sample indices
    """
    n_samples = len(shapley_values)
    n_select = int(n_samples * selection_ratio)
    
    if selection_type == 'top':
        # Select samples with highest Shapley values
        sorted_indices = np.argsort(shapley_values)[::-1]
        selected = sorted_indices[:n_select]
    elif selection_type == 'bottom':
        # Select samples with lowest Shapley values
        sorted_indices = np.argsort(shapley_values)
        selected = sorted_indices[:n_select]
    elif selection_type == 'random':
        # Random selection
        selected = np.random.permutation(n_samples)[:n_select]
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")
    
    return selected


def compute_perplexity(model: nn.Module, data_loader: DataLoader, 
                      device: str = 'cuda') -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Language model
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            outputs = model(x)
            logits = outputs.logits
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += y.numel()
            
            # Limit evaluation batches for speed
            if batch_idx >= 50:
                break
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


class GPT2PerplexityEvaluator:
    """Evaluate GPT-2 perplexity with different data selection strategies."""
    
    def __init__(self, data_dir: str, output_dir: str, device: str = 'cuda'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.train_tokens, self.val_tokens, self.test_tokens = load_data(data_dir)
        print(f"Data loaded: train={len(self.train_tokens)}, val={len(self.val_tokens)}, test={len(self.test_tokens)}")
    
    def evaluate_full_model(self, model_path: Optional[str] = None) -> Dict:
        """
        Evaluate perplexity of a trained model.
        
        Args:
            model_path: Path to saved model (if None, use untrained model)
            
        Returns:
            Dictionary with perplexity results
        """
        from transformers import GPT2LMHeadModel, GPT2Config
        
        # Load model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        else:
            print("Using untrained GPT-2 model for baseline")
            config = GPT2Config()
            model = GPT2LMHeadModel(config).to(self.device)
        
        # Create validation dataloader
        val_dataset = TokenDataset(self.val_tokens, block_size=128)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Compute perplexity
        perplexity = compute_perplexity(model, val_loader, self.device)
        
        print(f"Validation Perplexity: {perplexity:.4f}")
        
        return {
            'perplexity': perplexity,
            'model_path': model_path,
        }
    
    def evaluate_shapley_selection(self, shapley_dir: str, 
                                  selection_ratio: float = 0.8) -> Dict:
        """
        Evaluate perplexity with Shapley-based data selection.
        
        Args:
            shapley_dir: Directory containing Shapley values
            selection_ratio: Fraction of data to keep
            
        Returns:
            Dictionary with results for different selection strategies
        """
        from transformers import GPT2LMHeadModel, GPT2Config
        
        # Load Shapley values
        shapley_values = load_shapley_values(shapley_dir)
        
        results = {}
        
        # Evaluate each selection strategy
        strategies = ['top', 'bottom', 'random']
        
        for strategy in strategies:
            print(f"\nEvaluating {strategy} {int(selection_ratio*100)}% selection...")
            
            # Select indices
            selected_indices = select_data_by_shapley(shapley_values, selection_ratio, strategy)
            print(f"  Selected {len(selected_indices)} samples")
            
            # Create filtered dataset
            filtered_tokens = self.train_tokens[selected_indices * 512]  # Approximate
            filtered_dataset = TokenDataset(filtered_tokens, block_size=128)
            filtered_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=False)
            
            # Load or create model
            config = GPT2Config()
            model = GPT2LMHeadModel(config).to(self.device)
            
            # Quick training (for demonstration)
            print(f"  Quick training on selected data...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            model.train()
            for i, (x, y) in enumerate(filtered_loader):
                if i >= 100:  # Quick training
                    break
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    y.view(-1)
                )
                loss.backward()
                optimizer.step()
            
            # Evaluate
            perplexity = compute_perplexity(model, filtered_loader, self.device)
            
            results[strategy] = {
                'strategy': strategy,
                'selection_ratio': selection_ratio,
                'num_samples': len(selected_indices),
                'perplexity': perplexity,
            }
            
            print(f"  Perplexity: {perplexity:.4f}")
        
        return results
    
    def run_full_evaluation(self, shapley_dir: Optional[str] = None,
                          selection_ratios: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]) -> Dict:
        """
        Run full evaluation comparing different selection ratios.
        
        Args:
            shapley_dir: Directory with Shapley values
            selection_ratios: List of selection ratios to evaluate
            
        Returns:
            Complete evaluation results
        """
        results = {
            'selection_ratios': selection_ratios,
            'evaluations': {},
        }
        
        if shapley_dir:
            print("\n" + "=" * 60)
            print("Shapley-Based Data Selection Evaluation")
            print("=" * 60)
            
            shapley_values = load_shapley_values(shapley_dir)
            
            for ratio in selection_ratios:
                print(f"\n--- Selection Ratio: {int(ratio*100)}% ---")
                
                ratio_results = {}
                for strategy in ['top', 'bottom', 'random']:
                    selected = select_data_by_shapley(shapley_values, ratio, strategy)
                    ratio_results[strategy] = len(selected)
                
                results['evaluations'][f'{int(ratio*100)}pct'] = ratio_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description='GPT-2 Perplexity Evaluation')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--shapley_dir', type=str, default=None,
                       help='Directory containing Shapley values')
    parser.add_argument('--output', type=str, default='./perplexity_results',
                       help='Output directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained GPT-2 model')
    parser.add_argument('--selection_ratio', type=float, default=0.8,
                       help='Data selection ratio (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPT-2 Perplexity Evaluation")
    print("=" * 80)
    
    evaluator = GPT2PerplexityEvaluator(args.data_dir, args.output, args.device)
    
    results = {}
    
    # Evaluate baseline (full model)
    print("\n1. Baseline perplexity (untrained model)...")
    baseline = evaluator.evaluate_full_model(args.model_path)
    results['baseline'] = baseline
    
    # Evaluate Shapley-based selection if shapley_dir provided
    if args.shapley_dir:
        print("\n2. Shapley-based data selection evaluation...")
        selection_results = evaluator.evaluate_shapley_selection(
            args.shapley_dir, args.selection_ratio
        )
        results['shapley_selection'] = selection_results
    
    # Save results
    output_file = os.path.join(args.output, 'perplexity_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Perplexity Evaluation Summary")
    print("=" * 80)
    
    if 'baseline' in results:
        print(f"Baseline perplexity: {results['baseline']['perplexity']:.4f}")
    
    if 'shapley_selection' in results:
        print("\nShapley-based selection results:")
        for strategy, data in results['shapley_selection'].items():
            print(f"  {strategy}: {data.get('perplexity', 'N/A'):.4f}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
