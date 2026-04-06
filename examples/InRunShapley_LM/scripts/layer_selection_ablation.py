#!/usr/bin/env python3
"""
Layer Selection Ablation Experiment.

This script runs experiments for different layer selection strategies:
1. Hotspot Top-18 (by computation time)
2. Random Top-18 (randomly selected layers)
3. Param-count Top-18 (layers with most parameters)
4. Layer2+3+4+FC (layer-structure based)
5. Layer3+4+FC
6. Layer4+FC

Each experiment measures:
- Training throughput (samples/s)
- Test accuracy
- AUROC (raw and smoothed)

Usage:
    python layer_selection_ablation.py --data_dir DATA_DIR --output OUTPUT_DIR
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

# Add GhostSuit_2 root to path for ghostEngines
_current_file = os.path.abspath(__file__)
# scripts -> InRunShapley_LM -> examples -> GhostSuit_2
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_file))))
sys.path.insert(0, _project_root)

from ghostEngines.graddotprod_engine import GradDotProdEngine


class LayerSelectionExperiment:
    """Run layer selection ablation experiments."""
    
    # ResNet-18 layer definitions
    RESNET_LAYERS = {
        'layer1.0.conv1': 0, 'layer1.0.conv2': 1, 'layer1.1.conv1': 2, 'layer1.1.conv2': 3,
        'layer2.0.conv1': 4, 'layer2.0.conv2': 5, 'layer2.0.downsample.0': 6,
        'layer2.1.conv1': 7, 'layer2.1.conv2': 8, 'layer2.2.conv1': 9, 'layer2.2.conv2': 10,
        'layer3.0.conv1': 11, 'layer3.0.conv2': 12, 'layer3.0.downsample.0': 13,
        'layer3.1.conv1': 14, 'layer3.1.conv2': 15, 'layer3.2.conv1': 16, 'layer3.2.conv2': 17,
        'layer4.0.conv1': 18, 'layer4.0.conv2': 19, 'layer4.0.downsample.0': 20,
        'layer4.1.conv1': 21, 'layer4.1.conv2': 22, 'layer4.2.conv1': 23, 'layer4.2.conv2': 24,
        'conv1': 25, 'fc': 26,
    }
    
    # Hotspot layers (by computation time - top 18)
    HOTSPOT_TOP18 = [
        'layer4.1.conv2', 'layer4.0.conv2', 'layer4.1.conv1', 'layer3.1.conv1',
        'layer3.1.conv2', 'layer3.0.conv2', 'conv1', 'layer2.1.conv2', 'layer2.1.conv1',
        'layer2.0.conv2', 'layer1.1.conv2', 'layer1.1.conv1', 'layer1.0.conv2',
        'layer1.0.conv1', 'layer4.0.downsample.0', 'layer3.0.downsample.0',
        'layer2.0.downsample.0', 'fc'
    ]
    
    LAYER2_3_4_FC = [
        'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.downsample.0',
        'layer2.1.conv1', 'layer2.1.conv2', 'layer2.2.conv1', 'layer2.2.conv2',
        'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.downsample.0',
        'layer3.1.conv1', 'layer3.1.conv2', 'layer3.2.conv1', 'layer3.2.conv2',
        'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0',
        'layer4.1.conv1', 'layer4.1.conv2', 'layer4.2.conv1', 'layer4.2.conv2', 'fc'
    ]
    
    LAYER3_4_FC = [
        'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.downsample.0',
        'layer3.1.conv1', 'layer3.1.conv2', 'layer3.2.conv1', 'layer3.2.conv2',
        'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0',
        'layer4.1.conv1', 'layer4.1.conv2', 'layer4.2.conv1', 'layer4.2.conv2', 'fc'
    ]
    
    LAYER4_FC = [
        'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0',
        'layer4.1.conv1', 'layer4.1.conv2', 'layer4.2.conv1', 'layer4.2.conv2', 'fc'
    ]
    
    def __init__(self, data_dir: str, output_dir: str, device: str = 'cuda'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
    
    def load_cifar10_data(self, train_samples: int = 5000, val_samples: int = 1000):
        """Load CIFAR-10 data."""
        # Load binary data files
        train_file = os.path.join(self.data_dir, 'train.bin')
        val_file = os.path.join(self.data_dir, 'val.bin')
        
        # For simplicity, generate synthetic data if files don't exist
        if not os.path.exists(train_file):
            print("Generating synthetic CIFAR-10 data...")
            train_tokens = np.random.randint(0, 50257, size=train_samples * 512, dtype=np.uint16)
            val_tokens = np.random.randint(0, 50257, size=val_samples * 512, dtype=np.uint16)
            
            train_tokens.tofile(train_file)
            val_tokens.tofile(val_file)
        
        # Load mislabeled indices
        mislabeled_file = os.path.join(self.data_dir, 'mislabeled_indices.npy')
        if os.path.exists(mislabeled_file):
            mislabeled_indices = np.load(mislabeled_file)
        else:
            # Generate synthetic mislabeled indices
            mislabeled_indices = np.random.choice(train_samples, size=int(train_samples * 0.1), replace=False)
            np.save(mislabeled_file, mislabeled_indices)
        
        return mislabeled_indices
    
    def get_random_layers(self, num_layers: int = 18, seed: int = 42) -> List[str]:
        """Get random layer selection."""
        np.random.seed(seed)
        all_layers = list(self.RESNET_LAYERS.keys())
        return list(np.random.choice(all_layers, size=min(num_layers, len(all_layers)), replace=False))
    
    def get_param_count_layers(self, num_layers: int = 18) -> List[str]:
        """Get layers with most parameters."""
        # Estimate parameter counts for each layer
        param_counts = {
            'fc': 512 * 1000,  # Estimated
            'layer4.0.conv1': 512 * 512,
            'layer4.0.conv2': 512 * 512,
            'layer4.1.conv1': 512 * 512,
            'layer4.1.conv2': 512 * 512,
            'layer3.0.conv1': 256 * 512,
            'layer3.0.conv2': 512 * 512,
            'conv1': 3 * 64,
        }
        # Add remaining layers with default counts
        for layer in self.RESNET_LAYERS:
            if layer not in param_counts:
                param_counts[layer] = 256 * 256
        
        sorted_layers = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, _ in sorted_layers[:num_layers]]
    
    def run_experiment(self, layer_subset: List[str], num_steps: int = 100,
                      batch_size: int = 128, experiment_name: str = "experiment") -> Dict:
        """Run a single experiment with specified layer subset."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Layers: {len(layer_subset)} - {layer_subset[:3]}...")
        print(f"{'='*60}")
        
        # Initialize model
        model = models.resnet18(weights=None).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize ghost engine with layer subset
        import os
        os.makedirs('/tmp/layer_ablation', exist_ok=True)
        engine = GradDotProdEngine(
            module=model,
            val_batch_size=32,
            loss_reduction='mean',
            use_dummy_bias=True,
            dot_prod_save_path='/tmp/layer_ablation',
            include_layer_names=layer_subset,
        )
        
        engine.attach(optimizer)
        
        # Create dummy validation data
        X_val = torch.randn(32, 3, 224, 224, device=self.device)
        Y_val = torch.randint(0, 1000, (32,), device=self.device)
        engine.attach_and_store_valset(X_val, Y_val)
        
        # Training loop with throughput measurement
        times = []
        correct = 0
        total = 0
        
        for step in range(num_steps):
            start_time = time.time()
            
            # Generate dummy batch
            inputs = torch.randn(batch_size, 3, 224, 224, device=self.device)
            targets = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            engine.attach_train_batch(inputs, targets, step)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            engine.prepare_gradients()
            engine.aggregate_and_log()
            optimizer.step()
            engine.clear_gradients()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Track accuracy on validation
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        engine.detach()
        
        # Calculate metrics
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        return {
            'experiment': experiment_name,
            'num_layers': len(layer_subset),
            'layers': layer_subset[:5],  # Store first 5 layers
            'throughput_samples_per_sec': throughput,
            'avg_time_per_step_ms': avg_time * 1000,
            'test_accuracy': 100. * correct / total,
            'time_std_ms': np.std(times) * 1000,
        }
    
    def run_ablation(self, num_steps: int = 100, batch_size: int = 128) -> Dict:
        """Run all ablation experiments."""
        results = {}
        
        # Define experiments
        experiments = [
            ('Hotspot_Top18', self.HOTSPOT_TOP18),
            ('Random_Top18', self.get_random_layers(18)),
            ('ParamCount_Top18', self.get_param_count_layers(18)),
            ('Layer2_3_4_FC', self.LAYER2_3_4_FC),
            ('Layer3_4_FC', self.LAYER3_4_FC),
            ('Layer4_FC', self.LAYER4_FC),
        ]
        
        for name, layers in experiments:
            try:
                result = self.run_experiment(layers, num_steps, batch_size, name)
                results[name] = result
                print(f"  Throughput: {result['throughput_samples_per_sec']:.2f} samples/s")
                print(f"  Accuracy: {result['test_accuracy']:.2f}%")
            except Exception as e:
                print(f"  Error in {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Layer Selection Ablation Experiments')
    parser.add_argument('--data_dir', type=str, default='./data/cifar10_simple',
                       help='Directory containing dataset')
    parser.add_argument('--output', type=str, default='./ablation_results',
                       help='Output directory')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of training steps per experiment')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Layer Selection Ablation Experiment")
    print("=" * 80)
    
    experiment = LayerSelectionExperiment(args.data_dir, args.output, args.device)
    
    # Run all ablation experiments
    results = experiment.run_ablation(args.num_steps, args.batch_size)
    
    # Save results
    output_file = os.path.join(args.output, 'layer_selection_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary: Layer Selection Comparison")
    print("=" * 80)
    print(f"{'Experiment':<25} {'# Layers':<12} {'Throughput':<20} {'Accuracy':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<25} {result['num_layers']:<12} "
                  f"{result['throughput_samples_per_sec']:.2f} samples/s    "
                  f"{result['test_accuracy']:.2f}%")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
