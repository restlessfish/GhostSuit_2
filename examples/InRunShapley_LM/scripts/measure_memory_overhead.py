#!/usr/bin/env python3
"""
Memory Overhead Measurement for ClusterSuit.

This script measures GPU memory overhead of using the ghost dot-product engine
compared to regular training. It compares memory usage for:
1. Regular training (no ghost engine)
2. Full-layer ghost computation
3. Hotspot-layer only ghost computation

Usage:
    python measure_memory_overhead.py [--model MODEL_TYPE] [--batch_size BS]
"""

import os
import sys
import argparse
import json
from typing import Dict, List

import torch
import torch.nn as nn

# Add GhostSuit_2 root to path for ghostEngines
_current_file = os.path.abspath(__file__)
# scripts -> InRunShapley_LM -> examples -> GhostSuit_2
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_file))))
sys.path.insert(0, _project_root)


def get_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        }
    return {
        'allocated_mb': 0,
        'reserved_mb': 0,
        'max_allocated_mb': 0,
    }


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_snapshots = []
    
    def snapshot(self, label: str):
        """Take a memory snapshot."""
        info = get_memory_info()
        info['label'] = label
        self.memory_snapshots.append(info)
        return info
    
    def compare(self, before_label: str, after_label: str) -> Dict:
        """Compare memory between two snapshots."""
        before = next((s for s in self.memory_snapshots if s['label'] == before_label), None)
        after = next((s for s in self.memory_snapshots if s['label'] == after_label), None)
        
        if before is None or after is None:
            return {}
        
        return {
            'allocated_diff_mb': after['allocated_mb'] - before['allocated_mb'],
            'reserved_diff_mb': after['reserved_mb'] - before['reserved_mb'],
            'peak_increase_mb': after['max_allocated_mb'] - before['max_allocated_mb'],
        }


class ResNetMemoryTest:
    """Test memory usage with ResNet-18."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def test_regular_training(self, batch_size: int = 128, iterations: int = 10) -> Dict:
        """Test memory usage for regular training."""
        import torchvision.models as models
        
        reset_memory_stats()
        profiler = MemoryProfiler(self.device)
        
        model = models.resnet18(weights=None).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        profiler.snapshot('init')
        
        for i in range(iterations):
            inputs = torch.randn(batch_size, 3, 224, 224, device=self.device)
            targets = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                profiler.snapshot(f'after_iter_{i}')
        
        final_info = profiler.snapshot('final')
        max_info = get_memory_info()
        
        return {
            'init_allocated': profiler.memory_snapshots[0]['allocated_mb'],
            'peak_allocated': max_info['max_allocated_mb'],
            'final_allocated': final_info['allocated_mb'],
        }
    
    def test_with_ghost_engine(self, batch_size: int = 128, iterations: int = 10, 
                               hotspot_layers: List[str] = None) -> Dict:
        """Test memory usage with ghost engine."""
        import torchvision.models as models
        
        reset_memory_stats()
        profiler = MemoryProfiler(self.device)
        
        model = models.resnet18(weights=None).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize ghost engine
        from ghostEngines.graddotprod_engine import GradDotProdEngine
        
        engine = GradDotProdEngine(
            module=model,
            val_batch_size=32,
            loss_reduction='mean',
            use_dummy_bias=True,
            dot_prod_save_path='/tmp/ghost_memory_test',
            include_layer_names=hotspot_layers,
        )
        
        import os
        os.makedirs('/tmp/ghost_memory_test', exist_ok=True)
        
        profiler.snapshot('init')
        engine.attach(optimizer)
        profiler.snapshot('after_attach')
        
        # Create validation data
        X_val = torch.randn(32, 3, 224, 224, device=self.device)
        Y_val = torch.randint(0, 1000, (32,), device=self.device)
        engine.attach_and_store_valset(X_val, Y_val)
        
        for i in range(iterations):
            inputs = torch.randn(batch_size, 3, 224, 224, device=self.device)
            targets = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            engine.attach_train_batch(inputs, targets, i)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            engine.prepare_gradients()
            engine.aggregate_and_log()
            optimizer.step()
            engine.clear_gradients()
            
            if i == 0:
                profiler.snapshot(f'after_iter_{i}')
        
        final_info = profiler.snapshot('final')
        max_info = get_memory_info()
        
        engine.detach()
        
        return {
            'init_allocated': profiler.memory_snapshots[0]['allocated_mb'],
            'after_attach_allocated': profiler.memory_snapshots[1]['allocated_mb'],
            'peak_allocated': max_info['max_allocated_mb'],
            'final_allocated': final_info['allocated_mb'],
            'hotspot_layers': hotspot_layers,
        }


class GPT2MemoryTest:
    """Test memory usage with GPT-2."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def test_regular_training(self, batch_size: int = 8, seq_length: int = 128, 
                            iterations: int = 5) -> Dict:
        """Test memory usage for regular GPT-2 training."""
        from transformers import GPT2LMHeadModel, GPT2Config
        
        reset_memory_stats()
        profiler = MemoryProfiler(self.device)
        
        config = GPT2Config(n_layer=4, n_positions=seq_length)  # Smaller model for testing
        model = GPT2LMHeadModel(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        profiler.snapshot('init')
        
        for i in range(iterations):
            input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
            labels = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.logits.view(-1, 50257), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            if i == 0:
                profiler.snapshot(f'after_iter_{i}')
        
        final_info = profiler.snapshot('final')
        max_info = get_memory_info()
        
        return {
            'init_allocated': profiler.memory_snapshots[0]['allocated_mb'],
            'peak_allocated': max_info['max_allocated_mb'],
            'final_allocated': final_info['allocated_mb'],
        }
    
    def test_with_ghost_engine(self, batch_size: int = 8, seq_length: int = 128,
                               iterations: int = 5, hotspot_layers: List[str] = None) -> Dict:
        """Test memory usage with GPT-2 + ghost engine."""
        from transformers import GPT2LMHeadModel, GPT2Config
        
        reset_memory_stats()
        profiler = MemoryProfiler(self.device)
        
        config = GPT2Config(n_layer=4, n_positions=seq_length)
        model = GPT2LMHeadModel(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        from ghostEngines.graddotprod_engine import GradDotProdEngine
        
        engine = GradDotProdEngine(
            module=model,
            val_batch_size=4,
            loss_reduction='mean',
            use_dummy_bias=True,
            dot_prod_save_path='/tmp/ghost_memory_test',
            include_layer_names=hotspot_layers,
        )
        
        profiler.snapshot('init')
        engine.attach(optimizer)
        profiler.snapshot('after_attach')
        
        X_val = torch.randint(0, 50257, (4, seq_length), device=self.device)
        Y_val = torch.randint(0, 50257, (4,), device=self.device)
        engine.attach_and_store_valset(X_val, Y_val)
        
        for i in range(iterations):
            input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
            labels = torch.randint(0, 50257, (batch_size,), device=self.device)
            
            engine.attach_train_batch(input_ids, labels, i)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.logits[:, :-1, :].contiguous().view(-1, 50257), 
                           labels[:, 1:].contiguous().view(-1))
            loss.backward()
            
            engine.prepare_gradients()
            engine.aggregate_and_log()
            optimizer.step()
            engine.clear_gradients()
            
            if i == 0:
                profiler.snapshot(f'after_iter_{i}')
        
        final_info = profiler.snapshot('final')
        max_info = get_memory_info()
        
        engine.detach()
        
        return {
            'init_allocated': profiler.memory_snapshots[0]['allocated_mb'],
            'after_attach_allocated': profiler.memory_snapshots[1]['allocated_mb'],
            'peak_allocated': max_info['max_allocated_mb'],
            'final_allocated': final_info['allocated_mb'],
            'hotspot_layers': hotspot_layers,
        }


def main():
    parser = argparse.ArgumentParser(description='Measure Memory Overhead of Ghost Engine')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'gpt2'],
                       help='Model to test')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations')
    parser.add_argument('--output', type=str, default='./memory_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Memory Overhead Measurement")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    
    results = {}
    
    if args.model == 'resnet18':
        tester = ResNetMemoryTest(device)
        
        print("\n1. Testing regular training...")
        results['regular'] = tester.test_regular_training(args.batch_size, args.iterations)
        
        print("2. Testing with ghost engine (all layers)...")
        results['ghost_all_layers'] = tester.test_with_ghost_engine(args.batch_size, args.iterations)
        
        print("3. Testing with ghost engine (hotspot layers)...")
        hotspot = ['layer4.0', 'layer4.1', 'fc']  # Example hotspot layers
        results['ghost_hotspot'] = tester.test_with_ghost_engine(
            args.batch_size, args.iterations, hotspot_layers=hotspot
        )
    
    elif args.model == 'gpt2':
        tester = GPT2MemoryTest(device)
        
        print("\n1. Testing regular GPT-2 training...")
        results['regular'] = tester.test_regular_training(args.batch_size, 128, args.iterations)
        
        print("2. Testing GPT-2 with ghost engine (all layers)...")
        results['ghost_all_layers'] = tester.test_with_ghost_engine(
            args.batch_size, 128, args.iterations
        )
        
        print("3. Testing GPT-2 with ghost engine (hotspot layers)...")
        hotspot = ['transformer.h.2', 'transformer.h.3']  # Example hotspot layers
        results['ghost_hotspot'] = tester.test_with_ghost_engine(
            args.batch_size, 128, args.iterations, hotspot_layers=hotspot
        )
    
    # Calculate overhead
    regular_mem = results['regular']['peak_allocated']
    ghost_all_mem = results['ghost_all_layers']['peak_allocated']
    ghost_hotspot_mem = results['ghost_hotspot']['peak_allocated']
    
    results['overhead'] = {
        'ghost_all_layers_mb': ghost_all_mem - regular_mem,
        'ghost_all_layers_percent': 100 * (ghost_all_mem - regular_mem) / regular_mem,
        'ghost_hotspot_mb': ghost_hotspot_mem - regular_mem,
        'ghost_hotspot_percent': 100 * (ghost_hotspot_mem - regular_mem) / regular_mem,
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("Memory Usage Summary (MB)")
    print("=" * 80)
    print(f"Regular training peak:        {regular_mem:.2f}")
    print(f"Ghost (all layers) peak:       {ghost_all_mem:.2f} (+{results['overhead']['ghost_all_layers_mb']:.2f}, {results['overhead']['ghost_all_layers_percent']:.1f}%)")
    print(f"Ghost (hotspot only) peak:     {ghost_hotspot_mem:.2f} (+{results['overhead']['ghost_hotspot_mb']:.2f}, {results['overhead']['ghost_hotspot_percent']:.1f}%)")
    
    # Save results
    output_file = os.path.join(args.output, f'memory_overhead_{args.model}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
