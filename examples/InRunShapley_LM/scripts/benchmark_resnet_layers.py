#!/usr/bin/env python3
"""
ResNet-18 Layer-wise Computation Time Benchmark.

This script profiles each layer of ResNet-18 to identify hotspot layers
for the ClusterSuit framework.

Usage:
    python benchmark_resnet_layers.py [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ResNet18Profiler:
    """Profiler for ResNet-18 layer-wise computation time."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.layer_times = {}
        self.hooks = []
    
    def load_model(self):
        """Load ResNet-18 model."""
        print("Loading ResNet-18 model...")
        self.model = models.resnet18(weights=None).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def profile_with_hooks(self, num_iterations: int = 50, warmup: int = 10) -> Dict:
        """Profile layers using forward hooks."""
        print("\nProfiling ResNet-18 layers with hooks...")
        
        # Storage for timing data
        self.layer_times = {}
        module_list = []
        
        # Store timing data in a mutable container
        timing_storage = {'times': {}, 'current_module': None}
        
        def pre_hook(name):
            def hook_fn(module, input):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                module._hook_start_time = time.perf_counter()
            return hook_fn
        
        def forward_hook(name):
            def hook_fn(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if hasattr(module, '_hook_start_time'):
                    elapsed = (time.perf_counter() - module._hook_start_time) * 1000
                    if name not in timing_storage['times']:
                        timing_storage['times'][name] = []
                    timing_storage['times'][name].append(elapsed)
            return hook_fn
        
        # Register hooks on conv layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook1 = module.register_forward_pre_hook(pre_hook(name))
                hook2 = module.register_forward_hook(forward_hook(name))
                self.hooks.append(hook1)
                self.hooks.append(hook2)
                module_list.append(name)
        
        # Create dummy input
        dummy_input = torch.randn(32, 3, 224, 224, device=self.device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile iterations
        print(f"Running {num_iterations} profiling iterations...")
        timing_storage['times'] = {}
        
        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(dummy_input)
                
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{num_iterations} completed")
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        self.layer_times = timing_storage['times']
        return self._compute_statistics()
    
    def _compute_statistics(self) -> Dict:
        """Compute statistics from collected timings."""
        results = {}
        
        for name, times in self.layer_times.items():
            if len(times) > 0:
                avg = sum(times) / len(times)
                results[name] = {
                    'layer_name': name,
                    'avg_time_ms': avg,
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'std_time_ms': (sum((t - avg)**2 for t in times) / len(times)) ** 0.5,
                    'count': len(times),
                }
        
        return results
    
    def profile_full_forward(self, num_iterations: int = 50) -> Dict:
        """Profile full forward pass for comparison."""
        print("\nProfiling full forward pass...")
        
        dummy_input = torch.randn(32, 3, 224, 224, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append((time.perf_counter() - start) * 1000)
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
        }
    
    def identify_hotspot_layers(self, results: Dict, top_k: int = 18) -> List:
        """Identify top-K hotspot layers by computation time."""
        sorted_layers = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True)
        return sorted_layers[:top_k]
    
    def print_results(self, results: Dict, forward_time: Dict, hotspot_layers: List):
        """Print profiling results."""
        print("\n" + "=" * 80)
        print("ResNet-18 Layer Computation Time Analysis")
        print("=" * 80)
        
        print(f"\nFull Forward Pass: {forward_time['avg_time_ms']:.4f} ms")
        
        # Sort by computation time
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Layer Name':<30} {'Avg (ms)':<12} {'Range (ms)':<20}")
        print("-" * 80)
        
        total_time = sum(r['avg_time_ms'] for r in results.values())
        
        hotspot_names = [name for name, _ in hotspot_layers]
        hotspot_time = sum(data['avg_time_ms'] for name, data in hotspot_layers)
        
        for rank, (layer_name, data) in enumerate(sorted_results, 1):
            time_range = f"{data['min_time_ms']:.3f} - {data['max_time_ms']:.3f}"
            hotspot_mark = " [HOTSPOT]" if layer_name in hotspot_names else ""
            print(f"{rank:<6} {layer_name:<30} {data['avg_time_ms']:<12.4f} {time_range:<20}{hotspot_mark}")
        
        print("-" * 80)
        print(f"Total layer time: {total_time:.4f} ms")
        print(f"\nTop-{len(hotspot_layers)} Hotspot Layers:")
        print(f"  Time: {hotspot_time:.4f} ms ({100*hotspot_time/total_time:.1f}% of total)")
        print(f"  Layers: {[name for name, _ in hotspot_layers]}")


def main():
    parser = argparse.ArgumentParser(description='ResNet-18 Layer-wise Computation Time Benchmark')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations for timing')
    parser.add_argument('--top_k', type=int, default=18,
                       help='Number of hotspot layers to identify')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    profiler = ResNet18Profiler(device=args.device)
    profiler.load_model()
    
    results = profiler.profile_with_hooks(num_iterations=args.iterations)
    forward_time = profiler.profile_full_forward(num_iterations=args.iterations)
    hotspot_layers = profiler.identify_hotspot_layers(results, top_k=args.top_k)
    
    profiler.print_results(results, forward_time, hotspot_layers)
    
    output_data = {
        'model': 'ResNet-18',
        'total_layers': len(results),
        'iterations': args.iterations,
        'layer_times': results,
        'forward_pass_time': forward_time,
        'hotspot_layers': [{'name': name, **data} for name, data in hotspot_layers],
        'hotspot_layer_names': [name for name, _ in hotspot_layers],
    }
    
    output_file = os.path.join(args.output, 'resnet18_layer_times.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_data


if __name__ == "__main__":
    main()
