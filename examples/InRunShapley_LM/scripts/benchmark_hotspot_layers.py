#!/usr/bin/env python3
"""
GPT-2 Layer-wise Computation Time Benchmark.

This script profiles each layer of GPT-2 to identify hotspot layers
for the ClusterSuit framework.

Usage:
    python benchmark_hotspot_layers.py [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class GPT2HotspotProfiler:
    """Profiler for GPT-2 layer-wise computation time."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.layer_times = {}
        self.hooks = []
    
    def load_model(self):
        """Load GPT-2 model."""
        from transformers import GPT2LMHeadModel
        
        print("Loading GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def profile_all_layers(self, num_iterations: int = 50, warmup: int = 10) -> Dict:
        """Profile all GPT-2 transformer layers."""
        print(f"\nProfiling {len(self.model.transformer.h)} transformer layers...")
        
        results = {}
        num_layers = len(self.model.transformer.h)
        
        # Create dummy input
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
        
        # First, get the hidden states by running through embedding
        hidden_states = self.model.transformer.wte(input_ids)
        position_ids = torch.arange(0, seq_length, device=self.device).unsqueeze(0)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        
        # Warmup full model
        print("Warming up...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile each layer independently
        for i in range(num_layers):
            print(f"  Profiling layer {i}...", end=" ", flush=True)
            
            times = []
            for _ in range(num_iterations):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    # Run through this layer only
                    _ = self.model.transformer.h[i](hidden_states, None)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)  # Convert to ms
            
            avg_time = sum(times) / len(times)
            results[i] = {
                'layer_name': f'transformer.h.{i}',
                'avg_time_ms': avg_time,
                'min_time_ms': min(times),
                'max_time_ms': max(times),
            }
            print(f"{avg_time:.4f} ms")
        
        return results
    
    def profile_full_forward(self, num_iterations: int = 50) -> Dict:
        """Profile full forward pass."""
        print("\nProfiling full forward pass...")
        
        input_ids = torch.randint(0, 50257, (4, 128), device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(input_ids)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append((time.perf_counter() - start) * 1000)
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
        }
    
    def identify_hotspot_layers(self, results: Dict, top_k: int = 6) -> List[int]:
        """Identify top-K hotspot layers by computation time."""
        sorted_layers = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True)
        return [layer_idx for layer_idx, _ in sorted_layers[:top_k]]
    
    def print_results(self, results: Dict, forward_time: Dict, hotspot_layers: List[int]):
        """Print profiling results."""
        print("\n" + "=" * 80)
        print("GPT-2 Layer Computation Time Analysis")
        print("=" * 80)
        
        print(f"\nFull Forward Pass: {forward_time['avg_time_ms']:.4f} ms")
        
        # Sort by computation time
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Layer Name':<20} {'Avg (ms)':<15} {'Hotspot':<10}")
        print("-" * 60)
        
        total_time = sum(r['avg_time_ms'] for r in results.values())
        
        for rank, (layer_idx, data) in enumerate(sorted_results, 1):
            is_hotspot = "Yes" if layer_idx in hotspot_layers else ""
            print(f"{rank:<6} {data['layer_name']:<20} {data['avg_time_ms']:<15.4f} {is_hotspot:<10}")
        
        print("-" * 60)
        print(f"Total time: {total_time:.4f} ms")
        
        # Hotspot analysis
        hotspot_time = sum(results[idx]['avg_time_ms'] for idx in hotspot_layers)
        print(f"\nHotspot Layers (top-{len(hotspot_layers)}): {hotspot_layers}")
        print(f"Hotspot time: {hotspot_time:.4f} ms ({100*hotspot_time/total_time:.1f}% of total)")


def main():
    parser = argparse.ArgumentParser(description='GPT-2 Layer-wise Computation Time Benchmark')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations for timing')
    parser.add_argument('--top_k', type=int, default=6,
                       help='Number of hotspot layers to identify')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Set to single GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Warning: Multiple GPUs detected. Using only GPU 0.")
        torch.cuda.set_device(0)
    
    profiler = GPT2HotspotProfiler(device=args.device)
    profiler.load_model()
    
    results = profiler.profile_all_layers(num_iterations=args.iterations)
    forward_time = profiler.profile_full_forward(num_iterations=args.iterations)
    hotspot_layers = profiler.identify_hotspot_layers(results, top_k=args.top_k)
    
    profiler.print_results(results, forward_time, hotspot_layers)
    
    # Save results
    output_data = {
        'model': 'GPT-2',
        'total_layers': len(results),
        'iterations': args.iterations,
        'layer_times': results,
        'forward_pass_time': forward_time,
        'hotspot_layers': hotspot_layers,
        'hotspot_layer_names': [f'transformer.h.{i}' for i in hotspot_layers],
    }
    
    output_file = os.path.join(args.output, 'gpt2_layer_times.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output_data


if __name__ == "__main__":
    main()
