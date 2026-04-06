#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for ClusterSuit.

This script orchestrates all experiments needed for the ClusterSuit paper:
1. GPT-2 Layer-wise Computation Time Analysis
2. ResNet-18 Layer-wise Computation Time Analysis
3. Layer Selection Ablation (Hotspot, Random, Param-count, Layer-structure)
4. K-Value AUROC Evaluation (K=2, 5, 10)
5. Memory Overhead Measurement
6. GPT-2 Perplexity Evaluation

Usage:
    python run_all_experiments.py --experiments EXP_1,EXP_2,... --output OUTPUT_DIR

Example:
    python run_all_experiments.py --experiments all --output ./paper_experiments
"""

import os
import sys
import argparse
import json
import subprocess
import time
from typing import List, Dict, Optional
from datetime import datetime


class ExperimentRunner:
    """Orchestrate and run all ClusterSuit experiments."""
    
    EXPERIMENTS = {
        'gpt2_hotspot': {
            'script': 'benchmark_hotspot_layers.py',
            'args': ['--output', '{output}/gpt2_hotspot', '--iterations', '100', '--top_k', '6'],
            'description': 'GPT-2 Layer-wise Computation Time Analysis',
        },
        'resnet_hotspot': {
            'script': 'benchmark_resnet_layers.py', 
            'args': ['--output', '{output}/resnet_hotspot', '--iterations', '100'],
            'description': 'ResNet-18 Layer-wise Computation Time Analysis',
        },
        'memory_overhead': {
            'script': 'measure_memory_overhead.py',
            'args': ['--model', 'resnet18', '--batch_size', '128', '--iterations', '20', '--output', '{output}/memory'],
            'description': 'Memory Overhead Measurement',
        },
        'layer_ablation': {
            'script': 'layer_selection_ablation.py',
            'args': ['--data_dir', './data/cifar10_simple', '--output', '{output}/layer_ablation', '--num_steps', '100'],
            'description': 'Layer Selection Ablation Experiment',
        },
        'k_value_auroc': {
            'script': 'evaluate_auroc_by_k.py',
            'args': ['--output', '{output}/k_auroc'],
            'description': 'K-Value AUROC Evaluation',
        },
        'gpt2_perplexity': {
            'script': 'evaluate_gpt2_perplexity.py',
            'args': ['--data_dir', './data/cifar10_simple', '--output', '{output}/perplexity'],
            'description': 'GPT-2 Perplexity Evaluation',
        },
    }
    
    def __init__(self, output_dir: str, scripts_dir: str = None):
        self.output_dir = output_dir
        self.scripts_dir = scripts_dir or os.path.dirname(os.path.abspath(__file__))
        self.experiment_results = {}
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
    
    def run_experiment(self, exp_name: str) -> Dict:
        """Run a single experiment."""
        if exp_name not in self.EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}")
            return {'status': 'error', 'message': f'Unknown experiment: {exp_name}'}
        
        exp_info = self.EXPERIMENTS[exp_name]
        script_path = os.path.join(self.scripts_dir, exp_info['script'])
        
        if not os.path.exists(script_path):
            return {
                'status': 'error',
                'message': f'Script not found: {script_path}',
                'experiment': exp_name,
                'description': exp_info['description'],
            }
        
        print(f"\n{'='*80}")
        print(f"Running Experiment: {exp_name}")
        print(f"Description: {exp_info['description']}")
        print(f"Script: {script_path}")
        print(f"{'='*80}")
        
        # Prepare arguments
        args = [arg.format(output=self.run_dir) for arg in exp_info['args']]
        
        # Run script
        start_time = time.time()
        try:
            result = subprocess.run(
                ['python', script_path] + args,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            elapsed = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed_seconds': elapsed,
                'experiment': exp_name,
                'description': exp_info['description'],
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'elapsed_seconds': time.time() - start_time,
                'experiment': exp_name,
                'description': exp_info['description'],
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'elapsed_seconds': time.time() - start_time,
                'experiment': exp_name,
                'description': exp_info['description'],
            }
    
    def run_all(self, experiments: List[str] = None) -> Dict:
        """Run all experiments."""
        if experiments is None or 'all' in experiments:
            experiments = list(self.EXPERIMENTS.keys())
        
        print(f"\n{'#'*80}")
        print(f"# Running {len(experiments)} Experiments")
        print(f"# Output directory: {self.run_dir}")
        print(f"{'#'*80}")
        
        results = {
            'experiments': experiments,
            'output_dir': self.run_dir,
            'timestamp': datetime.now().isoformat(),
            'results': {},
        }
        
        for exp_name in experiments:
            result = self.run_experiment(exp_name)
            results['results'][exp_name] = result
            
            if result['status'] == 'success':
                print(f"\n[SUCCESS] {exp_name} completed in {result['elapsed_seconds']:.1f}s")
            else:
                print(f"\n[{'ERROR' if result['status'] == 'error' else result['status'].upper()}] {exp_name}")
                if 'message' in result:
                    print(f"  Message: {result['message']}")
        
        # Save results summary
        summary_file = os.path.join(self.run_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print("Experiment Summary")
        print(f"{'='*80}")
        
        success_count = sum(1 for r in results['results'].values() if r['status'] == 'success')
        print(f"Total experiments: {len(experiments)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(experiments) - success_count}")
        print(f"\nResults saved to: {summary_file}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a summary report of all experiments."""
        report = []
        report.append("=" * 80)
        report.append("ClusterSuit Experiment Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output directory: {self.run_dir}")
        report.append("=" * 80)
        
        # Iterate through results
        for exp_name, result in self.experiment_results.items():
            report.append(f"\n## {exp_name}")
            report.append(f"Status: {result.get('status', 'unknown')}")
            
            if 'description' in result:
                report.append(f"Description: {result['description']}")
            
            if 'elapsed_seconds' in result:
                report.append(f"Time: {result['elapsed_seconds']:.1f}s")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Run ClusterSuit Paper Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all experiments:
    python run_all_experiments.py --experiments all --output ./paper_experiments

  Run specific experiments:
    python run_all_experiments.py --experiments gpt2_hotspot,layer_ablation --output ./paper_experiments

Available experiments:
  - gpt2_hotspot: GPT-2 Layer-wise Computation Time Analysis
  - resnet_hotspot: ResNet-18 Layer-wise Computation Time Analysis
  - memory_overhead: Memory Overhead Measurement
  - layer_ablation: Layer Selection Ablation Experiment
  - k_value_auroc: K-Value AUROC Evaluation
  - gpt2_perplexity: GPT-2 Perplexity Evaluation
        """
    )
    
    parser.add_argument('--experiments', type=str, default='all',
                       help='Comma-separated list of experiments to run, or "all"')
    parser.add_argument('--output', type=str, default='./paper_experiments',
                       help='Output directory for results')
    parser.add_argument('--scripts_dir', type=str, default=None,
                       help='Directory containing experiment scripts')
    
    args = parser.parse_args()
    
    # Parse experiments
    experiments = [e.strip() for e in args.experiments.split(',')]
    
    # Initialize runner
    runner = ExperimentRunner(args.output, args.scripts_dir)
    
    # Run experiments
    results = runner.run_all(experiments)
    
    # Print completion message
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
