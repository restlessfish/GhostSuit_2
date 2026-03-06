#!/usr/bin/env python3
"""
Run In-Run Data Shapley training with local dataset.

This script demonstrates how to use the simple dataloader with local binary files.
"""

import os
import sys
import torch
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple_dataloader import SimpleDataLoader
from inrun_shapley_engine import InRunShapleyEngine
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW


def run_training_with_local_data(data_dir, output_dir, num_steps=10):
    """
    Run training with local dataset.
    
    Args:
        data_dir: Directory containing train.bin, val.bin, test.bin
        output_dir: Directory to save results
        num_steps: Number of training steps
    """
    print("=" * 80)
    print("In-Run Data Shapley Training with Local Dataset")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataloader = SimpleDataLoader(data_dir, block_size=512)  # Smaller block size for testing
    
    # Load model
    print("\nLoading GPT-2 model...")
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: Model download requires internet connection")
        return
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.train()
    model.config.use_cache = False
    
    # Apply transformers support if available
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
        print("Applied transformers forward swapper")
    except ImportError:
        print("Warning: transformers_support not available")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    dot_prod_save_path = os.path.join(output_dir, 'grad_dotprods')
    os.makedirs(dot_prod_save_path, exist_ok=True)
    
    # Get validation batch
    print("\nPreparing validation batch...")
    X_val, Y_val = dataloader.get_batch('val', batch_size=2, device=device)
    
    # Initialize In-Run Shapley engine
    print("\nInitializing In-Run Shapley Engine...")
    engine = InRunShapleyEngine(
        module=model,
        val_batch_size=2,
        loss_reduction='mean',
        use_dummy_bias=True,
        dot_prod_save_path=dot_prod_save_path,
        log_grad_norms=False,
        order=1,
        accumulate_shapley=True,
        shapley_save_interval=5,
    )
    
    engine.attach(optimizer)
    engine.attach_and_store_valset(X_val, Y_val)
    
    # Training loop
    print(f"\nRunning {num_steps} training steps...")
    print("-" * 80)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    for step in range(num_steps):
        # Get training batch
        X_train, Y_train, batch_idx = dataloader.get_batch(
            'train', batch_size=4, device=device, return_idx=True, generator=generator
        )
        
        # Attach batch
        engine.attach_train_batch(X_train, Y_train, step, batch_idx.cpu().tolist())
        
        # Zero gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        # Concatenate train and val for forward pass
        X_cat = torch.cat((X_train, X_val), dim=0)
        Y_cat = torch.cat((Y_train, Y_val), dim=0)
        
        # Forward pass
        outputs = model(input_ids=X_cat, labels=Y_cat)
        loss = outputs.loss
        
        print(f"Step {step+1}/{num_steps}: Loss = {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Prepare gradients
        engine.prepare_gradients()
        
        # Optimizer step
        optimizer.step()
        
        # Aggregate and log
        engine.aggregate_and_log()
        engine.clear_gradients()
        
        # Save if needed
        if engine.should_save_shapley(step):
            engine.save_shapley_values(step)
            print(f"  → Saved Shapley values at step {step}")
    
    # Final save
    print("\nSaving final results...")
    engine.save_shapley_values(iter_num=num_steps)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    shapley_values = engine.get_accumulated_shapley_values()
    print(f"\nTotal samples with Shapley values: {len(shapley_values)}")
    
    if shapley_values:
        shapley_array = engine.get_shapley_array()
        print(f"\nShapley value statistics:")
        print(f"  Mean: {np.mean(shapley_array):.6f}")
        print(f"  Std:  {np.std(shapley_array):.6f}")
        print(f"  Min:  {np.min(shapley_array):.6f}")
        print(f"  Max:  {np.max(shapley_array):.6f}")
        print(f"  Positive: {np.sum(shapley_array > 0)}")
        print(f"  Negative: {np.sum(shapley_array < 0)}")
        
        # Show top samples
        sorted_indices = np.argsort(shapley_array)[::-1]
        print(f"\nTop 5 samples by Shapley value:")
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"  {i+1}. Sample {idx}: {shapley_array[idx]:.6f}")
    
    # Cleanup
    engine.detach()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run In-Run Shapley with local dataset')
    parser.add_argument('--data_dir', type=str, default='./data/test_dataset',
                       help='Directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./results/inrun_shapley_test',
                       help='Directory to save results')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of training steps')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please run prepare_dataset.py first to create the dataset")
        sys.exit(1)
    
    try:
        run_training_with_local_data(args.data_dir, args.output_dir, args.num_steps)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
