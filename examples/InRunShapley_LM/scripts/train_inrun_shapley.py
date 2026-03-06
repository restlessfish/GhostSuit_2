#!/usr/bin/env python3
"""
Training script for In-Run Data Shapley with local dataset.

This script runs the complete training loop and computes Shapley values.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # InRunShapley_LM
root_dir = os.path.dirname(os.path.dirname(parent_dir))  # GhostSuite-0.35
sys.path.append(parent_dir)
sys.path.append(root_dir)

from src.simple_dataloader import SimpleDataLoader
from src.inrun_shapley_engine import InRunShapleyEngine


def create_gpt2_model_from_config(config_dict):
    """Create GPT-2 model from configuration dictionary."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        raise ImportError("transformers library required")
    
    # Create config
    gpt2_config = GPT2Config(
        vocab_size=config_dict.get('vocab_size', 50257),
        n_positions=config_dict.get('block_size', 1024),
        n_embd=config_dict.get('n_embd', 768),
        n_layer=config_dict.get('n_layer', 12),
        n_head=config_dict.get('n_head', 12),
    )
    
    # Create model
    model = GPT2LMHeadModel(gpt2_config)
    return model


def train_inrun_shapley(
    data_dir,
    output_dir,
    num_steps=100,
    batch_size=4,
    val_batch_size=1,
    block_size=512,
    learning_rate=3e-4,
    shapley_save_interval=10,
    eval_interval=20,
    device='cuda',
    warmup_steps=50,  # IMPROVEMENT: Add warmup
    use_lr_schedule=True,  # IMPROVEMENT: Use learning rate schedule
):
    """
    Main training function for In-Run Data Shapley.
    
    Args:
        data_dir: Directory containing dataset files
        output_dir: Directory to save results
        num_steps: Number of training steps
        batch_size: Training batch size
        val_batch_size: Validation batch size
        block_size: Sequence length
        learning_rate: Learning rate
        shapley_save_interval: How often to save Shapley values
        eval_interval: How often to evaluate
        device: Device to use
    """
    print("=" * 80)
    print("In-Run Data Shapley Training")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {num_steps}")
    print(f"Batch size: {batch_size}, Val batch size: {val_batch_size}")
    print("=" * 80)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print(f"\n[1/6] Loading dataset from {data_dir}...")
    try:
        dataloader = SimpleDataLoader(data_dir, block_size=block_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create model
    print(f"\n[2/6] Creating GPT-2 model...")
    # Skip pretrained download, create from config directly
    print("Creating model from config (skipping pretrained download)...")
    config = {
        'vocab_size': 50257,
        'block_size': block_size,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
    }
    model = create_gpt2_model_from_config(config)
    print("✓ Created GPT-2 model from config")
    
    model = model.to(device)
    model.train()
    model.config.use_cache = False
    
    # Apply transformers support if available
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
        print("✓ Applied transformers forward swapper")
    except ImportError:
        print("⚠ transformers_support not available, continuing without it")
    
    # Create optimizer with learning rate schedule
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    print(f"✓ Created optimizer (lr={learning_rate})")
    
    # IMPROVEMENT: Add learning rate scheduler
    scheduler = None
    if use_lr_schedule:
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"✓ Created learning rate scheduler (warmup={warmup_steps} steps)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    dot_prod_save_path = os.path.join(output_dir, 'grad_dotprods')
    os.makedirs(dot_prod_save_path, exist_ok=True)
    
    # IMPROVEMENT: Get multiple validation batches for rotation
    print(f"\n[3/6] Preparing validation batches...")
    X_val_list = []
    Y_val_list = []
    num_val_batches = min(8, max(4, val_batch_size * 2))  # Get more validation samples
    for _ in range(num_val_batches):
        X_v, Y_v = dataloader.get_batch('val', batch_size=1, device=device)
        X_val_list.append(X_v)
        Y_val_list.append(Y_v)
    
    # Use the first batch for engine attachment
    X_val = X_val_list[0]
    Y_val = Y_val_list[0]
    print(f"✓ Prepared {num_val_batches} validation batches (shape: {X_val.shape})")
    
    # Initialize In-Run Shapley engine
    print(f"\n[4/6] Initializing In-Run Shapley Engine...")
    engine = InRunShapleyEngine(
        module=model,
        val_batch_size=val_batch_size,
        loss_reduction='mean',
        use_dummy_bias=True,
        dot_prod_save_path=dot_prod_save_path,
        log_grad_norms=False,
        order=1,
        accumulate_shapley=True,
        shapley_save_interval=shapley_save_interval,
    )
    
    engine.attach(optimizer)
    engine.attach_and_store_valset(X_val, Y_val)
    print("✓ In-Run Shapley Engine initialized")
    
    # Save training configuration
    config = {
        'data_dir': data_dir,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        'block_size': block_size,
        'learning_rate': learning_rate,
        'shapley_save_interval': shapley_save_interval,
        'device': str(device),
        'start_time': datetime.now().isoformat(),
    }
    config_file = os.path.join(output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved training config to {config_file}")
    
    # Training loop
    print(f"\n[5/6] Starting training...")
    print("-" * 80)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    losses = []
    shapley_stats = []
    
    for step in range(num_steps):
        # Get training batch
        X_train, Y_train, batch_idx = dataloader.get_batch(
            'train', batch_size=batch_size, device=device, 
            return_idx=True, generator=generator
        )
        
        # IMPROVEMENT: Rotate validation samples for better coverage
        val_idx = step % len(X_val_list)
        X_val_current = X_val_list[val_idx]
        Y_val_current = Y_val_list[val_idx]
        
        # Attach batch
        engine.attach_train_batch(X_train, Y_train, step, batch_idx.cpu().tolist())
        
        # Zero gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        # Concatenate train and val for forward pass
        X_cat = torch.cat((X_train, X_val_current), dim=0)
        Y_cat = torch.cat((Y_train, Y_val_current), dim=0)
        
        # Forward and backward pass with saved_tensors context
        with engine.saved_tensors_context():
            # Forward pass
            outputs = model(input_ids=X_cat, labels=Y_cat)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
        
        # Prepare gradients
        engine.prepare_gradients()
        
        # Optimizer step
        optimizer.step()
        
        # IMPROVEMENT: Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Aggregate and log
        engine.aggregate_and_log()
        engine.clear_gradients()
        
        # IMPROVEMENT: Log learning rate
        if (step + 1) % 10 == 0 and scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            print(f" (lr={current_lr:.2e})", end='')
        
        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        
        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:4d}/{num_steps}: Loss = {loss_value:.4f}", end='')
            
            # Show Shapley stats if available
            if engine.shapley_values:
                result = engine.get_shapley_array(compact=True)
                if isinstance(result, tuple) and len(result) == 2:
                    shapley_values, _ = result
                else:
                    shapley_values = result
                if len(shapley_values) > 0:
                    mean_shap = np.mean(shapley_values)
                    std_shap = np.std(shapley_values)
                    print(f" | Shapley: mean={mean_shap:.4f}, std={std_shap:.4f}, samples={len(engine.shapley_values)}")
                else:
                    print()
            else:
                print()
        
        # Save Shapley values
        if engine.should_save_shapley(step):
            engine.save_shapley_values(step)
            print(f"  → Saved Shapley values at step {step}")
        
        # Evaluation
        if (step + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(5):  # Average over 5 batches
                    X_eval, Y_eval = dataloader.get_batch('val', batch_size=2, device=device)
                    outputs = model(input_ids=X_eval, labels=Y_eval)
                    eval_losses.append(outputs.loss.item())
                avg_eval_loss = np.mean(eval_losses)
                print(f"  → Eval loss: {avg_eval_loss:.4f}")
            model.train()
    
    # Final save
    print(f"\n[6/6] Saving final results...")
    engine.save_shapley_values(iter_num=num_steps)
    engine.save_shapley_history(iter_num=num_steps)
    
    # Save training statistics
    stats = {
        'final_loss': float(losses[-1]) if losses else None,
        'avg_loss': float(np.mean(losses)) if losses else None,
        'min_loss': float(np.min(losses)) if losses else None,
        'max_loss': float(np.max(losses)) if losses else None,
        'total_steps': num_steps,
        'end_time': datetime.now().isoformat(),
    }
    
    if engine.shapley_values:
        result = engine.get_shapley_array(compact=True)
        if isinstance(result, tuple) and len(result) == 2:
            shapley_values, _ = result
        else:
            shapley_values = result
        if len(shapley_values) > 0:
            stats['shapley'] = {
                'num_samples': len(engine.shapley_values),
                'mean': float(np.mean(shapley_values)),
                'std': float(np.std(shapley_values)),
                'min': float(np.min(shapley_values)),
                'max': float(np.max(shapley_values)),
                'positive_count': int(np.sum(shapley_values > 0)),
                'negative_count': int(np.sum(shapley_values < 0)),
            }
        else:
            stats['shapley'] = {
                'num_samples': len(engine.shapley_values),
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'positive_count': 0,
                'negative_count': 0,
            }
    
    stats_file = os.path.join(output_dir, 'training_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved training stats to {stats_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    if losses:
        print(f"\nLoss statistics:")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        print(f"  Min loss: {np.min(losses):.4f}")
        print(f"  Max loss: {np.max(losses):.4f}")
    
    if engine.shapley_values:
        shapley_array = engine.get_shapley_array()
        print(f"\nShapley value statistics:")
        print(f"  Total samples: {len(engine.shapley_values)}")
        print(f"  Mean: {np.mean(shapley_array):.6f}")
        print(f"  Std:  {np.std(shapley_array):.6f}")
        print(f"  Min:  {np.min(shapley_array):.6f}")
        print(f"  Max:  {np.max(shapley_array):.6f}")
        print(f"  Positive: {np.sum(shapley_array > 0)}")
        print(f"  Negative: {np.sum(shapley_array < 0)}")
        
        # Show top and bottom samples
        sorted_indices = np.argsort(shapley_array)[::-1]
        print(f"\nTop 5 samples by Shapley value:")
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"  {i+1}. Sample {idx}: {shapley_array[idx]:.6f}")
        
        print(f"\nBottom 5 samples by Shapley value:")
        for i, idx in enumerate(sorted_indices[-5:]):
            print(f"  {i+1}. Sample {idx}: {shapley_array[idx]:.6f}")
    
    # Cleanup
    engine.detach()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train In-Run Data Shapley')
    parser.add_argument('--data_dir', type=str, default='./data/test_dataset',
                       help='Directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./results/inrun_shapley_train',
                       help='Directory to save results')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                       help='Validation batch size')
    parser.add_argument('--block_size', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--shapley_save_interval', type=int, default=10,
                       help='How often to save Shapley values')
    parser.add_argument('--eval_interval', type=int, default=20,
                       help='How often to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please run prepare_dataset.py first to create the dataset")
        sys.exit(1)
    
    try:
        stats = train_inrun_shapley(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            block_size=args.block_size,
            learning_rate=args.learning_rate,
            shapley_save_interval=args.shapley_save_interval,
            eval_interval=args.eval_interval,
            device=args.device
        )
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
