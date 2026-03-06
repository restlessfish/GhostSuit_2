"""
Simple test script to verify In-Run Data Shapley implementation.
Uses a small GPT-2 model with synthetic data for quick testing.
"""

import os
import sys
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inrun_shapley_engine import InRunShapleyEngine
from torch.optim import AdamW


def create_synthetic_data(tokenizer, num_samples=20, seq_length=128):
    """Create synthetic text data for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Natural language processing enables understanding.",
        "Deep learning models achieve remarkable results.",
        "Transformer architectures revolutionized NLP.",
    ] * (num_samples // 5 + 1)
    texts = texts[:num_samples]
    
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=seq_length
    )
    
    # Create labels (shifted input ids for language modeling)
    labels = inputs['input_ids'].clone()
    labels[inputs['attention_mask'] == 0] = -100
    
    return inputs, labels


def test_inrun_shapley():
    """Test In-Run Shapley implementation."""
    print("=" * 80)
    print("Testing In-Run Data Shapley Implementation")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load small GPT-2 model
    print("\nLoading GPT-2 model...")
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
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
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    train_inputs, train_labels = create_synthetic_data(tokenizer, num_samples=20, seq_length=64)
    val_inputs, val_labels = create_synthetic_data(tokenizer, num_samples=2, seq_length=64)
    
    train_inputs = {k: v.to(device) for k, v in train_inputs.items()}
    train_labels = train_labels.to(device)
    val_inputs = {k: v.to(device) for k, v in val_inputs.items()}
    val_labels = val_labels.to(device)
    
    # Create output directory
    output_dir = './test_inrun_shapley_output'
    os.makedirs(output_dir, exist_ok=True)
    dot_prod_save_path = os.path.join(output_dir, 'grad_dotprods')
    os.makedirs(dot_prod_save_path, exist_ok=True)
    
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
        shapley_save_interval=2,
    )
    
    engine.attach(optimizer)
    engine.attach_and_store_valset(
        val_inputs['input_ids'], 
        val_labels
    )
    
    # Run a few training steps
    print("\nRunning training steps...")
    num_steps = 5
    batch_size = 4
    
    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        
        # Get a batch
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(train_inputs['input_ids']))
        
        if start_idx >= len(train_inputs['input_ids']):
            break
        
        X_batch = {k: v[start_idx:end_idx] for k, v in train_inputs.items()}
        Y_batch = train_labels[start_idx:end_idx]
        batch_idx = list(range(start_idx, end_idx))
        
        # Attach batch
        engine.attach_train_batch(X_batch['input_ids'], Y_batch, step, batch_idx)
        
        # Zero gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        # Concatenate train and val for forward pass
        X_cat = torch.cat((X_batch['input_ids'], val_inputs['input_ids']), dim=0)
        Y_cat = torch.cat((Y_batch, val_labels), dim=0)
        
        # Forward pass
        outputs = model(input_ids=X_cat, labels=Y_cat)
        loss = outputs.loss
        
        print(f"Loss: {loss.item():.4f}")
        
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
            engine.save_shapley_history(step)
    
    # Final save
    print("\nSaving final Shapley values...")
    engine.save_shapley_values(iter_num=num_steps)
    
    # Check results
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    shapley_values = engine.get_accumulated_shapley_values()
    print(f"\nTotal samples with Shapley values: {len(shapley_values)}")
    
    if shapley_values:
        shapley_array = engine.get_shapley_array()
        print(f"Shapley value statistics:")
        print(f"  Mean: {np.mean(shapley_array):.6f}")
        print(f"  Std:  {np.std(shapley_array):.6f}")
        print(f"  Min:  {np.min(shapley_array):.6f}")
        print(f"  Max:  {np.max(shapley_array):.6f}")
        print(f"  Positive values: {np.sum(shapley_array > 0)}")
        print(f"  Negative values: {np.sum(shapley_array < 0)}")
        
        # Show top and bottom samples
        sorted_indices = np.argsort(shapley_array)[::-1]
        print(f"\nTop 5 samples by Shapley value:")
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"  {i+1}. Sample {idx}: {shapley_array[idx]:.6f}")
        
        print(f"\nBottom 5 samples by Shapley value:")
        for i, idx in enumerate(sorted_indices[-5:]):
            print(f"  {i+1}. Sample {idx}: {shapley_array[idx]:.6f}")
    
    # Check saved files
    print(f"\nChecking saved files in {dot_prod_save_path}...")
    import glob
    shapley_files = glob.glob(os.path.join(dot_prod_save_path, "shapley_*.pt"))
    shapley_arrays = glob.glob(os.path.join(dot_prod_save_path, "shapley_*.npy"))
    print(f"  Found {len(shapley_files)} Shapley value files")
    print(f"  Found {len(shapley_arrays)} Shapley array files")
    
    # Cleanup
    engine.detach()
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
    return shapley_values


if __name__ == "__main__":
    try:
        shapley_values = test_inrun_shapley()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
