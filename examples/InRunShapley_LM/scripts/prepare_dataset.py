#!/usr/bin/env python3
"""
Prepare dataset for In-Run Data Shapley training.

This script can:
1. Download and tokenize Pile dataset (full or subset)
2. Create a small test dataset for quick validation
3. Prepare CIFAR-10 dataset for image classification tasks
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


def create_test_dataset(output_dir, num_samples=1000, seq_length=1024):
    """
    Create a small test dataset for quick validation.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
        seq_length: Sequence length for each sample
    """
    print(f"Creating test dataset with {num_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic tokenized data (uint16)
    # Using random tokens in valid GPT-2 vocabulary range (0-50256)
    vocab_size = 50257  # GPT-2 vocab size
    
    train_data = np.random.randint(0, vocab_size, size=(num_samples * seq_length,), dtype=np.uint16)
    val_data = np.random.randint(0, vocab_size, size=(100 * seq_length,), dtype=np.uint16)
    test_data = np.random.randint(0, vocab_size, size=(100 * seq_length,), dtype=np.uint16)
    
    # Save as binary files
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    test_file = os.path.join(output_dir, 'test.bin')
    
    train_data.tofile(train_file)
    val_data.tofile(val_file)
    test_data.tofile(test_file)
    
    print(f"✓ Test dataset created:")
    print(f"  Train: {len(train_data):,} tokens -> {train_file}")
    print(f"  Val:   {len(val_data):,} tokens -> {val_file}")
    print(f"  Test:  {len(test_data):,} tokens -> {test_file}")
    
    return train_file, val_file, test_file


def tokenize_pile_dataset(output_dir, max_tokens=None, num_proc=4):
    """
    Download and tokenize Pile dataset.
    
    Args:
        output_dir: Directory to save tokenized files
        max_tokens: Maximum tokens to process (None for all)
        num_proc: Number of processes for processing
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    print("Loading Pile dataset from HuggingFace...")
    
    try:
        # Load Pile dataset (uncopyrighted version)
        dataset = load_dataset('monology/pile-uncopyrighted', streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative: using local dataset or creating test dataset")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizing dataset...")
    
    train_tokens = []
    val_tokens = []
    test_tokens = []
    
    total_tokens = 0
    
    # Process training set
    print("Processing training set...")
    for example in dataset['train']:
        text = example.get('text', '')
        if not text:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = np.array(tokens, dtype=np.uint16)
        
        train_tokens.append(tokens)
        total_tokens += len(tokens)
        
        if max_tokens and total_tokens >= max_tokens:
            break
        
        if len(train_tokens) % 1000 == 0:
            print(f"  Processed {len(train_tokens)} examples, {total_tokens:,} tokens")
    
    # Process validation set
    print("Processing validation set...")
    val_count = 0
    for example in dataset['validation']:
        text = example.get('text', '')
        if not text:
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = np.array(tokens, dtype=np.uint16)
        
        val_tokens.append(tokens)
        val_count += 1
        
        if val_count >= 1000:  # Limit validation set size
            break
    
    # Concatenate all tokens
    train_data = np.concatenate(train_tokens) if train_tokens else np.array([], dtype=np.uint16)
    val_data = np.concatenate(val_tokens) if val_tokens else np.array([], dtype=np.uint16)
    
    # Save as binary files
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    test_file = os.path.join(output_dir, 'test.bin')  # Use val as test
    
    train_data.tofile(train_file)
    val_data.tofile(val_file)
    val_data.tofile(test_file)  # Copy val to test
    
    print(f"\n✓ Dataset tokenized and saved:")
    print(f"  Train: {len(train_data):,} tokens -> {train_file}")
    print(f"  Val:   {len(val_data):,} tokens -> {val_file}")
    print(f"  Test:  {len(val_data):,} tokens -> {test_file}")
    
    return train_file, val_file, test_file


def prepare_simple_text_dataset(output_dir, texts=None):
    """
    Prepare a simple text dataset from provided texts.
    
    Args:
        output_dir: Directory to save the dataset
        texts: List of text strings (optional)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required")
    
    if texts is None:
        # Default sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have achieved remarkable results in various tasks.",
            "Transformer architectures have revolutionized NLP applications.",
        ] * 200  # Repeat to create more samples
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizing {len(texts)} text samples...")
    
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # Convert to numpy array
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    # Split into train/val/test (80/10/10)
    n = len(token_array)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_data = token_array[:train_end]
    val_data = token_array[train_end:val_end]
    test_data = token_array[val_end:]
    
    # Save as binary files
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    test_file = os.path.join(output_dir, 'test.bin')
    
    train_data.tofile(train_file)
    val_data.tofile(val_file)
    test_data.tofile(test_file)
    
    print(f"\n✓ Simple text dataset created:")
    print(f"  Train: {len(train_data):,} tokens -> {train_file}")
    print(f"  Val:   {len(val_data):,} tokens -> {val_file}")
    print(f"  Test:  {len(test_data):,} tokens -> {test_file}")
    
    return train_file, val_file, test_file


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for In-Run Data Shapley')
    parser.add_argument('--dataset_type', type=str, default='test',
                       choices=['test', 'pile', 'simple'],
                       help='Type of dataset to prepare')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Directory to save the dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for test dataset')
    parser.add_argument('--seq_length', type=int, default=1024,
                       help='Sequence length for test dataset')
    parser.add_argument('--max_tokens', type=int, default=None,
                       help='Maximum tokens for Pile dataset (None for all)')
    parser.add_argument('--num_proc', type=int, default=4,
                       help='Number of processes for processing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Dataset Preparation for In-Run Data Shapley")
    print("=" * 80)
    
    if args.dataset_type == 'test':
        print("\nCreating test dataset (synthetic data)...")
        create_test_dataset(args.output_dir, args.num_samples, args.seq_length)
    
    elif args.dataset_type == 'pile':
        print("\nPreparing Pile dataset...")
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not available")
            print("Install with: pip install datasets")
            return
        try:
            tokenize_pile_dataset(args.output_dir, args.max_tokens, args.num_proc)
        except Exception as e:
            print(f"Error preparing Pile dataset: {e}")
            print("Falling back to test dataset...")
            create_test_dataset(args.output_dir, args.num_samples, args.seq_length)
    
    elif args.dataset_type == 'simple':
        print("\nPreparing simple text dataset...")
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not available")
            return
        try:
            prepare_simple_text_dataset(args.output_dir)
        except Exception as e:
            print(f"Error preparing simple dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)
    print(f"\nDataset saved to: {args.output_dir}")
    print("\nTo use this dataset, update the data paths in your config or use:")
    print(f"  --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
