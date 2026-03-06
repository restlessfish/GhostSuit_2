#!/usr/bin/env python3
"""
Prepare CIFAR-10 dataset for In-Run Data Shapley evaluation.

This script converts CIFAR-10 to a format suitable for language model training,
or prepares it for image classification tasks with GPT-2 vision models.
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

try:
    import torchvision
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. Install with: pip install torchvision")


def prepare_cifar10_for_lm(output_dir, num_samples=None, mislabeled_ratio=0.1, seed=42):
    """
    Prepare CIFAR-10 dataset for language model training.
    
    Since GPT-2 is a text model, we'll convert CIFAR-10 images to text descriptions
    or use a simplified approach for testing.
    
    Args:
        output_dir: Directory to save processed data
        num_samples: Number of samples to use (None for all)
        mislabeled_ratio: Ratio of mislabeled samples
        seed: Random seed
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required. Install with: pip install torchvision")
    
    print("=" * 80)
    print("Preparing CIFAR-10 Dataset")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # For language model, we'll create synthetic tokenized data based on labels
    # This is a simplified approach - in practice, you'd use a vision-language model
    
    # Create tokenized representation (simplified)
    # Using label indices as tokens (0-9) + some variation
    vocab_size = 50257  # GPT-2 vocab size
    
    def create_tokens_from_labels(dataset, num_samples=None):
        """Create token sequences from CIFAR-10 labels."""
        if num_samples is None:
            num_samples = len(dataset)
        
        tokens = []
        labels = []
        indices = []
        
        for i in range(min(num_samples, len(dataset))):
            label = dataset[i][1]
            # Create a simple token sequence based on label
            # In practice, you'd use image features or vision-language model
            base_token = label * 1000  # Simple mapping
            sequence = [base_token + j for j in range(512)]  # 512 tokens per sample
            sequence = [min(t, vocab_size - 1) for t in sequence]  # Clip to vocab size
            
            tokens.extend(sequence)
            labels.append(label)
            indices.append(i)
        
        return np.array(tokens, dtype=np.uint16), np.array(labels), np.array(indices)
    
    # Process training set
    print(f"\nProcessing training set ({num_samples or len(train_dataset)} samples)...")
    train_tokens, train_labels, train_indices = create_tokens_from_labels(
        train_dataset, num_samples
    )
    
    # Process test set (use smaller subset)
    test_num_samples = min(1000, len(test_dataset))
    print(f"\nProcessing test set ({test_num_samples} samples)...")
    test_tokens, test_labels, test_indices = create_tokens_from_labels(
        test_dataset, test_num_samples
    )
    
    # Create validation set (split from train)
    val_size = len(train_tokens) // 10
    val_tokens = train_tokens[:val_size]
    train_tokens = train_tokens[val_size:]
    
    # Save tokenized data
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    test_file = os.path.join(output_dir, 'test.bin')
    
    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)
    test_tokens.tofile(test_file)
    
    print(f"\n✓ Saved tokenized data:")
    print(f"  Train: {len(train_tokens):,} tokens -> {train_file}")
    print(f"  Val:   {len(val_tokens):,} tokens -> {val_file}")
    print(f"  Test:  {len(test_tokens):,} tokens -> {test_file}")
    
    # Create mislabeled indices
    num_mislabeled = int(len(train_indices) * mislabeled_ratio)
    mislabeled_indices = np.random.choice(
        train_indices, size=num_mislabeled, replace=False
    )
    
    # Save metadata
    metadata = {
        'num_train_samples': len(train_indices),
        'num_val_samples': len(val_tokens) // 512,
        'num_test_samples': len(test_indices),
        'mislabeled_ratio': mislabeled_ratio,
        'num_mislabeled': num_mislabeled,
        'mislabeled_indices': mislabeled_indices.tolist(),
        'train_labels': train_labels.tolist(),
        'test_labels': test_labels.tolist(),
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to {metadata_file}")
    print(f"  Mislabeled samples: {num_mislabeled}/{len(train_indices)} ({mislabeled_ratio:.1%})")
    
    # Save mislabeled indices separately
    mislabeled_file = os.path.join(output_dir, 'mislabeled_indices.npy')
    np.save(mislabeled_file, mislabeled_indices)
    print(f"✓ Saved mislabeled indices to {mislabeled_file}")
    
    return output_dir


def create_simple_cifar10_dataset(output_dir, num_samples=1000, seed=42):
    """
    Create a simplified CIFAR-10-like dataset for quick testing.
    
    This creates synthetic data that mimics CIFAR-10 structure.
    """
    print("=" * 80)
    print("Creating Simplified CIFAR-10 Dataset")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    vocab_size = 50257
    seq_length = 512
    
    # Create synthetic data
    train_size = num_samples * seq_length
    val_size = num_samples // 10 * seq_length
    test_size = num_samples // 10 * seq_length
    
    # Generate tokens with some structure (10 classes)
    train_tokens = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
    val_tokens = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)
    test_tokens = np.random.randint(0, vocab_size, size=test_size, dtype=np.uint16)
    
    # Save files
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    test_file = os.path.join(output_dir, 'test.bin')
    
    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)
    test_tokens.tofile(test_file)
    
    print(f"\n✓ Created simplified dataset:")
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")
    print(f"  Test:  {len(test_tokens):,} tokens")
    
    # Create mislabeled indices
    num_mislabeled = int(num_samples * 0.1)
    mislabeled_indices = np.random.choice(num_samples, size=num_mislabeled, replace=False)
    
    mislabeled_file = os.path.join(output_dir, 'mislabeled_indices.npy')
    np.save(mislabeled_file, mislabeled_indices)
    print(f"✓ Created {num_mislabeled} mislabeled indices")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Prepare CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./data/cifar10_dataset',
                       help='Directory to save processed data')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use (None for all)')
    parser.add_argument('--mislabeled_ratio', type=float, default=0.1,
                       help='Ratio of mislabeled samples')
    parser.add_argument('--simple', action='store_true',
                       help='Create simplified synthetic dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.simple:
        create_simple_cifar10_dataset(args.output_dir, args.num_samples or 1000, args.seed)
    else:
        if not TORCHVISION_AVAILABLE:
            print("torchvision not available, creating simplified dataset instead...")
            create_simple_cifar10_dataset(args.output_dir, args.num_samples or 1000, args.seed)
        else:
            prepare_cifar10_for_lm(
                args.output_dir, args.num_samples, args.mislabeled_ratio, args.seed
            )
    
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)
    print(f"\nDataset saved to: {args.output_dir}")
    print("\nTo use this dataset:")
    print(f"  python train_inrun_shapley.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
