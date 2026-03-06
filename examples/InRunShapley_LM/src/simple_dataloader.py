"""
Simple dataloader for In-Run Shapley that works with local binary files.
"""

import os
import sys
import numpy as np
import torch
from typing import Optional, Tuple

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SimpleDataLoader:
    """Simple dataloader for binary tokenized data."""
    
    def __init__(self, data_dir: str, block_size: int = 1024):
        """
        Initialize dataloader.
        
        Args:
            data_dir: Directory containing train.bin, val.bin, test.bin
            block_size: Sequence length
        """
        self.data_dir = data_dir
        self.block_size = block_size
        
        # Load data files
        train_file = os.path.join(data_dir, 'train.bin')
        val_file = os.path.join(data_dir, 'val.bin')
        test_file = os.path.join(data_dir, 'test.bin')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found: {train_file}")
        
        # Load as memory-mapped arrays
        self.train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
        self.val_data = np.memmap(val_file, dtype=np.uint16, mode='r') if os.path.exists(val_file) else self.train_data
        self.test_data = np.memmap(test_file, dtype=np.uint16, mode='r') if os.path.exists(test_file) else self.val_data
        
        print(f"Loaded dataset:")
        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val:   {len(self.val_data):,} tokens")
        print(f"  Test:  {len(self.test_data):,} tokens")
    
    def get_batch(self, split: str, batch_size: int, device: str = 'cuda',
                  return_idx: bool = False, generator: Optional[torch.Generator] = None):
        """
        Get a batch of data.
        
        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            device: Device to move data to
            return_idx: Whether to return indices
            generator: Random generator for reproducibility
            
        Returns:
            X, Y (and optionally indices)
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        elif split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Generate random indices
        max_idx = len(data) - self.block_size
        if max_idx <= 0:
            raise ValueError(f"Data too short for block_size {self.block_size}")
        
        if generator is not None:
            ix = torch.randint(0, max_idx, (batch_size,), generator=generator)
        else:
            ix = torch.randint(0, max_idx, (batch_size,))
        
        # Extract sequences
        x = torch.stack([
            torch.from_numpy((data[i:i+self.block_size]).astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64))
            for i in ix
        ])
        
        # Move to device
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        
        if return_idx:
            return x, y, ix
        else:
            return x, y


def load_simple_dataset(data_dir: str, block_size: int = 1024):
    """
    Load simple dataset from directory.
    
    Args:
        data_dir: Directory containing binary files
        block_size: Sequence length
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    loader = SimpleDataLoader(data_dir, block_size)
    
    return {
        'train': loader,
        'val': loader,
        'test': loader
    }
