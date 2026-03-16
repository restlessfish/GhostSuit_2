#!/usr/bin/env python3
"""
Train ResNet18 on CIFAR-10 with real noisy labels from CIFAR-10N and compute
first-order or second-order In-Run Data Shapley values with GhostSuit.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
from examples.InRunShapley_LM.src.inrun_shapley_engine import InRunShapleyEngine


CIFAR10N_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-10_human.pt"


def ensure_cifar10n_labels(noise_file: str):
    """Download CIFAR-10N labels if not present."""
    if os.path.exists(noise_file):
        return
    
    print(f"Downloading CIFAR-10N labels to {noise_file}...")
    os.makedirs(os.path.dirname(noise_file), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(CIFAR10N_URL, noise_file)
        print(f"✓ Downloaded CIFAR-10N labels")
    except Exception as e:
        raise RuntimeError(f"Failed to download CIFAR-10N labels: {e}")


class IndexedNoisyCIFAR10(Dataset):
    """CIFAR-10 dataset with CIFAR-10N noisy labels and sample indices."""
    
    def __init__(self, root: str, noise_type: str, train: bool, download: bool, transform):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        self.train = train
        self.clean_targets = np.array(self.dataset.targets, dtype=np.int64)
        self.noisy_targets = self.clean_targets.copy()
        
        if train:
            noise_file = os.path.join(root, "CIFAR-10_human.pt")
            ensure_cifar10n_labels(noise_file)
            noise_data = torch.load(noise_file, map_location="cpu", weights_only=False)
            if noise_type not in noise_data:
                raise ValueError(f"Unknown noise_type={noise_type}. Available: {list(noise_data.keys())}")
            self.noisy_targets = np.asarray(noise_data[noise_type], dtype=np.int64)
            if self.noisy_targets.shape[0] != self.clean_targets.shape[0]:
                raise ValueError("CIFAR-10N label count does not match CIFAR-10 train split")
        
        self.mislabeled_mask = self.noisy_targets != self.clean_targets
        self.mislabeled_indices = np.flatnonzero(self.mislabeled_mask)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        image, _ = self.dataset[idx]
        label = int(self.noisy_targets[idx] if self.train else self.clean_targets[idx])
        return image, label, idx


def disable_inplace_ops(module: nn.Module) -> None:
    """Disable inplace operations to avoid issues with autograd hooks."""
    for submodule in module.modules():
        if hasattr(submodule, "inplace"):
            submodule.inplace = False


def train_cifar10_resnet18(
    data_root: str,
    output_dir: str,
    num_steps: int = 1000,
    batch_size: int = 128,
    val_batch_size: int = 32,
    learning_rate: float = 1e-3,
    noise_type: str = "aggre_label",
    shapley_order: int = 1,
    shapley_save_interval: int = 200,
    eval_interval: int = 200,
    num_workers: int = 4,
    device: str = "cuda",
):
    """Train ResNet18 on CIFAR-10N with In-Run Shapley."""
    
    print("=" * 80)
    print("CIFAR-10 / ResNet18 / In-Run Data Shapley Training")
    print("=" * 80)
    print(f"Order: {shapley_order} ({'first-order' if shapley_order == 1 else 'second-order'})")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Steps: {num_steps}, Batch size: {batch_size}, Val batch: {val_batch_size}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    dot_prod_save_path = os.path.join(output_dir, "grad_dotprods")
    os.makedirs(dot_prod_save_path, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    print("\n[1/5] Loading CIFAR-10 datasets...")
    train_dataset = IndexedNoisyCIFAR10(
        root=data_root,
        noise_type=noise_type,
        train=True,
        download=True,
        transform=train_transform,
    )
    
    test_dataset = IndexedNoisyCIFAR10(
        root=data_root,
        noise_type=noise_type,
        train=False,
        download=False,
        transform=val_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    num_mislabeled = len(train_dataset.mislabeled_indices)
    mislabeled_ratio = num_mislabeled / len(train_dataset)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Mislabeled: {num_mislabeled} ({mislabeled_ratio:.2%})")
    
    # Save mislabeled indices
    mislabeled_indices_file = os.path.join(output_dir, "mislabeled_indices.npy")
    np.save(mislabeled_indices_file, train_dataset.mislabeled_indices)
    print(f"✓ Saved mislabeled indices to {mislabeled_indices_file}")
    
    # Create model
    print("\n[2/5] Creating ResNet18 model...")
    model = resnet18(num_classes=10)
    disable_inplace_ops(model)  # Important for autograd hooks
    model = model.to(device)
    print("✓ ResNet18 model created")
    
    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
    )
    
    # Initialize Shapley engine
    print("\n[3/5] Initializing In-Run Shapley Engine...")
    engine = InRunShapleyEngine(
        module=model,
        val_batch_size=val_batch_size,
        loss_reduction='mean',
        use_dummy_bias=False,
        dot_prod_save_path=dot_prod_save_path,
        log_grad_norms=False,
        order=shapley_order,
        accumulate_shapley=True,
        shapley_save_interval=shapley_save_interval,
    )
    
    engine.attach(optimizer)
    print("✓ In-Run Shapley Engine initialized")
    
    # Save metadata
    metadata = {
        "dataset": "CIFAR-10",
        "noise_source": "CIFAR-10N",
        "noise_type": noise_type,
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "num_mislabeled": num_mislabeled,
        "mislabeled_ratio": mislabeled_ratio,
        "device": device,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "num_steps": num_steps,
        "shapley_order": shapley_order,
    }
    
    metadata_file = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    
    # Training loop
    print("\n[4/5] Starting training...")
    print("-" * 80)
    
    model.train()
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    losses = []
    test_accs = []
    start_time = time.time()
    
    for step in range(num_steps):
        # Get training batch
        try:
            X_train, Y_train, train_indices = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X_train, Y_train, train_indices = next(train_iter)
        
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        train_indices = train_indices.numpy()
        
        # Get validation batch
        try:
            X_val, Y_val, _ = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X_val, Y_val, _ = next(val_iter)
        
        X_val = X_val.to(device)
        Y_val = Y_val.to(device)
        
        # Attach batch
        engine.attach_train_batch(X_train, Y_train, step, train_indices.tolist())
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Concatenate train and val for forward pass
        X_cat = torch.cat((X_train, X_val), dim=0)
        Y_cat = torch.cat((Y_train, Y_val), dim=0)
        
        # Forward and backward pass (and optional second-order HVP) under the same
        # saved_tensors_context so that activations needed for HVP are captured.
        with engine.saved_tensors_context():
            # For second-order method, compute validation loss and HVP first.
            if engine.order == 2:
                model.eval()
                with torch.enable_grad():
                    val_outputs = model(X_val)
                    val_loss = F.cross_entropy(val_outputs, Y_val)
                    engine.set_validation_loss(val_loss)
                model.train()
            
            outputs = model(X_cat)
            loss = F.cross_entropy(outputs, Y_cat)
            loss.backward()
        
        # Prepare gradients
        engine.prepare_gradients()
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Aggregate and log
        engine.aggregate_and_log()
        engine.clear_gradients()
        
        losses.append(loss.item())
        
        # Evaluation
        if (step + 1) % eval_interval == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_test, Y_test, _ in val_loader:
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                    outputs = model(X_test)
                    _, predicted = outputs.max(1)
                    total += Y_test.size(0)
                    correct += predicted.eq(Y_test).sum().item()
            
            test_acc = correct / total
            test_accs.append(test_acc)
            model.train()
            print(f"Step {step+1:5d}/{num_steps}: Loss = {loss.item():.4f} | Test Acc = {test_acc:.4f}")
        else:
            if (step + 1) % 50 == 0:
                print(f"Step {step+1:5d}/{num_steps}: Loss = {loss.item():.4f}")
        
        # Save Shapley values
        if engine.should_save_shapley(step + 1):
            engine.save_shapley_values(step + 1)
    
    # Final evaluation
    print("\n[5/5] Final evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_test, Y_test, _ in val_loader:
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            outputs = model(X_test)
            _, predicted = outputs.max(1)
            total += Y_test.size(0)
            correct += predicted.eq(Y_test).sum().item()
    
    final_test_acc = correct / total
    
    # Save final Shapley values
    engine.save_shapley_values(num_steps)
    
    # Save training stats
    stats = {
        "final_loss": losses[-1] if losses else None,
        "avg_loss": np.mean(losses) if losses else None,
        "min_loss": np.min(losses) if losses else None,
        "max_loss": np.max(losses) if losses else None,
        "final_test_acc": final_test_acc,
        "best_test_acc": max(test_accs) if test_accs else final_test_acc,
        "total_steps": num_steps,
        "total_time": time.time() - start_time,
    }
    
    stats_file = os.path.join(output_dir, "training_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {stats['total_time']/60:.2f} min")
    print(f"Best test accuracy: {stats['best_test_acc']:.4f}")
    print("=" * 80)
    
    # Cleanup
    engine.cleanup()
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10N with In-Run Shapley')
    parser.add_argument('--data_root', type=str, default='./data/cifar10',
                       help='Root directory for CIFAR-10 data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=32,
                       help='Validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--noise_type', type=str, default='aggre_label',
                       choices=['aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3'],
                       help='CIFAR-10N noise type')
    parser.add_argument('--shapley_order', type=int, default=1, choices=[1, 2],
                       help='Order of Shapley approximation (1 for first-order, 2 for second-order)')
    parser.add_argument('--shapley_save_interval', type=int, default=200,
                       help='How often to save Shapley values')
    parser.add_argument('--eval_interval', type=int, default=200,
                       help='How often to evaluate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    try:
        stats = train_cifar10_resnet18(
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            learning_rate=args.learning_rate,
            noise_type=args.noise_type,
            shapley_order=args.shapley_order,
            shapley_save_interval=args.shapley_save_interval,
            eval_interval=args.eval_interval,
            num_workers=args.num_workers,
            device=args.device,
        )
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
