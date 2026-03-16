#!/usr/bin/env python3
"""
Train ResNet18 on CIFAR-10 with CIFAR-10N noisy labels and compute
first-order In-Run Data Shapley values with GhostSuit.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from examples.InRunShapley_LM.src.inrun_shapley_engine import InRunShapleyEngine

CIFAR10N_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-10_human.pt"


def ensure_cifar10n_labels(noise_file: str):
    if os.path.exists(noise_file):
        return
    os.makedirs(os.path.dirname(noise_file), exist_ok=True)
    print(f"Downloading CIFAR-10N labels to {noise_file}...")
    urllib.request.urlretrieve(CIFAR10N_URL, noise_file)
    print("Downloaded CIFAR-10N labels.")


class IndexedNoisyCIFAR10(Dataset):
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
        self.mislabeled_mask = self.noisy_targets != self.clean_targets
        self.mislabeled_indices = np.flatnonzero(self.mislabeled_mask)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, _ = self.dataset[idx]
        label = int(self.noisy_targets[idx] if self.train else self.clean_targets[idx])
        return image, label, idx


def disable_inplace_ops(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def train_cifar10_resnet18(
    data_root: str,
    output_dir: str,
    num_steps: int = 5000,
    batch_size: int = 128,
    val_batch_size: int = 32,
    learning_rate: float = 1e-3,
    noise_type: str = "aggre_label",
    shapley_order: int = 1,
    shapley_save_interval: int = 1000,
    eval_interval: int = 500,
    num_workers: int = 4,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    dot_prod_save_path = os.path.join(output_dir, "grad_dotprods")
    os.makedirs(dot_prod_save_path, exist_ok=True)

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

    train_dataset = IndexedNoisyCIFAR10(root=data_root, noise_type=noise_type, train=True, download=True, transform=train_transform)
    test_dataset = IndexedNoisyCIFAR10(root=data_root, noise_type=noise_type, train=False, download=False, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_mislabeled = len(train_dataset.mislabeled_indices)
    mislabeled_file = os.path.join(output_dir, "mislabeled_indices.npy")
    np.save(mislabeled_file, train_dataset.mislabeled_indices)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Mislabeled: {num_mislabeled}. Saved {mislabeled_file}")

    model = resnet18(num_classes=10)
    disable_inplace_ops(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    engine = InRunShapleyEngine(
        module=model,
        val_batch_size=val_batch_size,
        loss_reduction="mean",
        use_dummy_bias=False,
        dot_prod_save_path=dot_prod_save_path,
        log_grad_norms=False,
        order=shapley_order,
        accumulate_shapley=True,
        shapley_save_interval=shapley_save_interval,
    )
    engine.attach(optimizer)

    metadata = {
        "dataset": "CIFAR-10", "noise_source": "CIFAR-10N", "noise_type": noise_type,
        "num_train": len(train_dataset), "num_test": len(test_dataset), "num_mislabeled": num_mislabeled,
        "num_steps": num_steps, "batch_size": batch_size, "val_batch_size": val_batch_size,
        "shapley_order": shapley_order,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    model.train()
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    losses = []
    test_accs = []
    start = time.time()

    for step in range(num_steps):
        try:
            X_train, Y_train, train_indices = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X_train, Y_train, train_indices = next(train_iter)
        try:
            X_val, Y_val, _ = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X_val, Y_val, _ = next(val_iter)

        X_train, Y_train = X_train.to(device), Y_train.to(device)
        X_val, Y_val = X_val.to(device), Y_val.to(device)
        train_indices = train_indices.numpy().tolist()

        engine.attach_train_batch(X_train, Y_train, step, train_indices)
        optimizer.zero_grad()
        X_cat = torch.cat((X_train, X_val), dim=0)
        Y_cat = torch.cat((Y_train, Y_val), dim=0)
        with engine.saved_tensors_context():
            logits = model(X_cat)
            loss = F.cross_entropy(logits, Y_cat)
            loss.backward()
        engine.prepare_gradients()
        optimizer.step()
        scheduler.step()
        engine.aggregate_and_log()
        engine.clear_gradients()
        losses.append(loss.item())

        if (step + 1) % eval_interval == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_t, Y_t, _ in val_loader:
                    X_t, Y_t = X_t.to(device), Y_t.to(device)
                    pred = model(X_t).argmax(1)
                    total += Y_t.size(0)
                    correct += pred.eq(Y_t).sum().item()
            acc = correct / total
            test_accs.append(acc)
            model.train()
            print(f"Step {step+1:5d}/{num_steps}: Loss = {loss.item():.4f} | Test Acc = {acc:.4f}")
        elif (step + 1) % 50 == 0:
            print(f"Step {step+1:5d}/{num_steps}: Loss = {loss.item():.4f}")

        if engine.should_save_shapley(step + 1):
            engine.save_shapley_values(step + 1)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_t, Y_t, _ in val_loader:
            X_t, Y_t = X_t.to(device), Y_t.to(device)
            pred = model(X_t).argmax(1)
            total += Y_t.size(0)
            correct += pred.eq(Y_t).sum().item()
    final_acc = correct / total
    engine.save_shapley_values(num_steps)
    elapsed = time.time() - start

    stats = {
        "final_loss": losses[-1] if losses else None,
        "final_test_acc": final_acc,
        "best_test_acc": max(test_accs) if test_accs else final_acc,
        "total_steps": num_steps,
        "total_time": elapsed,
    }
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Training complete. Time: {elapsed/60:.2f} min. Best test acc: {stats['best_test_acc']:.4f}")
    engine.cleanup()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/cifar10")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--noise_type", type=str, default="aggre_label")
    parser.add_argument("--shapley_order", type=int, default=1)
    parser.add_argument("--shapley_save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train_cifar10_resnet18(
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
