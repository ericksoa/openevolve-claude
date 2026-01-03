"""
Training script for packing value function.

Usage:
    python train.py --data training_data.jsonl --epochs 100
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

from model import create_model


class PackingDataset(Dataset):
    """Dataset of packing states and final side lengths."""

    def __init__(self, jsonl_path: str, max_n: int = 50):
        self.samples = []
        self.max_n = max_n

        print(f"Loading data from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                features = torch.tensor(data['features'], dtype=torch.float32)
                target_n = torch.tensor([data['target_n']], dtype=torch.float32)
                final_side = torch.tensor([data['final_side']], dtype=torch.float32)

                self.samples.append((features, target_n, final_side))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for features, target_n, final_side in dataloader:
        features = features.to(device)
        target_n = target_n.to(device)
        final_side = final_side.to(device)

        optimizer.zero_grad()
        pred = model(features, target_n)
        loss = criterion(pred, final_side)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0

    with torch.no_grad():
        for features, target_n, final_side in dataloader:
            features = features.to(device)
            target_n = target_n.to(device)
            final_side = final_side.to(device)

            pred = model(features, target_n)
            loss = criterion(pred, final_side)
            mae = (pred - final_side).abs().mean()

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to training data JSONL')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-n', type=int, default=50)
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'attention'])
    parser.add_argument('--output', type=str, default='value_model.pt')
    parser.add_argument('--val-split', type=float, default=0.1)
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    dataset = PackingDataset(args.data, max_n=args.max_n)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = create_model(args.model_type, max_n=args.max_n)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'max_n': args.max_n,
                'model_type': args.model_type,
            }, args.output)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
