"""
DepthWizard - Kaggle GPU Fine-tuning Script for Person A.
Fine-tunes Depth Anything V2 Base on GAMUS (5,000 images) with a custom Tall-Building Weighted Loss.
Fixes the 8.8m underestimation problem on tall commercial structures.
"""

import os
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    pass


class TallBuildingWeightedLoss(nn.Module):
    """
    Specialized loss function combining Scale-Invariant Logarithmic (SILog) loss
    with an exponential height-weight penalty to prioritize tall structures.
    """
    def __init__(self, alpha: float = 2.5, lambda_tall: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_tall = lambda_tall

    def forward(self, pred: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor"):
        # Valid pixel mask
        valid = (mask > 0) & (target > 0.1) & (pred > 0.1)
        if not torch.any(valid):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p = pred[valid]
        t = target[valid]

        # 1. Scale-Invariant Logarithmic Loss (SILog)
        d = torch.log(p) - torch.log(t)
        silog = torch.sqrt(torch.mean(d ** 2) - 0.5 * (torch.mean(d) ** 2))

        # 2. Tall-Building Weighted L1 Loss
        # Objects above 15m get weighted significantly higher
        height_weights = 1.0 + self.alpha * torch.clamp(t / 40.0, 0.0, 2.0)
        tall_l1 = torch.mean(height_weights * torch.abs(p - t))

        # Combined loss
        total_loss = silog + self.lambda_tall * tall_l1
        return total_loss


def train():
    parser = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 on GAMUS")
    parser.add_argument("--model_name", type=str, default="depth-anything/Depth-Anything-V2-Base-hf")
    parser.add_argument("--dataset_path", type=str, default="./gamus-dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print(" DepthWizard - GAMUS Height Fine-Tuning Pipeline (Person A)")
    print(f" Model: {args.model_name}")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load Backbone
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name)
    model.to(device)

    # Initialize Dataset
    from dataset_gamus import GAMUSDataset
    train_dataset = GAMUSDataset(split="train", dataset_name_or_path=args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = TallBuildingWeightedLoss(alpha=2.5, lambda_tall=1.5)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device)
            target_h = batch["height"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=rgb)
            pred = outputs.predicted_depth.unsqueeze(1)

            # Resize pred if necessary
            if pred.shape[-2:] != target_h.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred, size=target_h.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(pred, target_h, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"depthwizard_gamus_epoch{epoch+1}.pt"))

    print("[DepthWizard Training] Training Complete. Checkpoints saved to:", args.output_dir)


if __name__ == "__main__":
    train()
