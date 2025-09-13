import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from monoclip_vit_b32 import MonoCLIP_ViT_B32

# ---------- Loss Functions ----------
class SILogLoss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        if mask.any():
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean()) ** 2) * 10.0
        return torch.tensor(0.0, device=depth_est.device)

def gradient_loss(pred):
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    return (dx.mean() + dy.mean())

def ssim_loss(pred, target):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()

def l1_loss(pred, target):
    return torch.abs(pred - target).mean()

def compute_total_loss(pred, target, mask, weights):
    loss_silog = SILogLoss()(pred, target, mask)
    loss_grad = gradient_loss(pred)
    loss_ssim = ssim_loss(pred, target)
    loss_l1 = l1_loss(pred, target)
    return (
        weights["silog"] * loss_silog +
        weights["grad"] * loss_grad +
        weights["ssim"] * loss_ssim +
        weights["l1"] * loss_l1
    )

# ---------- Dataset ----------
class NYUDepthV2Dataset(Dataset):
    def __init__(self, txt_file, data_root_dir, preprocess_fn, height, width, depth_scale=1000.0):
        self.data_root_dir = data_root_dir
        self.preprocess = preprocess_fn
        self.height = height
        self.width = width
        self.depth_scale = depth_scale
        with open(txt_file, 'r') as f:
            self.files = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_rel, depth_rel = self.files[idx]
        img_path = os.path.join(self.data_root_dir, img_rel)
        depth_path = os.path.join(self.data_root_dir, depth_rel)
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        depth = np.array(Image.open(depth_path), dtype=np.float32) / self.depth_scale
        depth = ToTensor()(depth)
        depth_resized = F.interpolate(depth.unsqueeze(0), size=(self.height, self.width), mode='nearest').squeeze(0)
        return {'image': image, 'depth': depth_resized}

# ---------- Train & Validate ----------
def train_one_epoch(model, dataloader, optimizer, device, epoch_num, scaler, weights):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch_num}")
    for batch in progress_bar:
        images, gt_depth = batch['image'].to(device), batch['depth'].to(device)
        mask = gt_depth > 1e-8
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            predicted_depth = model(images)
            loss = compute_total_loss(predicted_depth, gt_depth, mask, weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isnan(loss) and loss.item() > 0:
            total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

def validate(model, dataloader, device, weights):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for batch in progress_bar:
            images, gt_depth = batch['image'].to(device), batch['depth'].to(device)
            mask = gt_depth > 1e-8
            with autocast(device_type=device.type):
                predicted_depth = model(images)
                loss = compute_total_loss(predicted_depth, gt_depth, mask, weights)
            if not torch.isnan(loss) and loss.item() > 0:
                total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

# ---------- Main ----------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    model_args = argparse.Namespace(device=device, height=args.height, width=args.width)
    model = MonoCLIP_ViT_B32(model_args).to(device)

    train_dataset = NYUDepthV2Dataset(args.train_data_path, args.train_data_root, model.preprocess, args.height, args.width)
    val_dataset = NYUDepthV2Dataset(args.val_data_path, args.val_data_root, model.preprocess, args.height, args.width)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    backbone_params = [p for n, p in model.named_parameters() if 'visual' in n and p.requires_grad]
    adapter_params = [p for n, p in model.named_parameters() if 'adapters' in n]

    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': args.lr_adapters},
        {'params': backbone_params, 'lr': args.lr_backbone}
    ], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    scaler = GradScaler()
    best_val_loss, epochs_no_improve = float('inf'), 0

    loss_weights = {
        "silog": args.w_silog,
        "grad": args.w_grad,
        "ssim": args.w_ssim,
        "l1": args.w_l1
    }

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1, scaler, loss_weights)
        val_loss = validate(model, val_loader, device, loss_weights)

        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.model_path, "best_model.pth"))
            print(f"✅ New best model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs.")
            break

        scheduler.step(val_loss)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_root', type=str, required=True)
    parser.add_argument('--val_data_root', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='./checkpoints/monoclip_vit_b32')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_logs/monoclip_vit_b32')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_adapters', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--depth_scale', type=float, default=1000.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--w_silog', type=float, default=1.0)
    parser.add_argument('--w_grad', type=float, default=0.5)
    parser.add_argument('--w_ssim', type=float, default=0.2)
    parser.add_argument('--w_l1', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
