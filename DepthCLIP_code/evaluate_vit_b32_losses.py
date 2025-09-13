import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
from tqdm import tqdm
import numpy as np

# 🟠 مدل خود را اینجا وارد کنید
from monoclip_vit_b32 import MonoCLIP_ViT_B32

# ---------------------------------------------------
# 🟠 کلاس برای ذخیره و میانگین‌گیری متریک‌ها
# ---------------------------------------------------
class AverageMeter(object):
    def __init__(self, i=1, precision=4):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.i = i
        self.precision = precision

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.avg:.{self.precision}f}'

# ---------------------------------------------------
# 🟠 تابع محاسبه تمام متریک‌های ارزیابی عمق
# ---------------------------------------------------
def compute_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    
    log10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred)))

    return {
        'abs_rel': abs_rel.item(), 'rmse': rmse.item(), 'log10': log10.item(),
        'a1': a1.item(), 'a2': a2.item(), 'a3': a3.item()
    }


# ---------------------------------------------------
# 🟠 دیتاست اصلاح‌شده برای خواندن صحیح عمق
# ---------------------------------------------------
class DepthDataset(Dataset):
    def __init__(self, root_dir, path_file, image_transform=None, depth_transform=None, depth_scale=1000.0):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.depth_scale = depth_scale
        self.samples = []

        with open(path_file, 'r') as f:
            for line in f.readlines():
                rgb, depth = line.strip().split()
                self.samples.append((os.path.join(root_dir, rgb), os.path.join(root_dir, depth)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]
        image = Image.open(rgb_path).convert("RGB")
        
        # خواندن عمق به صورت صحیح (uint16) و تبدیل به متر
        depth = np.array(Image.open(depth_path), dtype=np.float32) / self.depth_scale
        depth = ToTensor()(depth)

        if self.image_transform:
            image = self.image_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        return image, depth

# ---------------------------------------------------
# 🟠 تابع ارزیابی بازنویسی‌شده
# ---------------------------------------------------
def evaluate(model, dataloader, device, cap=10.0, save_preds=False):
    model.eval()
    
    error_names = ['abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    errors = {name: AverageMeter() for name in error_names}
    
    pred_list = []
    gt_list = []

    with torch.no_grad():
        for images, gt_depths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            gt_depths = gt_depths.to(device)

            # ایجاد ماسک برای مقادیر نامعتبر (عمق صفر)
            mask = (gt_depths > 1e-3) & (gt_depths < cap)

            # پیش‌بینی مدل
            preds = model(images)
            
            # اطمینان از هم‌اندازه بودن پیش‌بینی و واقعیت
            preds = nn.functional.interpolate(
                preds, size=gt_depths.shape[-2:], mode='bilinear', align_corners=False
            )

            # اعمال کلاهک (cap) روی پیش‌بینی
            preds[preds < 1e-3] = 1e-3
            preds[preds > cap] = cap
            
            # برش Garg برای ارزیابی استاندارد
            h, w = gt_depths.shape[-2:]
            crop = (int(h * 0.40810811), int(h * 0.99189189),
                    int(w * 0.03594771), int(w * 0.96405229))
            
            gt_cropped = gt_depths[..., crop[0]:crop[1], crop[2]:crop[3]]
            pred_cropped = preds[..., crop[0]:crop[1], crop[2]:crop[3]]
            mask_cropped = mask[..., crop[0]:crop[1], crop[2]:crop[3]]

            # مقیاس‌بندی پیش‌بینی با میانه (یک روش استاندارد)
            median_scale = torch.median(gt_cropped[mask_cropped]) / torch.median(pred_cropped[mask_cropped])
            pred_cropped *= median_scale
            
            # محاسبه خطاها فقط روی پیکسل‌های معتبر
            gt_valid = gt_cropped[mask_cropped]
            pred_valid = pred_cropped[mask_cropped]

            batch_errors = compute_errors(gt_valid, pred_valid)
            for name in error_names:
                errors[name].update(batch_errors[name])

            if save_preds:
                pred_list.append(preds.cpu().numpy())
                gt_list.append(gt_depths.cpu().numpy())

    # چاپ نتایج نهایی
    print("\n" + "="*30)
    print("📊 Final Evaluation Metrics (NYU Depth V2)")
    print("="*30)
    print(f"{'Metric':<10} | {'Value'}")
    print("-"*30)
    for name in error_names:
        print(f"{name:<10} | {errors[name].avg:.4f}")
    print("="*30)

    if save_preds:
        np.save("pred_depths.npy", np.concatenate(pred_list))
        np.save("gt_depths.npy", np.concatenate(gt_list))

# ---------------------------------------------------
# 🟠 تابع اصلی (main)
# ---------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # تبدیل‌های تصویر مشابه CLIP
    image_transform = Compose([
        Resize((args.height, args.width), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    val_dataset = DepthDataset(
        root_dir=args.val_data_root,
        path_file=args.val_data_path,
        image_transform=image_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = MonoCLIP_ViT_B32(args)
    # ❗️ توجه: مطمئن شوید که فایل مدل با معماری تعریف شده سازگار است
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    evaluate(model, val_loader, device, save_preds=args.save_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth Estimation Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--val_data_root', type=str, required=True, help='Root directory of the validation data.')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to the text file listing validation files.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--save_preds', action='store_true', help='If set, saves predictions and ground truth to .npy files.')
    
    # پارامترهای مورد نیاز مدل
    parser.add_argument('--height', type=int, default=224, help='Input image height for the model.')
    parser.add_argument('--width', type=int, default=224, help='Input image width for the model.')

    args = parser.parse_args()
    main(args)