# AdaDepthCLIP: Adapter-based CLIP for Monocular Depth Estimation

This repository contains our experiments on **monocular depth estimation** using the **CLIP ViT-B/32 backbone**, **MLP adapters**, and a **composite loss function**.  
Our best configuration — *ViT-B/32 + Adapters + Composite Loss* — significantly improves prediction accuracy compared to baseline reproductions.

---

## Key Features
- **CLIP ViT-B/32 backbone** with partial fine-tuning  
- **MLP Adapters** inserted at layers 2, 5, 8, 11  
- **Composite Loss**: combination of SILog, Gradient, SSIM, and L1  
- Easy training on **NYU Depth V2** dataset  

---

## Results

### Performance Comparison

| Backbone + Adapters + Loss              | δ1 ↑  | δ2 ↑  | δ3 ↑  | AbsRel ↓ | Log10 ↓ | RMSE ↓ |
|-----------------------------------------|-------|-------|-------|----------|---------|--------|
| ResNet50 Backbone 🧊 + MLP adaptor       | 0.390 | 0.680 | 0.848 | 0.393    | 0.158   | 1.176  |
| ResNet50 Backbone 🔥 + MLP adaptor       | 0.344 | 0.674 | 0.858 | 0.327    | 0.162   | 1.190  |
| ViT-B/32 Backbone 🔥 + MLP adaptors      | 0.394 | 0.681 | 0.854 | 0.408    | 0.155   | 1.150  |
| ViT-B/32 Backbone 🔥 + Adapters + Loss   | 0.417 | 0.701 | 0.890 | 0.377    | 0.147   | 1.096  |
| **ViT-B/32 + Adapters + Composite Loss**| **0.503** | **0.791** | **0.925** | **0.310** | **0.121** | **0.843** |

**Our best model outperforms all baselines.**

---

## Installation

```bash
git clone https://github.com/ReyhanehAhani/AdaDepthCLIP.git
cd AdaDepthCLIP
conda create -n adadepth python=3.10
conda activate adadepth
pip install -r requirements.txt
```

Dependencies include:
- `torch`, `torchvision`, `open_clip_torch`, `tqdm`, `tensorboard`, `PIL`, `numpy`

---

## Training

### Train baseline (ViT-B/32 + Adapters + Composite Loss)
```bash
python train_vit_b32_learnable_loss.py   --train_data_root ./nyu_images/train   --val_data_root ./nyu_images/test   --train_data_path ./datasets/nyudepthv2_train_files_with_gt_dense.txt   --val_data_path ./datasets/nyudepthv2_test_files_with_gt_dense.txt   --model_path ./checkpoints/monoclip_vit_b32   --log_dir ./tensorboard_logs/monoclip_vit_b32   --epochs 50 --batch_size 8
```

### Train with Mixture-of-Adapters (MoA v2)
```bash
bash run_train_vit_moa_v2.sh
```

---

## Evaluation
After training, evaluate the best model:
```bash
python eval.py --model_path ./checkpoints/monoclip_vit_b32/best_model.pth
```

Metrics include **AbsRel, Log10, RMSE, δ1, δ2, δ3**.

---

## Citation
If you use this repo, please cite it as:
```bibtex
@misc{adadepth2025,
  title   = {AdaDepthCLIP: Adapter-based CLIP for Monocular Depth Estimation},
  author  = {Reyhaneh Ahani},
  year    = {2025},
  url     = {https://github.com/ReyhanehAhani/AdaDepthCLIP}
}
```

---

## License
This project is released under the MIT License.
