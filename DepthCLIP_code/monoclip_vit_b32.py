import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math

depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1

class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, in_features)

    def forward(self, x):
        return x + self.up_project(self.activation(self.down_project(x)))


class MonoCLIP_ViT_B32(nn.Module):
    def __init__(self, args, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        super().__init__()
        self.args = args
        self.bins = len(bin_list)

        print(f"📦 Loading OpenCLIP model: {model_name} | Pretrained: {pretrained}")
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=args.device
        )
        self.visual = clip_model.visual
        self.text_encoder = clip_model.encode_text

        for param in self.parameters():
            param.requires_grad = False

        num_blocks_to_unfreeze = 4
        total_blocks = len(self.visual.transformer.resblocks)
        for i in range(total_blocks - num_blocks_to_unfreeze, total_blocks):
            print(f"🔓 Unfreezing ViT block {i}")
            for param in self.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True

        feature_dim = self.visual.transformer.width  # usually 768
        self.adapter_indices = [2, 5, 8, 11]
        self.adapters = nn.ModuleList([Adapter(feature_dim) for _ in self.adapter_indices])
        print(f"✅ Added {len(self.adapters)} adapters @ layers: {self.adapter_indices} | dim={feature_dim}")
        for param in self.adapters.parameters():
            param.requires_grad = True

        self.text_features = self.get_text_features(self.text_encoder, model_name, args.device)  # shape: (D=512, 7)
        print(f"✅ Text features shape: {self.text_features.shape}")

        # Project image features from 768 → 512
        self.project_to_text_dim = nn.Linear(feature_dim, self.text_features.shape[0])  # (768 → 512)

        self.register_buffer('bin_tensor', torch.tensor(bin_list, device=args.device).float())

    def get_text_features(self, encode_text_fn, model_name, device):
        with torch.no_grad():
            tokenizer = open_clip.get_tokenizer(model_name)
            text_weights = []
            for depth in depth_classes:
                texts = [template.format(obj, depth) for template in depth_templates for obj in obj_classes]
                tokens = tokenizer(texts).to(device)
                embeds = encode_text_fn(tokens)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                mean_embed = embeds.mean(dim=0)
                mean_embed = mean_embed / mean_embed.norm()
                text_weights.append(mean_embed)
            return torch.stack(text_weights, dim=1).float()  # shape: (512, 7)

    def forward(self, x):
        B = x.shape[0]
        print(f"📥 Input image shape: {x.shape}")

        x = self.visual.conv1(x)  # (B, C, H/patch, W/patch)
        print(f"🧩 After conv1: {x.shape}")
        
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)
        print(f"🧩 After reshape+permute: {x.shape}")

        x = torch.cat([
            self.visual.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (Seq, B, D)

        adapter_counter = 0
        all_patch_features = []
        for i, resblock in enumerate(self.visual.transformer.resblocks):
            x = resblock(x)
            if i in self.adapter_indices:
                print(f"🔧 Adapter at layer {i}")
                x_permuted = x.permute(1, 0, 2)
                adapted_x = self.adapters[adapter_counter](x_permuted)
                x = adapted_x.permute(1, 0, 2)
                adapter_counter += 1
                all_patch_features.append(x.permute(1, 0, 2)[:, 1:, :])  # Remove class token

        image_feats = torch.stack(all_patch_features, dim=0).mean(dim=0)  # (B, N, D)
        print(f"📐 Raw image feats shape (pre-proj): {image_feats.shape}")

        image_feats = self.project_to_text_dim(image_feats)  # (B, N, 512)
        print(f"📐 Projected image feats shape: {image_feats.shape}")
        
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        logits = 100.0 * (image_feats @ self.text_features)  # (B, N, 7)
        print(f"🎯 Logits shape: {logits.shape}")

        probs = F.softmax(logits / temperature, dim=-1)  # (B, N, 7)
        weighted_depth = (probs * self.bin_tensor).sum(dim=-1)  # (B, N)

        num_patches = weighted_depth.shape[1]
        side_len = int(math.sqrt(num_patches))
        depth_map = weighted_depth.view(B, 1, side_len, side_len)
        final_depth = F.interpolate(depth_map, size=(self.args.height, self.args.width), mode='bilinear', align_corners=False)

        print(f"🖼️ Final depth shape: {final_depth.shape}")
        return final_depth
