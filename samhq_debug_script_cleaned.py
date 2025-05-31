# Copyright by HQ-SAM team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from per_segment_anything.build_sam import sam_model_registry
from per_segment_anything.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc

import csv
import time
import psutil
import pynvml
import json

# ---------------------------
# GPU Monitoring Functions
# ---------------------------
def init_gpu_monitor():
    """Initialize NVML for GPU monitoring."""
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        print("[DEBUG] NVML initialized for GPU monitoring.")
    else:
        print("[DEBUG] CUDA not available; skipping GPU monitor initialization.")

def get_gpu_stats():
    """Collect and return GPU statistics as a dictionary."""
    stats = {}
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[DEBUG] Found {device_count} GPUs.")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats[f"gpu_{i}_memory_used_MB"] = mem_info.used / 1024**2
        stats[f"gpu_{i}_memory_total_MB"] = mem_info.total / 1024**2
        stats[f"gpu_{i}_utilization_%"] = util_info.gpu
        stats[f"gpu_{i}_temperature_C"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"[DEBUG] GPU {i}: used {stats[f'gpu_{i}_memory_used_MB']:.2f} MB, "
              f"total {stats[f'gpu_{i}_memory_total_MB']:.2f} MB, "
              f"utilization {stats[f'gpu_{i}_utilization_%']}%, "
              f"temperature {stats[f'gpu_{i}_temperature_C']} C.")
    return stats

def get_system_stats():
    """Return system wide statistics (CPU, RAM, Disk) as a dictionary."""
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_MB": psutil.virtual_memory().used / 1024**2,
        "ram_total_MB": psutil.virtual_memory().total / 1024**2,
        "disk_used_GB": psutil.disk_usage("/").used / 1024**3,
        "disk_total_GB": psutil.disk_usage("/").total / 1024**3,
    }
    print(f"[DEBUG] System stats: {stats}")
    return stats

def log_epoch_stats(epoch, output_path, start_time, end_time):
    """
    Logs resource usage stats to a CSV file.
    Debug: prints out the combined statistics.
    """
    system_stats = get_system_stats()
    gpu_stats = get_gpu_stats() if torch.cuda.is_available() else {}
    duration = end_time - start_time
    stats = {
        "epoch": epoch,
        "duration_sec": duration,
        **system_stats,
        **gpu_stats,
    }
    print(f"[DEBUG] Epoch {epoch} stats to log: {stats}")
    csv_file = os.path.join(output_path, "epoch_stats.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            print(f"[DEBUG] CSV header written to {csv_file}")
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writerow(stats)
    print(f"[DEBUG] Logged resource stats for epoch {epoch} to {csv_file}")

# ---------------------------
# Neural Network Components
# ---------------------------
class LayerNorm2d(nn.Module):
    """Custom 2D layer normalization."""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        print(f"[DEBUG] LayerNorm2d: output shape {x.shape}")
        return x

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        print(f"[DEBUG] MLP: output shape {x.shape}")
        return x

class MaskDecoderHQ(MaskDecoder):
    """HQ Mask Decoder extending the SAM MaskDecoder."""
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                         transformer=TwoWayTransformer(
                             depth=2,
                             embedding_dim=256,
                             mlp_dim=2048,
                             num_heads=8,
                         ),
                         num_multimask_outputs=3,
                         activation=nn.GELU,
                         iou_head_depth=3,
                         iou_head_hidden_dim=256)
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        checkpoint_dict = {"vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           "vit_h": "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"[DEBUG] HQ Decoder initialized using checkpoint: {checkpoint_path}")
        for n, p in self.named_parameters():
            p.requires_grad = True

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )
        
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2)
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool, hq_token_only: bool, interm_embeddings: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("[DEBUG] Forward pass in MaskDecoderHQ started.")
        # Permute early ViT features for proper channel alignment.
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        print(f"[DEBUG] vit_features reshaped to {vit_features.shape}")
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        print(f"[DEBUG] hq_features shape: {hq_features.shape}")

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            print(f"[DEBUG] Processing batch index {i_batch}")
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)
        print(f"[DEBUG] Combined masks shape: {masks.shape}")

        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
            print("[DEBUG] Multimask output selected.")
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]
            print("[DEBUG] Single mask output selected.")

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
        print("[DEBUG] HQ mask extracted.")
        
        return masks_hq if hq_token_only else (masks_sam, masks_hq)

    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                      hq_feature: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("[DEBUG] predict_masks() called.")
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        print(f"[DEBUG] Tokens shape after concat: {tokens.shape}")

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        print(f"[DEBUG] src shape before transformer: {src.shape}")

        hs, src = self.transformer(src, pos_src, tokens)
        print(f"[DEBUG] Transformer output hs shape: {hs.shape}")
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        print(f"[DEBUG] mask_tokens_out shape: {mask_tokens_out.shape}")

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        print(f"[DEBUG] upscaled embeddings shapes: SAM {upscaled_embedding_sam.shape}, Ours {upscaled_embedding_ours.shape}")
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
            print(f"[DEBUG] Processed hypernetwork token {i}")
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)
        print(f"[DEBUG] Predicted masks shape: {masks.shape}")
        
        iou_pred = self.iou_prediction_head(iou_token_out)
        print("[DEBUG] iou_prediction head output computed.")

        return masks, iou_pred

# ---------------------------
# Saving Functions
# ---------------------------
def save_encoder_and_decoder(encoder, decoder, output_path, epoch):
    """
    Saves encoder and decoder into separate directories.
    Debug messages added to confirm file paths.
    """
    encoder_dir = os.path.join(output_path, "encoderepoch")
    decoder_dir = os.path.join(output_path, "decoderepoch")
    
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
        print(f"[DEBUG] Created directory for encoder: {encoder_dir}")
    if not os.path.exists(decoder_dir):
        os.makedirs(decoder_dir)
        print(f"[DEBUG] Created directory for decoder: {decoder_dir}")
    
    encoder_checkpoint_path = os.path.join(encoder_dir, f"encoder_epoch_{epoch}.pth")
    torch.save(encoder.state_dict(), encoder_checkpoint_path)
    print(f"[DEBUG] Encoder saved at {encoder_checkpoint_path}")
    
    decoder_checkpoint_path = os.path.join(decoder_dir, f"decoder_epoch_{epoch}.pth")
    torch.save(decoder.state_dict(), decoder_checkpoint_path)
    print(f"[DEBUG] Decoder saved at {decoder_checkpoint_path}")

# ---------------------------
# Visualization Functions
# ---------------------------
def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        print("[DEBUG] No masks to annotate.")
        return
    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        out_file = filename + '_' + str(i) + '.png'
        plt.savefig(out_file, bbox_inches='tight', pad_inches=-0.1)
        print(f"[DEBUG] Saved annotation visualization to {out_file}")
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    print("[DEBUG] Mask visualized on axis.")

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    print("[DEBUG] Points displayed on image.")

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    print(f"[DEBUG] Box drawn at ({x0}, {y0}, {w}, {h}).")

# ---------------------------
# Argument Parsing
# ---------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--model-type", type=str, default="vit_l", help="Model type: one of ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to SAM checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=5, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='Rank for distributed processes')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str, help="Path to restore model for evaluation")
    return parser.parse_args()

# ---------------------------
# Main Pipeline
# ---------------------------
def main(net, train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)
    print(f"[DEBUG] Distributed mode initialized: world size {args.world_size}, rank {args.rank}, local_rank {args.local_rank}")
    print(f"[DEBUG] Arguments: {args}")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[DEBUG] Random seed set to: {seed}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    print(f"[DEBUG] SAM model of type '{args.model_type}' initialized.")

    if not args.eval:
        print("[DEBUG] Creating training dataloader...")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(
            train_im_gt_list,
            my_transforms=[RandomHFlip(), LargeScaleJitter()],
            batch_size=args.batch_size_train,
            training=True
        )
        print(f"[DEBUG] {len(train_dataloaders)} training dataloaders created.")
        
    print("[DEBUG] Creating validation dataloader...")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False
    )
    print(f"[DEBUG] {len(valid_dataloaders)} validation dataloaders created.")
    
    if torch.cuda.is_available():
        sam.cuda()
        net.cuda()
        print("[DEBUG] Moved models to CUDA.")
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)
    sam_without_ddp = sam.module

    if not args.eval:
        print("[DEBUG] Initializing optimizer for training...")
        optimizer = optim.Adam(sam_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        if args.restore_model:
            print(f"[DEBUG] Restoring model from: {args.restore_model}")
            if torch.cuda.is_available():
                sam_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                sam_without_ddp.load_state_dict(torch.load(args.restore_model, map_location="cpu"))
        evaluate(args, net, sam, valid_dataloaders, args.visualize)

def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    print("[DEBUG] Starting training process...")
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        print(f"[DEBUG] Output directory created (if not existing): {args.output}")
    training_metrics = {"loss": [], "val_iou": [], "val_boundary_iou": []}

    init_gpu_monitor()
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)
    print(f"[DEBUG] Training for {epoch_num} epochs with {train_num} batches per epoch.")

    lr_scheduler.step()
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.train()
    _ = sam.cuda()
    net.train()
    _ = net.cuda()
    print("[DEBUG] SAM and HQ models set to training mode.")
    
    for name, param in sam.image_encoder.named_parameters():
        param.requires_grad = True
    for name, param in sam.mask_decoder.named_parameters():
        param.requires_grad = True
    for name, param in net.named_parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0
    )
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)
    print("[DEBUG] Optimizer initialized and wrapped with DDP.")

    for epoch in range(epoch_start, epoch_num):
        print(f"[DEBUG] Starting epoch {epoch} with learning rate {optimizer.param_groups[0]['lr']}")
        start_time = time.time()
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for data in metric_logger.log_every(train_dataloaders, 1000):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            print(f"[DEBUG] Batch input shape: {inputs.shape}, label shape: {labels.shape}")
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
            except Exception:
                input_keys = ['box', 'noise_mask']
                print("[DEBUG] Not enough points for sampling; using 'box' and 'noise_mask' prompts only.")
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)
            
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = {}
                input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
            print(f"[DEBUG] Created batched input with {len(batched_input)} samples.")

            batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            print(f"[DEBUG] SAM returned output with keys: {list(batched_output[0].keys())}")
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(batch_len)], dim=0)
            image_pe = [batched_output[i]['image_pe'] for i in range(batch_len)]
            sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(batch_len)]
            dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(batch_len)]
            print(f"[DEBUG] Combined encoder embeddings shape: {encoder_embedding.shape}")
            
            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )
            print(f"[DEBUG] HQ masks shape: {masks_hq.shape}")
            
            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            print(f"[DEBUG] Loss computed: {loss.item()}")
            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[DEBUG] Optimizer step completed for current batch.")

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            epoch_loss += loss_value
        
        print(f"[DEBUG] Finished epoch {epoch} with average loss: {epoch_loss / train_num}")
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        
        val_iou = train_stats.get("val_iou_0", 0.0)
        val_boundary_iou = train_stats.get("val_boundary_iou_0", 0.0)
        training_metrics["loss"].append(epoch_loss / train_num)
        training_metrics["val_iou"].append(val_iou)
        training_metrics["val_boundary_iou"].append(val_boundary_iou)

        with open(os.path.join(args.output, "training_metrics.json"), "w") as f:
            json.dump(training_metrics, f, indent=4)
            print(f"[DEBUG] Training metrics saved to {args.output}/training_metrics.json")
        
        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        end_time = time.time()
        log_epoch_stats(epoch, args.output, start_time, end_time)
        
        sam.train()
        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_" + str(epoch) + ".pth"
            print(f"[DEBUG] Saving model checkpoint at {args.output + model_name}")
            misc.save_on_master(net.state_dict(), args.output + model_name)
            save_encoder_and_decoder(encoder=sam.module.image_encoder, decoder=sam.module.mask_decoder, output_path=args.output, epoch=epoch)
    print("[DEBUG] Training completed. Maximum epoch reached.")
    
    if misc.is_main_process():
        trained_sam_ckpt = sam.module.state_dict()
        hq_decoder_ckpt = torch.load(args.output + model_name)
        for key in hq_decoder_ckpt.keys():
            hq_key = 'mask_decoder.' + key
            trained_sam_ckpt[hq_key] = hq_decoder_ckpt[key]
        model_name = "/sam_hq_epoch_" + str(epoch) + ".pth"
        torch.save(trained_sam_ckpt, args.output + model_name)
        print(f"[DEBUG] Combined SAM+HQ model saved at {args.output + model_name}")

def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(len(preds)):
        iou += misc.mask_iou(postprocess_preds[i], target[i])
    result = iou / len(preds)
    print(f"[DEBUG] Computed IoU: {result}")
    return result

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(len(preds)):
        iou += misc.boundary_iou(target[i], postprocess_preds[i])
    result = iou / len(preds)
    print(f"[DEBUG] Computed Boundary IoU: {result}")
    return result

def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("[DEBUG] Starting evaluation...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print(f"[DEBUG] Evaluating on dataloader {k} with {len(valid_dataloader)} batches.")
        
        for data_val in metric_logger.log_every(valid_dataloader, 1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = (
                data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            )
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            labels_box = misc.masks_to_boxes(labels_val[:, 0, :, :])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = {}
                input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
            print(f"[DEBUG] Prepared batched input for evaluation with {len(batched_input)} samples.")
            
            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(batch_len)], dim=0)
            image_pe = [batched_output[i]['image_pe'] for i in range(batch_len)]
            sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(batch_len)]
            dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(batch_len)]
            print(f"[DEBUG] Evaluation embeddings shape: {encoder_embedding.shape}")
            
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )
            print(f"[DEBUG] Evaluation predicted masks shapes: SAM {masks_sam.shape}, HQ {masks_hq.shape}")
            iou = compute_iou(masks_hq, labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq, labels_ori)

            if visualize:
                print("[DEBUG] Visualization mode on.")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    save_base = os.path.join(args.output, f"{k}_{base}")
                    imgs_ii = imgs[ii].astype(np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base, imgs_ii, show_iou, show_boundary_iou)
            loss_dict = {"val_iou_" + str(k): iou, "val_boundary_iou_" + str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
        
        print("[DEBUG] Evaluation metrics for dataloader", k)
        metric_logger.synchronize_between_processes()
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        print(f"[DEBUG] Dataloader {k} averaged stats: {resstat}")
        test_stats.update(resstat)
    return test_stats

if __name__ == "__main__":
    # --------------------- Configure Train and Valid Datasets ---------------------
    dataset_dis = {"name": "DIS5K-TR",
                   "im_dir": "./data/DIS5K/DIS-TR/im",
                   "gt_dir": "./data/DIS5K/DIS-TR/gt",
                   "im_ext": ".jpg",
                   "gt_ext": ".png"}

    dataset_thin = {"name": "ThinObject5k-TR",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    dataset_fss = {"name": "FSS",
                   "im_dir": "./data/cascade_psp/fss_all",
                   "gt_dir": "./data/cascade_psp/fss_all",
                   "im_ext": ".jpg",
                   "gt_ext": ".png"}

    dataset_duts = {"name": "DUTS-TR",
                   "im_dir": "./data/cascade_psp/DUTS-TR",
                   "gt_dir": "./data/cascade_psp/DUTS-TR",
                   "im_ext": ".jpg",
                   "gt_ext": ".png"}

    dataset_duts_te = {"name": "DUTS-TE",
                       "im_dir": "./data/cascade_psp/DUTS-TE",
                       "gt_dir": "./data/cascade_psp/DUTS-TE",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}

    dataset_ecssd = {"name": "ECSSD",
                     "im_dir": "./data/cascade_psp/ecssd",
                     "gt_dir": "./data/cascade_psp/ecssd",
                     "im_ext": ".jpg",
                     "gt_ext": ".png"}

    dataset_msra = {"name": "MSRA10K",
                    "im_dir": "./data/cascade_psp/MSRA_10K",
                    "gt_dir": "./data/cascade_psp/MSRA_10K",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    dataset_coift_val = {"name": "COIFT",
                         "im_dir": "./data/thin_object_detection/COIFT/images",
                         "gt_dir": "./data/thin_object_detection/COIFT/masks",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                         "im_dir": "./data/thin_object_detection/HRSOD/images",
                         "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                         "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                         "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                       "im_dir": "./data/DIS5K/DIS-VD/im",
                       "gt_dir": "./data/DIS5K/DIS-VD/gt",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}

    dataset_renishaw_train = {"name": "augmented_data_with_all_renishaw_data",
                              "im_dir": "./augmented_data_with_all_renishaw_data/train/images",
                              "gt_dir": "./augmented_data_with_all_renishaw_data/train/masks",
                              "im_ext": ".jpg",
                              "gt_ext": ".png"}

    dataset_renishaw_test = {"name": "augmented_data_with_all_renishaw_data",
                             "im_dir": "./augmented_data_with_all_renishaw_data/test/images",
                             "gt_dir": "./augmented_data_with_all_renishaw_data/test/masks",
                             "im_ext": ".jpg",
                             "gt_ext": ".png"}
    
    # Uncomment below to use multiple datasets for training and validation:
    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val]
    train_datasets = [dataset_dis]
    valid_datasets = [dataset_dis_val]

    args = get_args_parser()
    net = MaskDecoderHQ(args.model_type)
    print("[DEBUG] Starting main pipeline.")
    main(net, train_datasets, valid_datasets, args)
