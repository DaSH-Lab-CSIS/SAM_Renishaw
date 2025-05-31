import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import SamPredictor
from per_segment_anything import sam_model_registry


def load_sam_checkpoint(sam, checkpoint_path):
    print(f"[DEBUG] Loading SAM checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Filter keys that are related to relative positional embeddings:
    keys_to_remove = [k for k in state_dict.keys() if "attn.rel_pos" in k]
    for key in keys_to_remove:
        print(f"[INFO] Removing key {key} from checkpoint state_dict")
        del state_dict[key]
    sam.load_state_dict(state_dict, strict=False)
    print("[DEBUG] Checkpoint loaded with filtered keys.")
    return sam


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam_f')
    parser.add_argument('--ckpt', type=str, default='', 
                        help="Path to the SAM checkpoint to load (passed to SAM).")
    parser.add_argument('--sam_type', type=str, default='vit_b')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--log_epoch', type=int, default=200)
    parser.add_argument('--ref_idx', type=str, default='00')
    args = parser.parse_args()
    print("[DEBUG] Command line arguments parsed:", args)
    return args


def main():
    args = get_arguments()
    print("[DEBUG] Args:", args)

    images_path = os.path.join(args.data, 'Images')
    masks_path = os.path.join(args.data, 'Annotations')
    output_path = os.path.join('./outputs', args.outdir)
    print(f"[DEBUG] Images path: {images_path}")
    print(f"[DEBUG] Masks path: {masks_path}")
    print(f"[DEBUG] Output path: {output_path}")

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
        print("[DEBUG] Created directory ./outputs")
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            print(f"[DEBUG] Processing object: {obj_name}")
            persam_f(args, obj_name, images_path, masks_path, output_path)


def persam_f(args, obj_name, images_path, masks_path, output_path):
    print(f"\n------------> Segment {obj_name}")
    
    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)
    print(f"[DEBUG] ref_image_path: {ref_image_path}")
    print(f"[DEBUG] ref_mask_path: {ref_mask_path}")
    
    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"[DEBUG] Output directory for {obj_name} ensured at: {output_path}")

    # Load images and masks
    print("[DEBUG] Loading reference image and mask...")
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()
    print("[DEBUG] Reference image, mask, and ground truth mask loaded.")

    print("======> Load SAM")
    if args.sam_type == 'vit_b':
        print("[DEBUG] Using SAM type 'vit_b'")
        sam_type = 'vit_b'
        # Use the checkpoint provided as an argument
        sam_ckpt = args.ckpt  
        if not sam_ckpt:
            raise ValueError("[ERROR] For vit_b, please provide a valid --ckpt argument.")
        device = "cuda"
        sam = sam_model_registry[sam_type](checkpoint=None).to(device)
        sam = load_sam_checkpoint(sam, sam_ckpt)
        sam = sam.to(device)
    elif args.sam_type == 'vit_t':
        print("[DEBUG] Using SAM type 'vit_t'")
        sam_type = 'vit_t'
        sam_ckpt = args.ckpt  
        if not sam_ckpt:
            raise ValueError("[ERROR] For vit_t, please provide a valid --ckpt argument.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    else:
        raise ValueError(f"[ERROR] Unsupported SAM type: {args.sam_type}")

    # Freeze SAM parameters
    for name, param in sam.named_parameters():
        param.requires_grad = False
    print("[DEBUG] SAM parameters frozen.")

    predictor = SamPredictor(sam)
    print("[DEBUG] SamPredictor initialized.")

    print("======> Obtain Self Location Prior")
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    print("[DEBUG] set_image() called on predictor.")
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    print(f"[DEBUG] Reference features shape: {ref_feat.shape}")

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    print("[DEBUG] Resized reference mask obtained.")

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]
    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0]
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
    print("[DEBUG] Target feature computed from reference features.")

    # Cosine similarity
    h, w, C = ref_feat.shape
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
    sim = target_feat @ ref_feat
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()
    print("[DEBUG] Cosine similarity map computed.")

    # Positive location prior
    topk_xy, topk_label = point_selection(sim, topk=1)
    print(f"[DEBUG] Top-1 point selected: {topk_xy}, label: {topk_label}")

    print("======> Start Training")
    # Learnable mask weights
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()
    print("[DEBUG] Mask_Weights initialized and set to training mode.")
    
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)
    print("[DEBUG] Optimizer and scheduler set up for mask weights.")

    for train_idx in range(args.train_epoch):
        # Run the decoder
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)
        print(f"[DEBUG] Training iteration {train_idx}: decoder output obtained.")

        # Weighted sum three-scale masks
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)
        print("[DEBUG] Weighted sum of logits computed.")

        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % args.log_epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[DEBUG] Train Epoch: {train_idx} / {args.train_epoch}")
            print(f"[DEBUG] LR: {current_lr:.6f}, Dice_Loss: {dice_loss.item():.4f}, Focal_Loss: {focal_loss.item():.4f}")

    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print("======> Mask weights:\n", weights_np)

    print("======> Start Testing")
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
        test_idx_str = '%02d' % test_idx
        test_image_path = os.path.join(test_images_path, test_idx_str + '.jpg')
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"[DEBUG] Testing on image: {test_image_path}")

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()
        print("[DEBUG] Similarity map computed for test image.")

        # Positive location prior
        topk_xy, topk_label = point_selection(sim, topk=1)
        print(f"[DEBUG] Test image top point: {topk_xy}, label: {topk_label}")

        # First-step prediction
        masks, scores, logits, logits_high = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)
        # Weighted sum three-scale masks for testing
        logits_high = logits_high * weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * weights_np[..., None]
        logit = logits.sum(0)

        # Cascaded Post-refinement-1
        y, x = np.nonzero(mask)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)
        print(f"[DEBUG] Post-refinement-1: best index {best_idx}")

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)
        print(f"[DEBUG] Post-refinement-2: best index {best_idx}")
        
        # Save masks visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx_str}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')
        print(f"[DEBUG] Visualization saved to {vis_mask_output_path}")
        plt.close()

        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx_str + '.png')
        cv2.imwrite(mask_output_path, mask_colors)
        print(f"[DEBUG] Final mask saved to {mask_output_path}")


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)
        print("[DEBUG] Mask_Weights parameters initialized.")


def point_selection(mask_sim, topk=1):
    # Top-k point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    print(f"[DEBUG] Point selection: {topk} points selected.")
    return topk_xy, topk_label


def calculate_dice_loss(inputs, targets, num_masks=1):
    """
    Compute the DICE loss.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha=0.25, gamma=2):
    """
    Compute the sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


if __name__ == '__main__':
    main()
