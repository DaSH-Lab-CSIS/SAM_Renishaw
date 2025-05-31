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

from show import *  # Helper functions for visualization (e.g. show_mask, show_points)
from per_segment_anything import SamPredictor  # SAM predictor class for inference
from per_segment_anything import sam_model_registry  # Registry mapping model types to SAM modules


def load_sam_checkpoint(sam, checkpoint_path):
    """
    Loads and filters the SAM checkpoint.
    It removes keys related to relative positional embeddings and loads the filtered state dict.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Remove any keys containing 'attn.rel_pos'
    keys_to_remove = [k for k in state_dict.keys() if "attn.rel_pos" in k]
    for key in keys_to_remove:
        print(f"[INFO] Removing key {key} from checkpoint state_dict")
        del state_dict[key]
    sam.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded with filtered keys.")
    return sam


def get_arguments():
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data', 
                        help="Path to the input data directory containing Images/ and Annotations/")
    parser.add_argument('--outdir', type=str, default='persam_f',
                        help="Output subdirectory name for storing results")
    parser.add_argument('--ckpt', type=str, default='',
                        help="Path to the SAM checkpoint to load")
    parser.add_argument('--sam_type', type=str, default='vit_b',
                        help="Type of SAM model; e.g., 'vit_b' or 'vit_t'")
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help="Learning rate for training mask weights")
    parser.add_argument('--train_epoch', type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument('--log_epoch', type=int, default=200,
                        help="Logging frequency (in iterations)")
    parser.add_argument('--ref_idx', type=str, default='00',
                        help="Reference image index to use for each object")
    
    args = parser.parse_args()
    return args


def main():
    """
    Main function that gets arguments, sets up paths and processes each object in the dataset.
    """
    args = get_arguments()
    print("Args:", args)

    # Define paths to images, annotations and output directory
    images_path = os.path.join(args.data, 'Images')
    masks_path = os.path.join(args.data, 'Annotations')
    output_path = os.path.join('./outputs', args.outdir)

    # Create the outputs directory if it doesn't exist
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    
    # Process each object folder in the images path (ignoring system files)
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam_f(args, obj_name, images_path, masks_path, output_path)


def persam_f(args, obj_name, images_path, masks_path, output_path):
    """
    Process the segmentation for a single object.

    This function performs:
      - Path setup for the reference images and masks.
      - Loading of the reference image and mask.
      - SAM model loading (using the checkpoint path passed via '--ckpt').
      - Extraction of a self location prior based on cosine similarity.
      - Training of learnable mask weights.
      - Testing and post-refinement on the test images.
    """
    print("\n------------> Segment " + obj_name)
    
    # Prepare file paths for the reference image and mask of the current object
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    # Ensure there is a dedicated output directory for the current object
    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load reference image and convert from BGR to RGB
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    # Load reference mask and convert from BGR to RGB
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    # Convert the reference mask to a binary tensor, moving it to GPU
    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

    # -----------------------
    # Load and Prepare SAM
    # -----------------------
    print("======> Load SAM" )
    if args.sam_type == 'vit_b':
        # For 'vit_b', use the checkpoint passed as an argument
        sam_type, sam_ckpt = 'vit_b', args.ckpt
        if not sam_ckpt:
            raise ValueError("For vit_b, please provide a valid --ckpt argument.")
        device = "cuda"
        sam = sam_model_registry[sam_type](checkpoint=None).to(device)
        sam = load_sam_checkpoint(sam, sam_ckpt)
        sam = sam.to(device)
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', args.ckpt
        if not sam_ckpt:
            raise ValueError("For vit_t, please provide a valid --ckpt argument.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    else:
        raise ValueError(f"Unsupported SAM type: {args.sam_type}")

    # Freeze all parameters of the SAM model as they are not updated during training
    for name, param in sam.named_parameters():
        param.requires_grad = False
    # Initialize the SAM predictor which handles the prompt-based prediction
    predictor = SamPredictor(sam)
    

    # ------------------------------
    # Obtain Self Location Prior
    # ------------------------------
    print("======> Obtain Self Location Prior" )
    # Encode the reference image using the predictor; this returns encoded mask features.
    ref_mask = predictor.set_image(ref_image, ref_mask)
    # Extract features from the reference image and rearrange dimensions
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    # Resize the reference mask to match the feature map size
    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Extract target features only from areas where the mask is present.
    target_feat = ref_feat[ref_mask > 0]
    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0]
    # Combine maximum and mean features into a single target feature vector
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

    # Compute cosine similarity between target feature and reference features
    h, w, C = ref_feat.shape
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
    sim = target_feat @ ref_feat
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    # Postprocess the similarity map using SAM's built-in method
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()

    # Select the top-1 most confident location (point) as the positive location prior
    topk_xy, topk_label = point_selection(sim, topk=1)

    # -----------------------
    # Training Phase
    # -----------------------
    print('======> Start Training')
    # Initialize learnable mask weights (to weight multi-scale predictions)
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()
    
    # Set up the optimizer and scheduler for training the mask weights
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    for train_idx in range(args.train_epoch):
        # Run the SAM decoder to obtain multi-scale masks and scores
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)

        # Weight and sum the three-scale masks using the learnable mask weights
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)

        # Compute loss using a combination of DICE loss and sigmoid focal loss
        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % args.log_epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))

    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np)

    # -----------------------
    # Testing Phase
    # -----------------------
    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
        test_idx = '%02d' % test_idx
        test_image_path = os.path.join(test_images_path, test_idx + '.jpg')
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Encode the test image using the predictor
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Compute cosine similarity between target feature and test image features
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

        # Select the top-1 point in the test image based on similarity map
        topk_xy, topk_label = point_selection(sim, topk=1)

        # First-step prediction from SAM using the point prompt
        masks, scores, logits, logits_high = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)

        # Weight the multi-scale logits and sum
        logits_high = logits_high * weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * weights_np[..., None]
        logit = logits.sum(0)

        # Cascaded Post-refinement-1: use non-zero region of the predicted mask as a box prompt
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

        # Cascaded Post-refinement-2: refine the mask further using the best index and updated box prompt
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
        
        # Save visualization: overlay the selected mask on test image
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')
        plt.close()

        # Save the final mask as a PNG image with a blue overlay (value [0, 0, 128])
        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


class Mask_Weights(nn.Module):
    """
    A simple network that learns weights for each mask scale.
    The weights are initialized to a fraction of one.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, topk=1):
    """
    Selects the top-k point(s) based on the similarity map.
    
    Args:
       mask_sim: A similarity map tensor.
       topk: The number of points to select.
    
    Returns:
       topk_xy: Coordinates of the selected points.
       topk_label: A label array (all 1's) for the selected points.
    """
    # Assume mask_sim is a 2D tensor
    w, h = mask_sim.shape
    topk_indices = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_indices // h).unsqueeze(0)
    topk_y = (topk_indices - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    return topk_xy.cpu().numpy(), topk_label


def calculate_dice_loss(inputs, targets, num_masks=1):
    """
    Computes the DICE loss, similar to the generalized IoU for masks.
    
    Args:
        inputs: Predictions tensor (logits) of arbitrary shape.
        targets: Ground truth binary mask tensor (same shape as inputs).
    
    Returns:
        Dice loss value.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha=0.25, gamma=2):
    """
    Computes the sigmoid focal loss used in dense detection.
    
    Args:
        inputs: Predictions tensor (logits) of arbitrary shape.
        targets: Ground truth binary mask tensor (same shape as inputs).
        alpha: Weighting factor to balance positive and negative examples.
        gamma: Exponent of the modulating factor to focus on hard examples.
    
    Returns:
        Focal loss value.
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
