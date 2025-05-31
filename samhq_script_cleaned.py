"""
This is the production-ready Personalize-SAM-HQ training script.
Each section is annotated with human-style comments to help you understand what is happening; I've tried to elucidate as much as possible.
"""

import os  # For handling file paths and directories
import argparse  # For parsing command-line arguments
import numpy as np  # For numerical operations
import torch  # Main PyTorch library
import torch.optim as optim  # For optimization routines
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for layers and activations
from torch.autograd import Variable  # For automatic differentiation (if needed)
import matplotlib.pyplot as plt  # For plotting figures
import cv2  # OpenCV for image processing
import random  # Python random number generator
from typing import Dict, List, Tuple  # For type annotations

# Importing SAM model related functions and classes
from per_segment_anything.build_sam import sam_model_registry
from per_segment_anything.modeling import TwoWayTransformer, MaskDecoder

# Importing custom utilities for data loading and loss calculations
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc

import csv  # For logging epoch stats to CSV files
import time  # For tracking timing
import psutil  # For system performance statistics
import pynvml  # For GPU monitoring
import json  # For saving training metrics in JSON format

# -----------------------------------------------------------------------------
# GPU Monitoring and System Statistics Functions
# -----------------------------------------------------------------------------

def init_gpu_monitor():
    """
    Initialize NVML for monitoring GPU stats if GPU is available.
    """
    if torch.cuda.is_available():
        pynvml.nvmlInit()


def get_gpu_stats() -> Dict:
    """
    Gather statistics about all available GPUs.
    Returns a dictionary with memory usage, utilization, and temperature.
    """
    stats = {}
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats[f"gpu_{i}_memory_used_MB"] = mem_info.used / 1024**2
        stats[f"gpu_{i}_memory_total_MB"] = mem_info.total / 1024**2
        stats[f"gpu_{i}_utilization_%"] = util_info.gpu
        stats[f"gpu_{i}_temperature_C"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    return stats


def get_system_stats() -> Dict:
    """
    Get system-wide statistics including CPU, RAM, and disk usage.
    Returns a dictionary of these statistics.
    """
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_MB": psutil.virtual_memory().used / 1024**2,
        "ram_total_MB": psutil.virtual_memory().total / 1024**2,
        "disk_used_GB": psutil.disk_usage("/").used / 1024**3,
        "disk_total_GB": psutil.disk_usage("/").total / 1024**3,
    }
    return stats


def log_epoch_stats(epoch, output_path, start_time, end_time):
    """
    Save overall statistics including GPU and system resources for the epoch into a CSV file.
    - epoch: Current epoch number.
    - output_path: Directory where the CSV will be written.
    - start_time, end_time: Timing for duration calculation.
    """
    system_stats = get_system_stats()  # CPU/RAM/Disk statistics
    gpu_stats = get_gpu_stats() if torch.cuda.is_available() else {}
    duration = end_time - start_time

    # Combine all stats into one dictionary
    stats = {
        "epoch": epoch,
        "duration_sec": duration,
        **system_stats,
        **gpu_stats,
    }

    # Define the CSV file path
    csv_file = os.path.join(output_path, "epoch_stats.csv")

    # If file not exists, write headers
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()

    # Append current epoch stats to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writerow(stats)

    print(f"Logged resource stats for epoch {epoch} to {csv_file}")

# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """
    Custom 2D Layer Normalization. Normalizes each channel of the feature maps.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Learnable scale and shift parameters for normalization
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the mean and variance across channels and normalize
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # Apply learnable scale and shift
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron module with configurable number of layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        # Build a list of hidden layers all with the same hidden_dim
        h = [hidden_dim] * (num_layers - 1)
        # Create linear layers connecting input, hidden, and output dimensions
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # Pass input through each layer and apply ReLU activation except for the last layer
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)  # Apply sigmoid if specified
        return x


class MaskDecoderHQ(MaskDecoder):
    """
    HQ Mask Decoder based on SAM's MaskDecoder.
    This module adds additional operations to enhance the mask predictions.
    """
    def __init__(self, model_type):
        # Initialize parent MaskDecoder with a fixed transformer configuration
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
        # Ensure the correct model type is supplied
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        
        # Map model type to a pretrained checkpoint for the mask decoder
        checkpoint_dict = {"vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           "vit_h": "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        # Load the pretrained weights
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        # Freeze parameters so they are not updated during training
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        # Map model type to the corresponding ViT dimension
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        # Create additional learnable tokens and MLP for further refinement
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1  # Increase to add HQ token

        # Module to compress ViT features from a higher dimension to transformer_dim
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )
        
        # Module to further process the image embeddings
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        # Module to transform and enhance mask features
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for predicting masks.
        Uses image embeddings plus prompt embeddings to generate HQ masks.
        """
        # Get early ViT features and adjust dimensions for fusion
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        # Process each image in the batch separately
        for i_batch in range(batch_len):
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

        # Depending on whether multiple masks are desired, select the appropriate ones
        if multimask_output:
            # Select mask with the highest IoU score
            mask_slice = slice(1, self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # Use only the first mask (default single mask output)
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        # Extract the HQ mask (which is the last token)
        masks_hq = masks[:, slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to predict masks using transformer and upscaling.
        """
        # Combine IoU token, mask tokens, and hf token into one tensor of tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # Concatenate these tokens with sparse prompts
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # For each prompt, repeat image embeddings so that they align with tokens
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings  # Fuse with dense prompt embeddings

        # Similarly, repeat positional encoding for the transformer
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run transformer to get hidden states (hs) and updated src
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]  # First token output for IoU prediction
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # Rest for mask prediction

        # Upscale the transformer output to feature map dimensions
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        # Fuse upscaled embedding with hq feature from the encoder
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        # Generate hypernetwork parameters from each mask token
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        
        # Predict masks using matrix multiplication: token parameters * upscaled embedding
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)
        
        # Get final IoU predictions from the iou head
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# -----------------------------------------------------------------------------
# Functions to Save Model Checkpoints
# -----------------------------------------------------------------------------

def save_encoder_and_decoder(encoder, decoder, output_path, epoch):
    """
    Save encoder and decoder weights separately into dedicated directories.
    This helps in tracking and resuming training.
    """
    encoder_dir = os.path.join(output_path, "encoderepoch")
    decoder_dir = os.path.join(output_path, "decoderepoch")
    
    # Create directories if they do not exist
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    if not os.path.exists(decoder_dir):
        os.makedirs(decoder_dir)
    
    # Save encoder checkpoint
    encoder_checkpoint_path = os.path.join(encoder_dir, f"encoder_epoch_{epoch}.pth")
    torch.save(encoder.state_dict(), encoder_checkpoint_path)
    print(f"Encoder saved at {encoder_checkpoint_path}")
    
    # Save decoder checkpoint
    decoder_checkpoint_path = os.path.join(decoder_dir, f"decoder_epoch_{epoch}.pth")
    torch.save(decoder.state_dict(), decoder_checkpoint_path)
    print(f"Decoder saved at {decoder_checkpoint_path}")

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    """
    Given a list of masks and associated information, generate visualization images.
    """
    if len(masks) == 0:
        return

    # Loop through each mask and overlay it on the original image
    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)  # Show original image
        show_mask(mask, plt.gca())  # Overlay mask
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        # Save the figure with a unique filename per mask
        plt.savefig(filename+'_'+str(i)+'.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_mask(mask, ax, random_color=False):
    """
    Helper to display a semi-transparent mask on a given axis.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_points(coords, labels, ax, marker_size=375):
    """
    Visualize points on the image. Green if positive, red if negative.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
def show_box(box, ax):
    """
    Draw a rectangular box on the given axis.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# -----------------------------------------------------------------------------
# Argument Parsing and Main Pipeline
# -----------------------------------------------------------------------------

def get_args_parser():
    """
    Define and return the argument parser.
    This allows the user to configure training via command-line arguments.
    """
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=False, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--learning_rate', default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument('--start_epoch', default=0, type=int, help="Epoch to start training")
    parser.add_argument('--lr_drop_epoch', default=10, type=int, help="Epoch interval to drop learning rate")
    parser.add_argument('--max_epoch_num', default=20, type=int, help="Maximum number of epochs to train")
    parser.add_argument('--input_size', default=[1024,1024], type=list, help="Input image size")
    parser.add_argument('--batch_size_train', default=4, type=int, help="Training batch size")
    parser.add_argument('--batch_size_valid', default=1, type=int, help="Validation batch size")
    parser.add_argument('--model_save_fre', default=1, type=int, help="Frequency of saving model checkpoints")
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='Rank for distributed processes')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed processes')
    parser.add_argument('--find_unused_params', action='store_true', help="Flag to find unused parameters in DDP")
    parser.add_argument('--eval', action='store_true', help="Run evaluation instead of training")
    parser.add_argument('--visualize', action='store_true', help="Visualize results during evaluation")
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):
    """
    Main entry point for training or evaluation.
    Sets up distributed training, loads data, initializes SAM, and calls training/evaluation routines.
    """
    # Initialize distributed training if needed.
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    # Set random seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the SAM model from the registry using the provided model type and checkpoint.
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    ### --- Prepare DataLoaders for Training and Validation ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                my_transforms=[RandomHFlip(), LargeScaleJitter()],
                                                                batch_size=args.batch_size_train,
                                                                training=True)
        print(len(train_dataloaders), " train dataloaders created")
    
    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms=[Resize(args.input_size)],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Initialize Distributed DataParallel for the SAM model ---
    if torch.cuda.is_available():
        sam.cuda()
        net.cuda()
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)
    sam_without_ddp = sam.module

    ### --- Depending on the flag, either Train or Evaluate the model ---
    if not args.eval:
        print("--- define optimizer for training the encoder only---")
        optimizer = optim.Adam(sam_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                sam_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                sam_without_ddp.load_state_dict(torch.load(args.restore_model, map_location="cpu"))
    
        evaluate(args, net, sam, valid_dataloaders, args.visualize)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    """
    Training loop.
    Trains the SAM encoder along with the HQ decoder and logs metrics, saving checkpoints along the way.
    """
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
    # Dictionary to store training metrics over epochs.
    training_metrics = {"loss": [], "val_iou": [], "val_boundary_iou": []}

    init_gpu_monitor()
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    lr_scheduler.step()

    # Reload SAM to reset optimizer state and prepare for training.
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.cuda()
    sam.train()
#------------------------------------------------------------------------------------------#
# Critical selection: Image Encoder, Mask Decoder, Prompt encoder, Mask Decoder HQ training
#------------------------------------------------------------------------------------------#

    
    # Allow gradients for SAM image encoder so it can be trained.
    for name, param in sam.image_encoder.named_parameters():
        param.requires_grad = True


    # Allow gradients for SAM prompt encoder so it can be trained.
    for name, param in sam.prompt_encoder.named_parameters():
        param.requires_grad = True


    # Allow gradients for SAM mask decoder so it can be trained.
    for name, param in sam.mask_decoder.named_parameters():
        param.requires_grad = True


    # Ensure all parameters of HQ decoder are trainable.
    for name, param in net.named_parameters():
        param.requires_grad = True
        
    # Redefine optimizer to take only the parameters that require gradients.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sam.parameters()),  
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0   
    )
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)
        
    # Begin training over epochs.
    for epoch in range(epoch_start, epoch_num): 
        print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        start_time = time.time()
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        epoch_loss = 0.0  # Accumulate loss for the epoch

        # Iterate over the batches in training dataloader.
        for data in metric_logger.log_every(train_dataloaders, 1000):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # Prepare input prompt keys and compute associated labels.
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
            except:
                # If there are too few points, remove point prompts.
                input_keys = ['box', 'noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            # Build a batch of dict input prompts.
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
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

            # Pass the inputs through SAM to get embeddings.
            batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            # Concatenate embeddings from each batch sample.
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            # Forward pass through the HQ decoder network.
            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            # Calculate losses: mask-level loss and dice loss.
            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            # Reduce losses across GPUs (if using distributed training).
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            epoch_loss += loss_value

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        # Gather training statistics from the metric logger.
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        # Save the loss and validation IoU metrics.
        val_iou = train_stats.get("val_iou_0", 0.0)
        val_boundary_iou = train_stats.get("val_boundary_iou_0", 0.0)
        training_metrics["loss"].append(epoch_loss / train_num)
        training_metrics["val_iou"].append(val_iou)
        training_metrics["val_boundary_iou"].append(val_boundary_iou)

        # Write training metrics to a JSON file.
        with open(args.output + "/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=4)

        lr_scheduler.step()  # Update learning rate if needed.
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)

        end_time = time.time()
        log_epoch_stats(epoch, args.output, start_time, end_time)
        
        sam.train()  # Switch SAM back to training mode
        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_" + str(epoch) + ".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.state_dict(), args.output + model_name)
            save_encoder_and_decoder(encoder=sam.module.image_encoder, decoder=sam.module.mask_decoder, output_path=args.output, epoch=epoch)
    print("Training Reaches The Maximum Epoch Number")
    
    # After training, merge the SAM and HQ decoder checkpoints.
    if misc.is_main_process():
        trained_sam_ckpt = sam.module.state_dict()
        hq_decoder_ckpt = torch.load(args.output + model_name)
        # Merge the HQ decoder weights into the SAM checkpoint namespace.
        for key in hq_decoder_ckpt.keys():
            hq_key = 'mask_decoder.' + key
            trained_sam_ckpt[hq_key] = hq_decoder_ckpt[key]
        # Save the combined checkpoint.
        model_name = "/sam_hq_epoch_" + str(epoch) + ".pth"
        torch.save(trained_sam_ckpt, args.output + model_name)
    print(f"Trained SAM+HQ model saved at {args.output + model_name}")


def compute_iou(preds, target):
    """
    Compute Intersection-over-Union (IoU) for predicted masks.
    Assumes one mask per image.
    """
    assert target.shape[1] == 1, 'only support one mask per image now'
    # If sizes differ, resize predictions to match target
    if(preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou += misc.mask_iou(postprocess_preds[i], target[i])
    return iou / len(preds)


def compute_boundary_iou(preds, target):
    """
    Compute Boundary IoU for the predicted masks.
    """
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou += misc.boundary_iou(target[i], postprocess_preds[i])
    return iou / len(preds)


def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    """
    Run evaluation on validation datasets.
    Computes IoU, boundary IoU and optionally visualizes the predictions.
    """
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader, 1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = (
                data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            )

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            # Convert masks into bounding boxes
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                # (Point or noise_mask could be added similarly if needed.)
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            # Compute IoU and boundary IoU between predictions and ground truth
            iou = compute_iou(masks_hq, labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq, labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                       
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        print('============================')
        # Synchronize statistics across distributed processes.
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)

    return test_stats


# -----------------------------------------------------------------------------
# Dataset Configuration and Script Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------- Configure Train and Valid Datasets ---------------------
    dataset_dis = {
        "name": "DIS5K-TR",
        "im_dir": "./data/DIS5K/DIS-TR/im",
        "gt_dir": "./data/DIS5K/DIS-TR/gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_thin = {
        "name": "ThinObject5k-TR",
        "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
        "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_fss = {
        "name": "FSS",
        "im_dir": "./data/cascade_psp/fss_all",
        "gt_dir": "./data/cascade_psp/fss_all",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_duts = {
        "name": "DUTS-TR",
        "im_dir": "./data/cascade_psp/DUTS-TR",
        "gt_dir": "./data/cascade_psp/DUTS-TR",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_duts_te = {
        "name": "DUTS-TE",
        "im_dir": "./data/cascade_psp/DUTS-TE",
        "gt_dir": "./data/cascade_psp/DUTS-TE",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_ecssd = {
        "name": "ECSSD",
        "im_dir": "./data/cascade_psp/ecssd",
        "gt_dir": "./data/cascade_psp/ecssd",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_msra = {
        "name": "MSRA10K",
        "im_dir": "./data/cascade_psp/MSRA_10K",
        "gt_dir": "./data/cascade_psp/MSRA_10K",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    # Valid set example configurations.
    dataset_coift_val = {
        "name": "COIFT",
        "im_dir": "./data/thin_object_detection/COIFT/images",
        "gt_dir": "./data/thin_object_detection/COIFT/masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_hrsod_val = {
        "name": "HRSOD",
        "im_dir": "./data/thin_object_detection/HRSOD/images",
        "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_thin_val = {
        "name": "ThinObject5k-TE",
        "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
        "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_dis_val = {
        "name": "DIS5K-VD",
        "im_dir": "./data/DIS5K/DIS-VD/im",
        "gt_dir": "./data/DIS5K/DIS-VD/gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    # Personal custom datasets for additional experiments.
    dataset_renishaw_train = {
        "name": "augmented_data_with_all_renishaw_data",
        "im_dir": "./augmented_data_with_all_renishaw_data/train/images",
        "gt_dir": "./augmented_data_with_all_renishaw_data/train/masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    dataset_renishaw_test = {
        "name": "augmented_data_with_all_renishaw_data",
        "im_dir": "./augmented_data_with_all_renishaw_data/test/images",
        "gt_dir": "./augmented_data_with_all_renishaw_data/test/masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    # Uncomment below to use multiple datasets for training and validation.
    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val]
    # Following two are used for illustration. To use the renishaw ones, make the dir structure accordingly
    train_datasets = [dataset_dis]
    valid_datasets = [dataset_dis_val]

    # Parse command-line arguments.
    args = get_args_parser()
    # Initialize the HQ decoder model with the specified model type.
    net = MaskDecoderHQ(args.model_type)

    # Run the main training or evaluation pipeline.
    main(net, train_datasets, valid_datasets, args)
