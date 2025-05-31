import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import json
import argparse

from per_segment_anything import sam_model_registry
from segment_anything import SamPredictor

def load_sam_checkpoint(sam, checkpoint_path):
    """
    Loads the SAM checkpoint from the given path, filters out keys related to
    relative positional embeddings and loads the state dictionary into the model.
    """
    print("[DEBUG] Loading checkpoint from:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    keys_to_remove = [k for k in state_dict.keys() if "attn.rel_pos" in k]
    for key in keys_to_remove:
        print(f"[DEBUG] Removing key {key} from checkpoint state_dict")
        del state_dict[key]
    sam.load_state_dict(state_dict, strict=False)
    print("[DEBUG] Checkpoint loaded with filtered keys.")
    return sam

# --------------------------
# Visualization Helper Functions
# --------------------------
def show_mask(mask, ax, random_color=False):
    """
    Displays a semi-transparent mask overlay on the provided axis.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        print("[DEBUG] Using random color for mask overlay.")
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    print("[DEBUG] Mask overlay displayed.")

def show_points(coords, labels, ax, marker_size=375):
    """
    Displays points on the image axis.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1],
               color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1],
               color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    print(f"[DEBUG] Displayed {len(pos_points)} positive and {len(neg_points)} negative points.")

def show_box(box, ax):
    """
    Draws a rectangular box (x0, y0, x1, y1) on the provided axis.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    print(f"[DEBUG] Box drawn: ({x0}, {y0}, {w}, {h})")

def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    """
    Saves the visualization for each mask with the score printed.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"[DEBUG] Processing visualization for mask index {i} with score {score:.3f}")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, plt.gca())
        print(f"[DEBUG] Score: {score:.3f}")
        plt.axis('off')
        out_file = filename + f'_{i}.png'
        plt.savefig(out_file, bbox_inches='tight', pad_inches=-0.1)
        print(f"[DEBUG] Output saved at {out_file}")
        plt.close()

# --------------------------
# Inference Function (for Distributed Setup)
# --------------------------
def inference(rank, world_size, images, input_points, input_labels, result_path, args):
    print(f"[DEBUG] Process rank {rank} starting inference.")
    setup(rank, world_size)
    
    # Initialize model
    sam_checkpoint = args.ckpt
    model_type = args.sam_type
    print(f"[DEBUG] Using model type: {model_type} and checkpoint: {sam_checkpoint}")
    # Setting device using rank or fallback (here we use device 1 for simplicity)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Process {rank} running on device: {device}")
    sam = sam_model_registry[model_type](checkpoint=None).to(device)
    sam = load_sam_checkpoint(sam, sam_checkpoint)
    sam = sam.to(device)
    model = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[1])
    predictor = SamPredictor(model.module)
    print(f"[DEBUG] SAM predictor initialized for process {rank}.")
    
    os.makedirs(result_path, exist_ok=True)
    num_images = len(images)
    images_per_rank = num_images // world_size
    start_index = rank * images_per_rank
    end_index = start_index + images_per_rank
    print(f"[DEBUG] Process {rank} handling images {start_index} to {end_index-1}.")
    for i in range(start_index, end_index):  # each rank processes its slice of images
        print(f"[DEBUG] Process {rank} processing image index {i}.")
        image = cv2.imread(images[i])
        if image is None:
            print(f"[DEBUG] Warning: Unable to read image at {images[i]}. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        input_box = None        
        if input_points[i] is not None:
            input_point = np.array(input_points[i], dtype=float)
        else:
            input_point = np.array([[0,0],[0,0]], dtype=float)
        if input_labels[i] is not None:
            input_label = input_labels[i]
        else:
            input_label = np.ones(input_point.shape[0])
        hq_token_only = False

        print(f"[DEBUG] Process {rank}, Image {i}: Input Points: {input_point}, Input Labels: {input_label}")

        # Run inference using the predictor
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        print(f"[DEBUG] Process {rank}, Image {i}: Inference complete. Number of masks: {len(masks)}")
        filename = os.path.join(result_path, f'example_{i}_rank_{rank}')
        show_res(masks, scores, input_point, input_label, input_box, filename, image)
    
    print(f"[DEBUG] Process {rank} finished inference.")
    cleanup()

def setup(rank, world_size):
    """
    Sets up the environment variables and initializes the distributed process group.
    """
    print(f"[DEBUG] Process {rank}: Setting up DDP environment.")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11123'
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[DEBUG] Process {rank}: DDP process group initialized.")

def cleanup():
    """
    Cleans up the distributed process group.
    """
    print("[DEBUG] Cleaning up DDP process group.")
    torch.distributed.destroy_process_group()

# --------------------------
# Main Function and Argument Parsing
# --------------------------
def get_arguments():
    """
    Parses command-line arguments including the SAM checkpoint path and model type.
    """
    parser = argparse.ArgumentParser(description="Debug Inference with SAM Encoder and Decoder")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to the SAM checkpoint (encoder+decoder) to load")
    parser.add_argument('--sam_type', type=str, default='vit_b',
                        help="SAM model type (e.g., 'vit_b' or 'vit_t')")
    args = parser.parse_args()
    print(f"[DEBUG] Parsed arguments: {args}")
    return args

def main():
    """
    Main function that collects image file paths and prompt data,
    prints the result path, and spawns distributed processes for inference.
    """
    args = get_arguments()
    world_size = 1  # Number of GPUs (adjust as necessary)
    image_dir = '/home/samhq/sam-hq/Personalize-SAM-HQ/input_imgs_renishaw_select'
    json_dir = '/home/samhq/sam-hq/Personalize-SAM-HQ/json_select'
    
    print("[DEBUG] Collecting images and prompt data.")
    image_files = os.listdir(image_dir)
    images = [os.path.join(image_dir, f) for f in image_files if f.endswith('.png')]
    
    input_points = []
    input_labels = []
    for image_file in image_files:
        if image_file.endswith('.png'):
            json_filename = image_file.replace('.png', '.json')
            json_path = os.path.join(json_dir, json_filename)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    clicked_points = np.array(json_data.get("clicked_points", []), dtype=float)
                    input_points.append(clicked_points)
                    input_labels.append(np.ones(clicked_points.shape[0]))
                    print(f"[DEBUG] Loaded {len(clicked_points)} clicked points from {json_filename}.")
            else:
                input_points.append(None)
                input_labels.append(None)
                print(f"[DEBUG] JSON file {json_filename} not found; appending None.")
    
    result_path = 'outputs/outdir_ren/'
    print(f"[DEBUG] Results will be stored in: {result_path}")
    
    mp.spawn(inference, args=(world_size, images, input_points, input_labels, result_path, args), nprocs=world_size)

if __name__ == "__main__":
    main()
