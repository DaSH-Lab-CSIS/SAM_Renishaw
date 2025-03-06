import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pycocotools import mask as mask_utils

def create_annotations(data_root, img_folder='im', gt_folder='gt', output_folder='../../phase2/annotations', split = 'train'):

    # File paths
    img_path = Path(data_root) / img_folder
    gt_path = Path(data_root) / gt_folder
    anno_path = Path(data_root) / output_folder / split
    anno_path.mkdir(parents=True, exist_ok=True)

    # Counters
    count = 0

    for img_file in tqdm(img_path.glob('*.jpg'), desc="Processing Images", unit="file"):
        name = img_file.stem  # Image name
        img = Image.open(img_file)  # img dtype = <class 'PIL.Image.Image'>
        mask = np.array(Image.open(gt_path / f"{name}.png").convert('1'))  # Note: png
        height, width = mask.shape

        # Compute bounding box
        coords = np.argwhere(mask)  # Coordinates of all non-zero elements in the binary mask
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]  # Convert bbox values to int

        # Compute bounding box center
        x_center = int((x_min + x_max) // 2)
        y_center = int((y_min + y_max) // 2)
        point_coords = [[x_center, y_center]]  # Explicitly cast to Python int

        # Compute segmentation (RLE format)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))  # RLE encoding
        rle['counts'] = rle['counts'].decode('utf-8')  # Decode counts to a string

        # Compute area
        area = int(mask.sum())  # Convert to Python int

        # Prepare annotation JSON
        annotation = {
            "image": {
                "file_name": img_file.name,
                "height": int(height),
                "width": int(width)
            },
            "annotations": [
                {
                    "bbox": bbox,
                    "point_coords": point_coords,
                    "segmentation": rle,
                    "area": area
                }
            ]
        }

        # Save JSON file
        with open(anno_path / f"{name}.json", 'w') as f:
            json.dump(annotation, f, indent=4)
            count += 1

    print(f"Total of: {count} annotations saved to {anno_path}\n")

#----------------------------------------------------------------------------------------

# Prepare raw Renishaw data
if __name__ == "__main__":
    # Train split
    data_root = './datasets/phase2-raw/train'
    create_annotations(data_root, split='train')
    # Test split
    data_root = './datasets/phase2-raw/test'
    create_annotations(data_root, split='val')
