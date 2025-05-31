  # Personalize SAM HQ

This repository contains the code for training, debugging, and inference of the Personalize-SAM-HQ framework. It integrates improvements from SAM, PerSAM, SAM-HQ to provide a refined segmentation solution. Below is a detailed guide on how everything is organized and how to get started.
(Note: All of the following is tuned for ViT-B, make changes where necessary to use other ViT models.)
---

## INSTALLATION

### Clone the Repository

```bash
git clone https://github.com/yourusername/Personalize-SAM-HQ.git
cd Personalize-SAM-HQ
```

### Setup the Environment

We provide a Conda environment file that lists all dependencies. To install run:

```bash
conda env create -f conda_dependencies.yml
conda activate your_env_name
```

---

## DEPENDENCIES

- **Python 3.8+**
- **PyTorch** with CUDA support
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **psutil**
- **pynvml** (for GPU monitoring; ensure your NVIDIA drivers are installed)
- **argparse** (standard library)
- **Other utility libraries** as listed in the conda_dependencies.yml

*Also see: [pynvml documentation](https://docs.nvidia.com/deploy/nvml-api/) for details on GPU monitoring.*

---

## SETUP

### Folder Structure and Data Setup

**Input Files:**

- **Images:**  
  For inference with SAMHQ Inference script, place all input images (PNG format) in a folder, e.g.:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/input_imgs_renishaw_select`
  For inference with Persam Inference script, the images and annotations need to be grouped on the basis of the subject and placed into the Annotation (PNG) and Images (JPG) folders respectively with names as double digit numbers starting from 00 (e.g. 00.png, 01.png, 02.png etc.). 
  Place image files (JPG/PNG) and any accompanying data required for training or evaluation in clearly defined folders for training; JPGs for images (im), PNGS for masks (gt). 

- **JSON Prompts (for Inference):**  
  Create corresponding JSON files with the same base name (with `.json` extension) in a separate folder, e.g.:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/json_select`  
  Each JSON must include a field called `"clicked_points"` (list of point coordinates) used as prompt input.

  For **Training** and **Inference**, organize your dataset to follow a similar directory structure:
  ```
      Personalize-SAM-HQ
      ├── app.py
      ├── conda_dependencies.yml
      ├── data
      │   ├── Annotations
      │   ├── DIS5K
      │   │   ├── DIS-TR
      │   │     ├── gt
      │   │     └── im
      │   │   ├── DIS-VD
      │   │     ├── gt
      │   │     └── im
      │   ├── Images
      ├── inference_ren.py
      ├── inference_ren_debug.py
      ├── input_imgs_renishaw_select
      │   ├── RVP_20240918102156.png
      │   ├── RVP_20240918102418.png
      │   ├── RVP_20240918114817.png
      │   ├── RVP_20240918114844.png
      │   └── RVP_20240918115714.png
      ├── json_select
      │   ├── RVP_20240918102156.json
      │   ├── RVP_20240918102418.json
      │   ├── RVP_20240918114817.json
      │   ├── RVP_20240918114844.json
      │   └── RVP_20240918115714.json
      ├── LICENSE.txt
      ├── outputs
      │   ├── outdir_ren
      ├── persam_f_ren.py
      ├── persam_f_ren_debug.py
      ├── per_segment_anything
      │   ├── automatic_mask_generator.py
      │   ├── build_sam.py
      │   ├── init.py
      │   ├── modeling
      │   ├── predictor.py
      │   ├── pycache
      │   └── utils
      ├── pretrained_checkpoint
      │   ├── sam_vit_b_maskdecoder.pth
      ├── README.md
      ├── requirements.txt
      ├── samhq_debug_script.py
      ├── samhq_script.py
      ├── sam_vit_b_01ec64.pth
      ├── segment_anything
      │   ├── automatic_mask_generator.py
      │   ├── build_sam_baseline.py
      │   ├── build_sam.py
      │   ├── init.py
      │   ├── modeling
      │   ├── predictor.py
      │   ├── pycache
      │   └── utils
      ├── show.py
      ├── utils
      │   ├── dataloader.py
      │   ├── loss_mask.py
      │   ├── misc.py
      │   └── pycache
      ├── weights
      │   └── mobile_sam.pt
      └── work_dirs
  ```

---

## TRAINING

### Command-Line Arguments (Training Script)

All training arguments are parsed by the `get_args_parser()` function in the debug training script (`samhq_debug_script.py`).

- **`--output` (required):**  
  Output directory to store model checkpoints, CSV logs, and JSON training metrics.  
  _Example:_  
  `--output ./outputs`

- **`--model-type` (optional):**  
  SAM model variant (e.g., `vit_h`, `vit_l`, `vit_b`).  
  _Default:_ `vit_l`  
  _Example:_  
  `--model-type vit_b`

- **`--checkpoint` (optional):**  
  Path to a pretrained SAM checkpoint (for evaluation or resuming).  
  _Example:_  
  `--checkpoint ./checkpoints/sam_checkpoint.pth`

- **`--device` (optional):**  
  Compute device, usually `"cuda"`.  
  _Default:_ `cuda`

- **`--seed` (optional):**  
  Random seed for reproducibility.  
  _Default:_ `42`

- **Training Hyperparameters:**
  - `--learning_rate`: e.g., `1e-3`
  - `--start_epoch`: e.g., `0`
  - `--lr_drop_epoch`: e.g., `10`
  - `--max_epoch_num`: e.g., `5`
  - `--batch_size_train`: e.g., `4`
  - `--batch_size_valid`: e.g., `1`
  - `--model_save_fre`: How frequently to save model checkpoints (e.g., every epoch)

- **Distributed Training Parameters:**
  - `--world_size`: e.g., `1`
  - `--dist_url`: e.g., `env://`
  - `--rank` and `--local_rank`: For DDP setup

- **Other Flags:**
  - `--find_unused_params`, `--eval`, `--visualize`: For evaluation mode and debugging.

### How to Run the Training Script

1. **Prepare Your Data:**  
   Organize your training images and annotations according to the folder structure described above.

2. **Run the Training Script:**

   Open a terminal, navigate to the repository root, and execute:

   ```bash
   python samhq_script.py --output ./outputs --model-type vit_b --checkpoint ./checkpoints/your_checkpoint.pth --device cuda --seed 42 --learning_rate 1e-3 --max_epoch_num 5 --batch_size_train 1 --batch_size_valid 1 --world_size 1
   ```

3. **Distributed Training:**  
   By default, the script uses a world size of 1. Modify `--world_size` and pass the appropriate device settings if running on multiple GPUs.

### Training Pipeline Detailed Information

- **Monitoring:**  
  GPU and system statistics (CPU, RAM, Disk) are logged per epoch in CSV files (located in the output directory).
- **Model Saving:**  
  The encoder and decoder checkpoints are saved under `encoderepoch` and `decoderepoch` folders respectively. A combined checkpoint (`sam_hq_epoch_{epoch}.pth`) is also saved.
- **Debug Information:**  
  In the provided debug script, extensive debug messages are printed during training to log data shapes, optimizer steps, and loss values.

---

## INFERENCE

There are two inference scripts supported:

### SAMHQ INFERENCE SCRIPT

**Folder Structure and Data Setup:**

- **Images Directory:**  
  Place input images (PNG format) in:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/input_imgs_renishaw_select`

- **JSON Prompts Directory:**  
  Place JSON files (with the same base names as images) containing `"clicked_points"` in:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/json_select`

**Command-Line Arguments:**

- **`--ckpt` (required):**  
  Path to the SAM checkpoint (encoder + decoder).  
  _Example:_  
  `--ckpt /home/samhq/sam-hq/Personalize-SAM-HQ/work_dirs/non_encoder/sam_hq_epoch_4.pth`

- **`--sam_type` (optional):**  
  SAM model type to use (default: `vit_b`).  
  _Example:_  
  `--sam_type vit_b`

**How to Run:**

1. **Prepare Data:**  
   Ensure the images and their corresponding JSON prompt files are in the specified folders.

2. **Run the Inference Script:**

   Open a terminal and run:

   ```bash
   python inference_encoder_decoder.py --ckpt /home/samhq/sam-hq/Personalize-SAM-HQ/work_dirs/non_encoder/sam_hq_epoch_4.pth --sam_type vit_b
   ```

3. **Distributed Processing:**  
   The script defaults to a `world_size` of 1. Modify device settings as needed.

### PERSAM INFERENCE SCRIPT

**Folder Structure and Data Setup:**

- **Images Directory:**  
  Place input images (JPG format) under:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/data/Images/`  
  Organize by object name (e.g., `chair`, `table`).

- **Annotations (Masks) Directory:**  
  Place corresponding mask images (PNG format) under:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/data/Annotations/`  
  Each object folder should contain the reference mask specified by `--ref_idx`.

**Command-Line Arguments:**

- **`--data` (optional):**  
  Root directory for input data.  
  _Default:_ `./data`

- **`--outdir` (optional):**  
  Output subdirectory name.  
  _Default:_ `persam_f`

- **`--ckpt` (required):**  
  Path to the SAM checkpoint, e.g.:  
  `--ckpt /home/samhq/sam-hq/Personalize-SAM-HQ/work_dirs/non_encoder/epoch_4.pth`

- **`--sam_type` (optional):**  
  SAM model type (default: `vit_b`)

- **`--lr`, `--train_epoch`, `--log_epoch` (optional):**  
  Learning rate, training iterations, and logging frequency.

- **`--ref_idx` (optional):**  
  Reference index for annotation selection (default: `00`).

**How to Run:**

Prepare your data (see above) then run:

```bash
python persam_f_ren.py --data ./data --outdir persam_f --ckpt /home/samhq/sam-hq/Personalize-SAM-HQ/work_dirs/non_encoder/epoch_4.pth --sam_type vit_b --lr 1e-3 --train_epoch 1000 --log_epoch 200 --ref_idx 00
```

**Distributed / Debug Information:**

The debug script prints detailed debug messages indicating current object processing, SAM model loading, training progress of mask weights, and segmentation outputs.

---

## OUTPUTS

### Where the Output Is Stored

- **Training Outputs:**  
  Model checkpoints, CSV files logging epoch statistics, and JSON training metrics are stored in the directory specified by `--output` (e.g., `./outputs`).  
  - Checkpoints:  
    - Encoder checkpoints: `encoderepoch/encoder_epoch_{epoch}.pth`  
    - Decoder checkpoints: `decoderepoch/decoder_epoch_{epoch}.pth`  
    - Combined model: `sam_hq_epoch_{epoch}.pth`
  - Logs: CSV files (e.g., `epoch_stats.csv`) and JSON files (e.g., `training_metrics.json`).

- **Inference Outputs (SAMHQ):**  
  Visualized images with mask overlays and printed scores are stored in:  
  `/home/samhq/sam-hq/Personalize-SAM-HQ/output/outdir_ren/`  
  *You can change this output path by modifying the `result_path` variable in the script.*

- **Inference Outputs (PERSAM):**  
  Results are stored under `./outputs/<outdir>/` (default, e.g. `./outputs/persam_f/`).  
  Each file follows a naming convention, for example:  
  `example_<image_index>_rank_<rank>_<mask_index>.png`  
  where `example_3_rank_0_0.png` corresponds to the first mask of image 3 processed on rank 0.

### Score Display

- **During Training:**  
  Loss values (mask loss and Dice loss), IoU scores, learning rate, and other metrics are printed on the console.
- **During Inference:**  
  Performance scores (e.g., IoU, boundary IoU) are overlaid on output visualizations and also printed in the terminal.

---

## ADDITIONAL DEBUG INFORMATION

The debug edition of the training and inference scripts (e.g., samhq_debug_script.py) prints extensive information to help diagnose issues:

- **GPU Monitoring:**  
  Outputs indicate whether NVML is initialized. Detailed statistics for each GPU (memory usage, utilization, temperature) are printed.

- **Neural Network Components:**  
  Debug messages include shapes of outputs from custom modules like `LayerNorm2d` and `MLP`. In the `MaskDecoderHQ`, layers print shape information when reshaping features and processing hypernetwork tokens.

- **Checkpoint & Model Loading:**  
  When a checkpoint is loaded (for SAM and HQ decoder), a debug message prints the checkpoint path and confirms parameter settings.

- **Training Pipeline:**  
  Status messages for:
  - Data loader setup (number of batches created)
  - Batch input shapes and prompt details
  - Loss computation and optimizer steps
  - Saving of model checkpoints and logging of metrics
- **Evaluation & Inference:**  
  The script prints:
  - Number of batches processed during evaluation
  - Shapes of embeddings, predicted masks, and computed IoU scores
  - File paths where output visualizations are saved

*All debug messages are printed with a `[DEBUG]` prefix for easier filtering. You can modify or remove these debug prints for production if desired.*

---

## PIPELINE SUMMARY

### How Everything Works
- **Data Preparation:**  
  Input images and annotations are organized into structured directories.
- **Pre-processing:**  
  Data augmentation and custom loader modifications facilitate working with challenging datasets (e.g., Renishaw).
- **Model Components:**  
  The encoder and mask decoder (extended from SAM) are combined with additional modules (PerSAM, SAM-HQ) for enhanced segmentation.
- **Training:**  
  Distributed training with detailed system monitoring is implemented. Checkpoints and logs are saved at regular intervals.
- **Inference:**  
  Two variants of the inference scripts support SAMHQ and PERSAM, allowing for both prompt-based and annotation-based segmentation.
- **Outputs:**  
  Results, including overlay visualizations and performance metrics, are stored in specified output directories.

### Diagram

Below is the overall pipeline diagram:
````

                   +---------------------+
                   |   Input Data        |
                   | (Images, Annotations, JSON Prompts) 
                   +----------+----------+
                              |
                              v
                   +---------------------+
                   | Data Preprocessing  |
                   |  - Data Augmentation|
                   |  - Custom Dataloaders|
                   +----------+----------+
                              |
                              v
                   +---------------------+
                   |  Model Components   | <----------------+
                   |  (Encoder &         |                  |
                   |  Mask Decoder:      |                  |
                   |  SAM, PerSAM, SAM-HQ)|                 |
                   +----------+----------+                  |
                              |                             |
                              v                             |
                   +-------------------------+              |
                   |  Training Pipeline      |              |
                   |  - Distributed DDP      |              |
                   |  - Debug Monitoring     |              |
                   |  - HQ Mask Decoder      | <-- attached only at final epoch
                   +----------+--------------+              |
                              |                            / \
                              |                           /   \
                   +----------+---------+       +--------+---------+
                   | Inference Script 1 |       | Inference Script 2 |
                   |      (SAMHQ)       |       |      (PERSAM)      |
                   +--------------------+       +--------------------+
                                                      |
                                                      v
                                             +-------------------+
                                             |  Output Results   |
                                             | (Visualizations,  |
                                             |  Metrics, Logs)   |
                                             +-------------------+
````
---

## WORK DONE TILL NOW

- **Literature Review:**  
  - PerSAM, AM-HQ, MobileSAM, Swin, Classification via Segmentation, Unet  
  - Speed improvements for neural networks on CPUs and optimized transformer inference
  - Sparse edge-processing for deep network training acceleration

- **Pre-processing Steps:**  
  - Data augmentation; converted skewered .TIF files into binary masks  
  - Custom data loader for the Renishaw dataset maintaining viability

- **Model Dry Runs:**  
  - Models validated for the Renishaw dataset (SAM-HQ, PerSAM/PerSAM-F, MobileSAM)

- **Model Training Experiments:**  
  - Experiments for both from-scratch and frozen module training  
  - Memory analysis per module/layer for PerSAM, SAM-HQ  
  - Memory constraint experiments comparing Swin Transformer with ViT using a dummy CNN

- **Model Crafting:**  
  - Augmented Encoder: Integrated PerSAM and SAM-HQ  
  - Augmented Mask Decoder: Merged features from both models  
  - Combined target-guided attention and target-semantic prompting from SAM-HQ and PerSAM-F

- **Script Development:**  
  - **Training Script:** Developed a new training script to support both from-scratch and frozen module experiments.
  - **Inference Script:** Developed a new inference script for both training setups.

---

## LINKS & RESOURCES




---

Feel free to contribute or open issues if you encounter any problems. Enjoy experimenting with Personalize SAM HQ!
