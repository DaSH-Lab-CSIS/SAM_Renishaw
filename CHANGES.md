# Changes Documentation: Modifications for PerSAM-HQ Integration

This document details the specific code changes made to integrate PerSAM's personalization capabilities with SAM-HQ's high-quality segmentation into a unified PerSAM-HQ framework. All modifications were made within the per_segment_anything directory.

## 1. Image Encoder (`image_encoder.py`)
### Change 1: Modified Position Embedding Handling


```python
# Modified code in PerSAM-HQ
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
 # ...existing params...
        use_abs_pos: bool = False,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
 # ...existing params.
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        # ...existing initialization code...
```

**Purpose**: Force the use of relative positional embeddings in the ViT encoder.

**Why**: The original SAM-HQ implementation used absolute positional embeddings only (use_rel_pos=False), while PerSAM relied on relative positional embeddings. Setting this parameter to True ensures compatibility with the rest of the PerSAM architecture and avoids conflicts in the attention mechanisms when integrating the two frameworks.

### Change 2: Modified Return Values to Include Intermediate Embeddings

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Original SAM-HQ return
    # return self.neck(features)
    
    # Modified return for PerSAM-HQ to include intermediate embeddings
    features = self.backbone(x)
    interm_embeddings = features[0]  # Extract intermediate embeddings from early layers
    embeddings = self.neck(features)
    return embeddings, interm_embeddings
```

**Purpose**: Extract and return intermediate feature embeddings from early layers of the ViT backbone.

**Why**: These intermediate embeddings are crucial for SAM-HQ's high-quality mask generation, as they contain more detailed spatial information that helps refine mask boundaries.

## 2. Predictor Module (`predictor.py`)

### Change 1: Extended `set_image()` to Accept Reference Masks
```python
def set_image(
    self,
    image: np.ndarray,
    mask: np.ndarray = None,  # Added parameter
    image_format: str = "RGB",
    cal_image=True            # Added parameter
) -> None:
    # ...existing code...
    
    # Transform the mask to the form expected by the model
    input_mask_torch = None
    if mask is not None:
      input_mask = self.transform.apply_image(mask)
      input_mask_torch = torch.as_tensor(input_mask, device=self.device)
      input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = self.set_torch_image(input_image_torch, image.shape[:2], transformed_mask=input_mask_torch)
    return input_mask
```

**Purpose**: Enable the model to accept reference masks for personalized segmentation.

**Why**: PerSAM requires a reference mask for the personalization process, which wasn't part of the original SAM-HQ implementation.

### Change 2: Modified `set_torch_image()` to Handle Reference Masks
```python
@torch.no_grad()
def set_torch_image(
    self,
    transformed_image: torch.Tensor,
    original_image_size: Tuple[int, ...],
    transformed_mask: torch.Tensor = None,  # Added parameter
    cal_image=True                          # Added parameter
) -> None:
    # ...existing code...
    
    if cal_image:
      self.reset_image()
      self.original_size = original_image_size
      self.input_size = tuple(transformed_image.shape[-2:])
      input_image = self.model.preprocess(transformed_image)
      self.features, self.interm_features = self.model.image_encoder(input_image)
      self.is_image_set = True

    if transformed_mask is not None:
      input_mask = self.model.preprocess(transformed_mask)  # pad to 1024
      return input_mask
```

**Purpose**: Enable conditional processing of image features based on the presence of a reference mask.

**Why**: Added flexibility to either compute new image embeddings or reuse existing ones, which is essential for efficient personalized segmentation.

### Change 3: Extended Prediction Methods to Support Attention Similarity and Target Embedding
```python
def predict(
    self,
    # ...existing parameters...
    attn_sim = None,           # Added parameter
    target_embedding = None,   # Added parameter
    hq_token_only:bool =False, # Added parameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ...existing code...
    
    masks, iou_predictions, low_res_masks, high_res_masks = self.predict_torch(
        # ...existing parameters...
        attn_sim=attn_sim,
        target_embedding=target_embedding,
        hq_token_only=hq_token_only,
    )
    
    # ...existing code...

@torch.no_grad()
def predict_torch(
    self,
    # ...existing parameters...
    attn_sim = None,           # Added parameter
    target_embedding = None,   # Added parameter
    hq_token_only: bool =False, # Added parameter
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ...existing code...
    
    # Predict masks
    low_res_masks, iou_predictions = self.model.mask_decoder(
        # ...existing parameters...
        attn_sim=attn_sim,
        target_embedding=target_embedding,
        hq_token_only=hq_token_only,
        interm_embeddings=self.interm_features,
    )
    
    # ...existing code...
```

**Purpose**: Enable the personalization mechanisms from PerSAM (target-guided attention and semantic prompting) in the prediction pipeline.

**Why**: These parameters allow the predictor to incorporate target similarity information and embedding guidance during mask generation, which are core to PerSAM's personalization approach.

## 3. Transformer Module (`transformer.py`)

### Change 1: Modified `TwoWayTransformer.forward()` to Support Target Embedding
```python
def forward(
    self,
    image_embedding: Tensor,
    image_pe: Tensor,
    point_embedding: Tensor,
    attn_sim: Tensor=None,         # Added parameter
    target_embedding=None          # Added parameter
) -> Tuple[Tensor, Tensor]:
    # ...existing code...
    
    # Apply transformer blocks and final layernorm
    for layer in self.layers:
        if target_embedding is not None:
            queries += target_embedding
        queries, keys = layer(
            queries=queries,
            keys=keys,
            query_pe=point_embedding,
            key_pe=image_pe,
            attn_sim=attn_sim,
        )
    
    # Apply the final attention layer from the points to the image
    q = queries + point_embedding
    k = keys + image_pe

    if target_embedding is not None:
        q += target_embedding
    # ...existing code...
```

**Purpose**: Incorporate target embedding information at multiple stages of the transformer processing.

**Why**: This enables the semantic prompting aspect of PerSAM by injecting the target embedding information before each transformer layer and final attention computation.

### Change 2: Modified `TwoWayAttentionBlock.forward()` to Accept Attention Similarity
```python
def forward(
    self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, attn_sim: Tensor
) -> Tuple[Tensor, Tensor]:
    # ...existing code...
    
    # Cross attention block, tokens attending to image embedding
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, attn_sim=attn_sim)
    # ...existing code...
```

**Purpose**: Pass the attention similarity information to the cross-attention mechanism.

**Why**: This enables the target-guided attention aspect of PerSAM by influencing the cross-attention maps based on similarity to the target object.

### Change 3: Extended `Attention.forward()` to Modify Attention Maps
```python
def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_sim: Tensor = None) -> Tensor:
    # ...existing code...
    
    # Attention
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)

    if attn_sim is not None:
        attn = attn + attn_sim
        attn = torch.softmax(attn, dim=-1)
    
    # ...existing code...
```

**Purpose**: Modify attention maps based on the provided attention similarity.

**Why**: This is the core mechanism of target-guided attention in PerSAM, where the attention weights are adjusted based on similarity to the target object.

## 4. SAM Model (`sam.py`)

### Change: Extended `forward()` to Handle Intermediate Embeddings
```python
def forward(
    self,
    batched_input: List[Dict[str, Any]],
    multimask_output: bool,
    hq_token_only: bool =False,  # Added parameter
) -> List[Dict[str, torch.Tensor]]:
    # ...existing code...
    
    input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings, interm_embeddings_2 = self.image_encoder(input_images)
    interm_embeddings = interm_embeddings_2[0] # early layer
    
    # ...existing code...
    
    low_res_masks, iou_predictions = self.mask_decoder(
        # ...existing parameters...
        hq_token_only=hq_token_only,
        interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0),
    )
    
    # ...existing code...
    
    outputs.append(
        {
            # ...existing keys...
            "encoder_embedding": curr_embedding.unsqueeze(0),
            "image_pe": self.prompt_encoder.get_dense_pe(),
            "sparse_embeddings":sparse_embeddings,
            "dense_embeddings":dense_embeddings, 
        }
    )
    return outputs, interm_embeddings_2
```

**Purpose**: Extract and pass intermediate embeddings from the image encoder to the mask decoder, and return additional information needed for personalization.

**Why**: SAM-HQ relies on intermediate embeddings for high-quality mask generation, while PerSAM needs the additional embedding information for transfer and personalization between images.

## 5. Mask Decoder (`mask_decoder.py`)

### Change: Extended Forward Methods to Support Personalization and High-Quality Features
```python
def forward(
    self,
    # ...existing parameters...
    attn_sim=None,             # Added parameter
    target_embedding=None,     # Added parameter
    hq_token_only: bool=False, # Added parameter
    interm_embeddings: torch.Tensor=None, # Added parameter
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ...existing code...
    masks, iou_pred = self.predict_masks(
        # ...existing parameters...
        attn_sim=attn_sim,
        target_embedding=target_embedding
    )
    # ...existing code...

def predict_masks(
    self,
    # ...existing parameters...
    attn_sim=None,           # Added parameter
    target_embedding=None    # Added parameter
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ...existing code...
    
    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens, attn_sim, target_embedding)
    # ...existing code...
```

**Purpose**: Enable the mask decoder to leverage both the personalization information from PerSAM and the high-quality features from SAM-HQ.

**Why**: This creates the bridge between personalization (through attention similarity and target embeddings) and high-quality segmentation (through intermediate embeddings), enabling the unified functionality of PerSAM-HQ.

## 6. High-Quality Mask Decoder (`mask_decoder_hq.py`)

### Change: Made Compatible with PerSAM
```python
# Modified to be compatible with perSAM
```

**Purpose**: Ensure the SAM-HQ specific mask decoder can work within the personalized segmentation framework.

**Why**: The original mask_decoder_hq.py was designed only for SAM-HQ and needed modifications to support the additional parameters and processing needed for personalization.

## 7. Training Script (samhq_script.py)

### Change: Enhanced Logging and Checkpointing


**Purpose**: Improve the training workflow with better progress tracking and model saving.

**Why**: Training a complex model like PerSAM-HQ requires comprehensive logging to monitor the training process and regular checkpointing to save progress.


## 8. Inference Path Integration

**Changes implemented:**
- Created two inference paths:
  1. **SAM-HQ Style**: Point/box prompts with high-quality output; samhq_script.py
  2. **PerSAM Style**: Reference-guided segmentation with personalization; inference_ren.py
- Integrated PerSAM-F's efficient post-processing for faster inference in both modes


## Architectural Integration Summary

The code modifications achieve two primary technical goals:

1. **Enable Dual Information Flow**:
   - Personalization information (attention similarity, target embeddings) flows from PerSAM's reference-based mechanisms to guide the segmentation process
   - High-quality information (intermediate embeddings) flows from SAM-HQ's enhanced image encoder to improve mask boundary precision

2. **Create Unified Processing Pipeline**:
   - The predictor module can now accept reference masks for personalization and provide high-quality outputs
   - The transformer and attention mechanisms incorporate both personalization guidance and high-quality feature processing
   - The mask decoder integrates both personalization cues and high-quality token information

These changes allow PerSAM-HQ to perform personalized, high-quality segmentation in a single, unified framework while maintaining compatibility with the original codebases for evaluation and comparison.

Similar code found with 1 license type
