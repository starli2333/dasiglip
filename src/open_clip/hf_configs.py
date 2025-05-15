# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union # Added List, Union

# Minimal dataclass definitions to satisfy imports in model.py
# These should ideally mirror the structure expected by VisionTransformer and TextTransformer
# or the actual CLIPVisionCfg and CLIPTextCfg from open_clip.model if they were moved.

@dataclass
class OpenCLIPViTConfig:
    # Common ViT parameters - ensure these match what VisionTransformer expects
    # Or, more likely, these are the same as CLIPVisionCfg in model.py
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: Optional[int] = None # Often width // heads
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.
    input_patchnorm: bool = False
    global_average_pool: bool = False
    output_tokens: bool = False # If true, output is sequence of tokens, not pooled
    # output_dim: int = 512 # This is usually the final embed_dim, not part of vision_cfg directly
    # act_layer: str = 'GELU' # Example
    # norm_layer: str = 'LayerNorm' # Example
    # Other ViT specific params if needed by your VisionTransformer
    timm_model_name: Optional[str] = None  # a valid timm model name
    timm_model_pretrained: bool = False  # use timm model pretrained weights
    timm_pool: str = 'avg'  # feature pooling for timm model
    timm_proj: str = 'linear'  # linear projection for timm model output
    timm_proj_bias: bool = False
    timm_drop: float = 0.  # dropout prob for timm model
    timm_attn_drop: float = 0.  # attn dropout prob for timm model

    def __post_init__(self):
        if self.head_width is None:
            # Assuming num_attention_heads would be width // 64 (common default head dim)
            # This is a placeholder, as heads are usually part of vision_cfg in OpenCLIP
            num_heads = self.width // 64
            self.head_width = self.width // num_heads if num_heads > 0 else self.width


@dataclass
class OpenCLIPTextConfig:
    # Common TextTransformer parameters - ensure these match
    # Or, more likely, these are the same as CLIPTextCfg in model.py
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    # output_dim: int = 512 # This is usually the final embed_dim
    # act_layer: str = 'GELU'
    # norm_layer: str = 'LayerNorm'
    # HF specific if wrapping HF models (though usually done in CustomTextCLIP)
    hf_model_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    hf_model_pretrained: bool = True
    proj_bias: bool = False # For text projection layer
    pool_type: str = 'argmax' # Or 'mean', 'cls', etc.

# This file might also contain mappings of OpenCLIP model names to HF Hub identifiers
# For example:
# HF_OPENCLIP_MODEL_MAP = {
#     "ViT-B-32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
#     # ... other mappings
# }

# If this file is *only* for the above dataclasses, the rest can be empty.
# If it's also for HF model name mappings (as OpenCLIP's hf_configs.py often is),
# those mappings would go here. For now, focusing on fixing the import error.

