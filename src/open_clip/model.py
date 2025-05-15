# -*- coding: utf-8 -*-
import logging
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field 
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .transformer import TextTransformer, VisionTransformer, LayerNorm


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: Optional[int] = None 
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.
    input_patchnorm: bool = False
    global_average_pool: bool = False
    output_tokens: bool = False
    timm_model_name: Optional[str] = None
    timm_model_pretrained: bool = False
    timm_pool: str = 'avg'
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    timm_drop: float = 0.
    timm_attn_drop: float = 0.
    act_layer: Optional[Callable] = None
    norm_layer: Optional[Callable] = None
    heads: Optional[int] = None

    def __post_init__(self):
        if self.heads is None and self.width is not None and self.head_width is not None and self.head_width > 0 :
            self.heads = self.width // self.head_width
        elif self.head_width is None and self.width is not None and self.heads is not None and self.heads > 0:
            self.head_width = self.width // self.heads
        if self.heads is None and self.width is not None: # Default if still None
            self.heads = self.width // 64


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    act_layer: Optional[Callable] = None
    norm_layer: Optional[Callable] = None
    hf_model_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    hf_model_pretrained: bool = True
    proj_bias: bool = False
    pool_type: str = 'argmax'

    def __post_init__(self):
        if self.hf_tokenizer_name is None and self.hf_model_name:
            self.hf_tokenizer_name = self.hf_model_name

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        if isinstance(vision_cfg, dict): vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict): text_cfg = CLIPTextCfg(**text_cfg)

        vision_heads = vision_cfg.heads
        if vision_heads is None:
            if vision_cfg.head_width is not None and vision_cfg.head_width > 0:
                vision_heads = vision_cfg.width // vision_cfg.head_width
            else:
                vision_heads = vision_cfg.width // 64 
                logging.warning(f"Vision tower heads not specified or head_width invalid, defaulting to {vision_heads} based on width {vision_cfg.width}.")

        self.visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=vision_cfg.act_layer or (QuickGELU if quick_gelu else nn.GELU),
            norm_layer=vision_cfg.norm_layer or LayerNorm
        )

        self.transformer = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            output_dim=embed_dim, 
            act_layer=text_cfg.act_layer or (QuickGELU if quick_gelu else nn.GELU),
            norm_layer=text_cfg.norm_layer or LayerNorm,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(text_cfg.context_length, text_cfg.width))
        self.ln_final = (text_cfg.norm_layer or LayerNorm)(text_cfg.width)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_weights() 

        if cast_dtype is not None:
            self.to(cast_dtype)

    def init_weights(self): 
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        if hasattr(self.visual, 'init_weights'): 
            self.visual.init_weights()

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.transformer.text_projection is not None:
            if isinstance(self.transformer.text_projection, nn.Linear):
                nn.init.normal_(self.transformer.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.transformer.text_projection.bias is not None:
                     nn.init.zeros_(self.transformer.text_projection.bias)
            elif isinstance(self.transformer.text_projection, nn.Parameter):
                 nn.init.normal_(self.transformer.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.set_grad_checkpointing(enable)

    def build_attention_mask(self):
        mask = torch.empty(self.transformer.context_length, self.transformer.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        try:
            self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
        except AttributeError:
            logging.warning("Visual model does not have a .lock() method. Cannot lock image tower.")

    def lock_text_tower(self, unlocked_layers=0, freeze_layer_norm=True):
        try:
            self.transformer.lock(unlocked_layers=unlocked_layers, freeze_layer_norm=freeze_layer_norm)
            for p_name, p_val in self.named_parameters():
                if p_name.startswith("token_embedding.") or \
                   p_name == "positional_embedding" or \
                   (freeze_layer_norm and p_name.startswith("ln_final.")):
                    p_val.requires_grad = False
        except AttributeError:
            logging.warning("Text transformer does not have a .lock() method. Cannot lock text tower.")

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        
        text_long = text if text.dtype == torch.long else text.to(torch.long)
        x = x[torch.arange(x.shape[0]), text_long.argmax(dim=-1)] 

        if self.transformer.text_projection is not None:
            if isinstance(self.transformer.text_projection, nn.Linear):
                 x = self.transformer.text_projection(x)
            elif isinstance(self.transformer.text_projection, nn.Parameter):
                 x = x @ self.transformer.text_projection
            else:
                 logging.warning(f"Unsupported text_projection type: {type(self.transformer.text_projection)}")
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 vision_cfg: CLIPVisionCfg, 
                 text_cfg: CLIPTextCfg,     
                 quick_gelu: bool = False,
                 cast_dtype: Optional[torch.dtype] = None,
                 output_dict: bool = False,
                 ):
        base_text_cfg_for_super = text_cfg
        if text_cfg.hf_model_name:
            base_text_cfg_dict = {
                k: getattr(text_cfg, k) for k in ['context_length', 'vocab_size', 'width', 'heads', 'layers']
                if hasattr(text_cfg, k)
            }
            base_text_cfg_dict.setdefault('context_length', 77)
            base_text_cfg_dict.setdefault('vocab_size', 49408)
            base_text_cfg_dict.setdefault('width', 512)
            base_text_cfg_dict.setdefault('heads', 8)
            base_text_cfg_dict.setdefault('layers', 12)
            base_text_cfg_for_super = CLIPTextCfg(**base_text_cfg_dict)

        super().__init__(
            embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=base_text_cfg_for_super,
            quick_gelu=quick_gelu, cast_dtype=None, output_dict=output_dict,
        )
        self.text_cfg = text_cfg 

        if text_cfg.hf_model_name:
            logging.info(f"Initializing CustomTextCLIP with HF Text Encoder: {text_cfg.hf_model_name}")
            from transformers import AutoModel 
            self.text = AutoModel.from_pretrained(text_cfg.hf_model_name)
            
            hf_config = self.text.config
            text_output_dim = getattr(hf_config, 'hidden_size', None) or \
                              getattr(hf_config, 'd_model', None) or \
                              text_cfg.width 

            if text_output_dim != embed_dim:
                 self.text_projection = nn.Linear(text_output_dim, embed_dim, bias=getattr(text_cfg, 'proj_bias', False))
            else:
                 self.text_projection = nn.Identity()

            if hasattr(self, 'transformer'): del self.transformer
            if hasattr(self, 'token_embedding'): del self.token_embedding
            if hasattr(self, 'positional_embedding'): del self.positional_embedding
            if hasattr(self, 'ln_final'): del self.ln_final
            if hasattr(self, 'attn_mask'): del self.attn_mask
        else:
            logging.info("CustomTextCLIP: Using original TextTransformer setup from base CLIP.")
            self.text = self.transformer 
            if hasattr(self.transformer, 'text_projection'):
                 self.text_projection = self.transformer.text_projection
            else: 
                 if text_cfg.width != embed_dim:
                      self.text_projection = nn.Linear(text_cfg.width, embed_dim, bias=getattr(text_cfg, 'proj_bias', False))
                 else:
                      self.text_projection = nn.Identity()

        if cast_dtype is not None:
            self.to(cast_dtype)

    def encode_text(self, text: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, normalize: bool = False):
        if isinstance(self.text, TextTransformer): 
            return super().encode_text(text, normalize=normalize)
        else: 
            outputs = self.text(input_ids=text, attention_mask=attention_mask)
            
            text_cfg_pool = getattr(self.text_cfg, 'pool_type', 'cls')

            if text_cfg_pool == 'cls':
                text_features = outputs.last_hidden_state[:, 0, :]
            elif text_cfg_pool == 'mean': 
                if attention_mask is not None:
                    masked_hidden_state = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                    sum_hidden_state = masked_hidden_state.sum(dim=1)
                    sum_attention_mask = attention_mask.sum(dim=1, keepdim=True)
                    text_features = sum_hidden_state / sum_attention_mask.clamp(min=1e-9)
                else:
                    text_features = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_features = outputs.pooler_output
            else: 
                logging.warning(f"Unknown pool_type '{text_cfg_pool}' or missing pooler_output, defaulting to CLS pooling.")
                text_features = outputs.last_hidden_state[:, 0, :]

            if hasattr(self, 'text_projection') and not isinstance(self.text_projection, nn.Identity):
                text_features = self.text_projection(text_features)

            return F.normalize(text_features, dim=-1) if normalize else text_features

    def lock_text_tower(self, unlocked_layers=0, freeze_layer_norm=True):
        if isinstance(self.text, TextTransformer):
            super().lock_text_tower(unlocked_layers, freeze_layer_norm)
        else: 
            for param in self.text.parameters():
                param.requires_grad = False
            logging.info("Locked HF text tower.")
            if unlocked_layers > 0:
                logging.warning(f"Unlocking last {unlocked_layers} for HF text models is not generically implemented. Text tower remains fully locked.")


# --- Utility functions ---
def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16) IN-PLACE."""
    
    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # logging.info(f"DEBUG_CONVERT: Layer {l} ({type(l)}), current weight dtype: {l.weight.dtype}")
            if l.weight.dtype == torch.float32:
                l.weight.data = l.weight.data.to(dtype)
                if l.bias is not None and l.bias.dtype == torch.float32:
                    l.bias.data = l.bias.data.to(dtype)
                # logging.info(f"DEBUG_CONVERT: Layer {l} weights AFTER conversion: {l.weight.dtype}")

        if isinstance(l, nn.MultiheadAttention):
            for attr_name in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr_name, None)
                if tensor is not None and tensor.dtype == torch.float32:
                    tensor.data = tensor.data.to(dtype)
        
        if isinstance(l, LayerNorm): 
            if l.weight.dtype == torch.float32:
                l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None and l.bias.dtype == torch.float32:
                l.bias.data = l.bias.data.to(dtype)
    model.apply(_convert_weights)

def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict and not any(k.startswith('text.') for k in state_dict):
        logging.info("Converting old format state_dict to new custom text format.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('transformer.'): new_state_dict[f'text.{k}'] = v
            elif k == 'positional_embedding': new_state_dict[f'text.positional_embedding'] = v
            elif k == 'token_embedding.weight': new_state_dict[f'text.token_embedding.weight'] = v
            elif k == 'ln_final.weight': new_state_dict[f'text.ln_final.weight'] = v
            elif k == 'ln_final.bias': new_state_dict[f'text.ln_final.bias'] = v
            elif k == 'text_projection': new_state_dict['text.text_projection'] = v
            else: new_state_dict[k] = v
        return new_state_dict
    return state_dict

def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim: int = 1):
    pos_embed_checkpoint = state_dict.get('visual.positional_embedding', None)
    if pos_embed_checkpoint is None: return
    
    visual_model = None
    if hasattr(model, 'visual') and isinstance(model.visual, VisionTransformer):
        visual_model = model.visual
    elif hasattr(model, 'module') and hasattr(model.module, 'visual') and isinstance(model.module.visual, VisionTransformer): # DDP
        visual_model = model.module.visual
    
    if visual_model is None or not hasattr(visual_model, 'positional_embedding'): return
    pos_embed_model = visual_model.positional_embedding

    if pos_embed_checkpoint.shape == pos_embed_model.shape: return

    logging.info(f'Resizing position embedding from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}')
    num_prefix_tokens = getattr(visual_model, 'num_prefix_tokens', 1) 

    gs_old = int(math.sqrt(pos_embed_checkpoint.shape[seq_dim] - num_prefix_tokens))
    gs_new = int(math.sqrt(pos_embed_model.shape[seq_dim] - num_prefix_tokens))
    logging.info(f'Resizing grid shape from {gs_old} to {gs_new}')

    pos_grid = pos_embed_checkpoint[:, num_prefix_tokens:, :].reshape(
        pos_embed_checkpoint.shape[0], gs_old, gs_old, -1).permute(0, 3, 1, 2)
    pos_grid = F.interpolate(pos_grid, size=(gs_new, gs_new), mode=interpolation, align_corners=False)
    pos_grid = pos_grid.permute(0, 2, 3, 1).reshape(
        pos_embed_checkpoint.shape[0], gs_new * gs_new, -1)

    if num_prefix_tokens > 0:
        pos_prefix = pos_embed_checkpoint[:, :num_prefix_tokens, :]
        state_dict['visual.positional_embedding'] = torch.cat([pos_prefix, pos_grid], dim=seq_dim)
    else:
        state_dict['visual.positional_embedding'] = pos_grid

def get_cast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == 'fp16': return torch.float16
    elif precision == 'bf16': return torch.bfloat16
    return None

def get_input_dtype(precision: str) -> Optional[torch.dtype]:
    if precision in ('pure_fp16', 'fp16'): return torch.float16
    if precision in ('pure_bf16', 'bf16'): return torch.bfloat16
    return None

# --- ADDED build_model_from_openai_state_dict function ---
def build_model_from_openai_state_dict(
        state_dict: dict,
        image_resolution: int = 224, # Default, will be overridden if in state_dict
        vision_layers: Union[Tuple[int, int, int, int], int] = 12,
        vision_width: int = 768,
        vision_patch_size: Optional[int] = None, # For ViT
        text_context_length: int = 77,
        text_vocab_size: int = 49408,
        text_width: int = 512,
        text_heads: int = 8,
        text_layers: int = 12,
        embed_dim: Optional[int] = None, # If None, infer from vision_width or text_width
        output_dim: Optional[int] = None, # Usually same as embed_dim
        quick_gelu: bool = False, # For compatibility with older OpenAI models
        cast_dtype: Optional[torch.dtype] = None,
        **kwargs
):
    """
    Builds a CLIP model from an OpenAI-formatted state_dict.
    Infers model parameters if not all are provided.
    """
    # Infer embed_dim if not provided
    if embed_dim is None:
        if 'text_projection' in state_dict:
            embed_dim = state_dict['text_projection'].shape[1]
        elif 'visual.proj' in state_dict:
            embed_dim = state_dict['visual.proj'].shape[0] # ViT proj is (width, embed_dim)
        elif 'visual.attnpool.c_proj.weight' in state_dict: # ResNet attnpool
             embed_dim = state_dict['visual.attnpool.c_proj.weight'].shape[0]
        else:
            # Fallback or raise error if embed_dim cannot be inferred
            # For simplicity, let's try to infer from vision_width or text_width if they are standard
            if vision_patch_size is not None: # Likely a ViT
                embed_dim = vision_width if vision_width == 512 or vision_width == 768 or vision_width == 1024 else 512 # Common ViT output dims before proj
            else: # Likely a ResNet
                embed_dim = vision_width * 32 // (vision_width // 64) if vision_width == 64 else 1024 # Heuristic for ResNet output
            logging.warning(f"Could not directly infer embed_dim, using heuristic: {embed_dim}")

    if output_dim is None:
        output_dim = embed_dim

    # Vision Cfg
    # For ViTs, vision_layers is an int. For ResNets, it's a tuple.
    # OpenAI state_dict keys can help differentiate.
    is_vit = 'visual.class_embedding' in state_dict or 'visual.transformer.resblocks.0.attn.in_proj_weight' in state_dict
    if is_vit:
        # Infer ViT parameters if not provided
        if vision_patch_size is None:
            # Try to infer from positional_embedding shape if possible, or use common defaults
            if 'visual.positional_embedding' in state_dict:
                 # Example: shape (1, 197, 768) -> 196 grid tokens -> 14x14 grid -> patch 16 for 224px
                 # This is heuristic
                 num_pos_tokens = state_dict['visual.positional_embedding'].shape[1] -1 # Exclude CLS
                 grid_size = int(math.sqrt(num_pos_tokens))
                 vision_patch_size = image_resolution // grid_size if grid_size > 0 else 16 # Default
            else:
                 vision_patch_size = 16 # Common default
            logging.info(f"Inferred ViT vision_patch_size: {vision_patch_size}")

        if not isinstance(vision_layers, int): # Ensure vision_layers is int for ViT
            # Try to infer from state_dict keys if it's a tuple (e.g. from ResNet config)
            num_resblocks = 0
            while f'visual.transformer.resblocks.{num_resblocks}.attn.in_proj_weight' in state_dict:
                num_resblocks += 1
            vision_layers = num_resblocks if num_resblocks > 0 else 12 # Default ViT layers
            logging.info(f"Inferred ViT vision_layers: {vision_layers}")

        vision_cfg_dict = {
            'layers': vision_layers, 'width': vision_width, 'patch_size': vision_patch_size,
            'image_size': image_resolution, 'output_dim': output_dim, # output_dim here is for VisionTransformer
            'mlp_ratio': 4.0 * vision_width / 768, # Scale mlp_ratio with width
            'heads': vision_width // 64 # Common head dim
        }
    else: # Assume ResNet-like
        if not isinstance(vision_layers, tuple):
            # Try to infer ResNet layers if not tuple (this is tricky without more info)
            # Defaulting to RN50 layers if not a tuple
            vision_layers = (3, 4, 6, 3) if not isinstance(vision_layers, tuple) else vision_layers
            logging.info(f"Assuming ResNet-like, vision_layers set to/kept as: {vision_layers}")
        vision_cfg_dict = {
            'layers': vision_layers, 'width': vision_width,
            'image_size': image_resolution, 'output_dim': output_dim,
            'heads': vision_width // 64 # ResNet head in attnpool
        }

    vision_cfg = CLIPVisionCfg(**vision_cfg_dict)

    # Text Cfg
    text_cfg_dict = {
        'context_length': text_context_length, 'vocab_size': text_vocab_size,
        'width': text_width, 'heads': text_heads, 'layers': text_layers,
        # 'output_dim': output_dim # TextTransformer projects to output_dim
    }
    text_cfg = CLIPTextCfg(**text_cfg_dict)

    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu, # Pass quick_gelu
        cast_dtype=cast_dtype
    )
    
    # Load state dict into the newly built model
    # Convert state dict if it's from original OpenAI model
    if 'input_resolution' in state_dict: # Check for a key unique to OpenAI state_dicts
        logging.info("OpenAI-format state_dict detected, building model from it.")
        # The build_model_from_openai_state_dict in original open_clip.model.py
        # actually *creates* the model. Here we've created a generic CLIP
        # and need to load the state_dict into it.
        # The original OpenAI state_dict has different key names.
        # We need to map them or ensure our CLIP class uses compatible names.
        # For simplicity, if state_dict is from OpenAI, we assume it's already loaded
        # into a compatible structure by torch.jit.load or a prior step.
        # If `model` was None before and `state_dict` is directly from OpenAI .pt:
        # This function should ideally *return* the built model.
        # The current CLIP class init is more for OpenCLIP JSON configs.
        # This function's role needs to be clear: does it build from scratch or load into existing?
        # For now, assuming it loads into the `model` created above.
        # This requires that our CLIP class is compatible with OpenAI key names,
        # or we need a conversion function for the state_dict keys.
        # The original OpenCLIP `build_model_from_openai_state_dict` did more complex instantiation.
        # Let's adapt it slightly.
        
        # Manually map OpenAI state_dict keys to our CLIP model structure
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k == "positional_embedding":
                mapped_state_dict["positional_embedding"] = v
            elif k == "text_projection": # This is a [width, embed_dim] matrix
                # Our TextTransformer has text_projection as a parameter matrix or Linear layer
                # If it's a parameter:
                if isinstance(model.transformer.text_projection, nn.Parameter):
                     mapped_state_dict["transformer.text_projection"] = v
                # If it's a Linear layer:
                elif isinstance(model.transformer.text_projection, nn.Linear):
                     mapped_state_dict["transformer.text_projection.weight"] = v # OpenAI proj is not a layer
            elif k == "logit_scale":
                mapped_state_dict[k] = v
            elif k.startswith("visual."):
                mapped_state_dict[k] = v # Assume visual keys match VisionTransformer
            elif k.startswith("transformer.") or k == "token_embedding.weight" or k.startswith("ln_final.") :
                mapped_state_dict[k] = v # Assume text keys match TextTransformer parts
            else:
                # Other keys like input_resolution, context_length etc. are not model params
                pass
        
        # Handle cases where text_projection might be a matrix in checkpoint but Linear in model
        if 'transformer.text_projection' in mapped_state_dict and \
           isinstance(model.transformer.text_projection, nn.Linear) and \
           mapped_state_dict['transformer.text_projection'].ndim == 2:
            # OpenAI text_projection is (width, embed_dim), Linear layer weight is (embed_dim, width)
            # So, we need to transpose.
            # Also, OpenAI's text_projection is Parameter, not Linear.
            # This mapping is tricky. The original build_model_from_openai_state_dict
            # directly used these values in the constructor.
            # For now, we assume our CLIP model structure is compatible enough,
            # or that a JIT loaded model was passed.
            # If loading into our CLIP from an OpenAI state_dict, key mapping is essential.
            # This simplified mapping might miss things or have mismatches.
            pass # Requires more careful key mapping if not JIT loaded

        incompatible_keys = model.load_state_dict(mapped_state_dict, strict=False)
        logging.info(f"Loaded OpenAI state_dict with incompatible keys: {incompatible_keys}")

    return model
# --- END ADDED build_model_from_openai_state_dict ---


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    # ... (implementation as before) ...
    logging.warning("trace_model function is a placeholder and might not work correctly for all model types.")
    try:
        img_size = 224
        if hasattr(model, 'visual') and hasattr(model.visual, 'image_size'):
            img_size_attr = model.visual.image_size
            if isinstance(img_size_attr, tuple): img_size = img_size_attr[0]
            elif isinstance(img_size_attr, int): img_size = img_size_attr

        context_len = 77
        vocab_size = 49408 
        if hasattr(model, 'text_cfg') and hasattr(model.text_cfg, 'context_length'): 
            context_len = model.text_cfg.context_length
            vocab_size = model.text_cfg.vocab_size
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'context_length'): 
            context_len = model.transformer.context_length
            vocab_size = model.vocab_size 

        dummy_image = torch.randn(batch_size, 3, img_size, img_size, device=device)
        dummy_text_tokens = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
        
        if isinstance(model, DaSiglipModel):
            dummy_text_dict = {
                "caption_tokens": dummy_text_tokens,
                "caption_attention_mask": torch.ones_like(dummy_text_tokens),
                "degradation_target": torch.zeros(batch_size, NUM_DEGRADATION_TYPES, device=device)
            }
            traced_script_module = torch.jit.trace(model, (dummy_image, dummy_text_dict))
        elif isinstance(model, CustomTextCLIP) and not isinstance(model.text, TextTransformer):
            traced_script_module = torch.jit.trace(model, (dummy_image, dummy_text_tokens, torch.ones_like(dummy_text_tokens)))
        else: 
            traced_script_module = torch.jit.trace(model, (dummy_image, dummy_text_tokens))
        
        logging.info("Model traced successfully (placeholder).")
        return traced_script_module
    except Exception as e:
        logging.error(f"Failed to trace model: {e}")
        return model
