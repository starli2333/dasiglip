# -*- coding: utf-8 -*-
import hashlib
import os
import urllib
import warnings
from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

# Corrected imports from .pretrained
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_tags_by_model, download_pretrained # Corrected import
from .transform import image_transform


_MODEL_INFO = {
    "RN50": {
        "name": "RN50",
        "url": "https://openaipublic.azureedge.net/clip/models/afeb01138d800569cf686844b7673800809583c57095674090059909053705EA/RN50.pt",
        "sha256": "afeb01138d800569cf686844b7673800809583c57095674090059909053705EA",
        "visual_input_resolution": 224,
        "embed_dim": 1024,
        "image_cfg": {"layers": [3, 4, 6, 3], "width": 64, "output_dim": 1024},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12, "output_dim": 1024}
    },
    "RN101": {
        "name": "RN101",
        "url": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5925025a8e4538c3bdbe8804a470a72f30b0d94affbf1/RN101.pt",
        "sha256": "8fa8567bab74a42d41c5925025a8e4538c3bdbe8804a470a72f30b0d94affbf1",
        "visual_input_resolution": 224,
        "embed_dim": 512,
        "image_cfg": {"layers": [3, 4, 23, 3], "width": 64, "output_dim": 512},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12, "output_dim": 512}
    },
    "RN50x4": {
        "name": "RN50x4",
        "url": "https://openaipublic.azureedge.net/clip/models/7e526bd135e4931c882ea1a051e97577813dd8985dd2b00d42aa70698c452185/RN50x4.pt",
        "sha256": "7e526bd135e4931c882ea1a051e97577813dd8985dd2b00d42aa70698c452185",
        "visual_input_resolution": 288, # Note: OpenAI uses 288 for this model
        "embed_dim": 640,
        "image_cfg": {"layers": [4, 6, 10, 6], "width": 80, "output_dim": 640}, # width=80, heads=10?
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 640, "heads": 10, "layers": 12, "output_dim": 640}
    },
    "RN50x16": {
        "name": "RN50x16",
        "url": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a0648286/RN50x16.pt",
        "sha256": "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a0648286",
        "visual_input_resolution": 384, # Note: OpenAI uses 384
        "embed_dim": 768,
        "image_cfg": {"layers": [6, 8, 18, 8], "width": 96, "output_dim": 768}, # width=96, heads=12?
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12, "output_dim": 768}
    },
    "RN50x64": {
        "name": "RN50x64",
        "url": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "sha256": "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c",
        "visual_input_resolution": 448, # Note: OpenAI uses 448
        "embed_dim": 1024,
        "image_cfg": {"layers": [3, 15, 36, 10], "width": 128, "output_dim": 1024}, # width=128, heads=16?
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 1024, "heads": 16, "layers": 12, "output_dim": 1024}
    },
    "ViT-B/32": {
        "name": "ViT-B/32",
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "visual_input_resolution": 224,
        "embed_dim": 512,
        "image_cfg": {"layers": 12, "width": 768, "patch_size": 32, "output_dim": 512},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12, "output_dim": 512}
    },
    "ViT-B/16": {
        "name": "ViT-B/16",
        "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "visual_input_resolution": 224,
        "embed_dim": 512,
        "image_cfg": {"layers": 12, "width": 768, "patch_size": 16, "output_dim": 512},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12, "output_dim": 512}
    },
    "ViT-L/14": {
        "name": "ViT-L/14",
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7fc7eacadd56cf3208/ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7fc7eacadd56cf3208",
        "visual_input_resolution": 224,
        "embed_dim": 768,
        "image_cfg": {"layers": 24, "width": 1024, "patch_size": 14, "output_dim": 768},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12, "output_dim": 768}
    },
    "ViT-L/14@336px": { # Name used by OpenAI for the 336px version
        "name": "ViT-L/14@336px", # Keep this name for consistency with OpenAI
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208657fc4ea6/ViT-L-14-336px.pt",
        "sha256": "3035c92b350959924f9f00213499208657fc4ea6",
        "visual_input_resolution": 336,
        "embed_dim": 768,
        "image_cfg": {"layers": 24, "width": 1024, "patch_size": 14, "output_dim": 768},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12, "output_dim": 768}
    },
}


def list_openai_models() -> List[str]:
    """Returns the names of the official CLIP models from OpenAI."""
    return list(_MODEL_INFO.keys())


def load_openai_model(
        name: str,
        precision: Optional[str] = None, # 'fp16', 'bf16', 'fp32'
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = False,
        cache_dir: Optional[str] = None,
):
    """Load an official CLIP model from OpenAI.

    Params:
    -------
    name : str
        A model name listed by `open_clip.list_openai_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None determined by loaded state_dict
    device : Optional[Union[str, torch.device]]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default False)
    cache_dir : Optional[str]
        Directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODEL_INFO:
        model_path = None
        # Try to download if not already a path
        if get_pretrained_url(name, 'openai'): # Check if 'openai' tag exists for this model
            model_path = download_pretrained(get_pretrained_cfg(name, 'openai'), cache_dir=cache_dir)
        elif os.path.exists(name): # If name is a path
            model_path = name

        if not model_path:
            raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(model_path, map_location="cpu")
            model = None # Will be built from state_dict
        
        # Build model from state_dict if not loaded from JIT
        if model is None:
            model_info = _MODEL_INFO[name]
            model = build_model_from_openai_state_dict(
                state_dict,
                image_resolution=model_info["visual_input_resolution"],
                vision_layers=model_info["image_cfg"]["layers"],
                vision_width=model_info["image_cfg"]["width"],
                vision_patch_size=model_info["image_cfg"].get("patch_size"), # ViT specific
                text_context_length=model_info["text_cfg"]["context_length"],
                text_vocab_size=model_info["text_cfg"]["vocab_size"],
                text_width=model_info["text_cfg"]["width"],
                text_heads=model_info["text_cfg"]["heads"],
                text_layers=model_info["text_cfg"]["layers"],
                embed_dim=model_info["embed_dim"], # Pass embed_dim
                output_dim=model_info["embed_dim"] # Assuming output_dim is embed_dim for OpenAI models
            )

    else: # name is a path to a JIT model or a state_dict
        if os.path.exists(name):
            model_path = name
            try:
                model = torch.jit.load(model_path, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu")
                model = None # Will be built from state_dict

            if model is None and state_dict is not None:
                logging.warning(f"Attempting to load model from state_dict path {model_path} without config. Model params will be inferred.")
                model = build_model_from_openai_state_dict(state_dict) # Build with inferred params
        else:
            raise FileNotFoundError(f"Model checkpoint or JIT archive not found at {name}")


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Model precision handling
    if precision is None: # Determine from model weights if not specified
        convert_to_float = False
        for p in model.parameters():
            if p.dtype == torch.float16:
                precision = 'fp16'
                break
            elif p.dtype == torch.bfloat16:
                precision = 'bf16'
                break
        else: # No low precision params found
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                precision = 'bf16' # Default to bfloat16 if available
            else:
                precision = 'fp32' # Fallback to fp32
            convert_to_float = True # Ensure model is in a float type if it was int or other
        if convert_to_float and precision != 'fp32': # If model was int and we default to bf16/fp16
            model = model.to(get_cast_dtype(precision))

    if precision == "fp16" or precision == "bf16":
        model = convert_weights_to_lp(model, dtype=get_cast_dtype(precision))
    elif precision == "fp32":
        model = model.float() # Ensure it's float32 if requested

    # The JIT model uses the quick_gelu activation function
    # Ensure this is handled if not using JIT
    if not jit and hasattr(model, 'quick_gelu'):
        model.apply(lambda m: m.to(torch.float32) if isinstance(m, nn.LayerNorm) else None) # Keep LNs in fp32 for stability with OpenAI models
        if name in _MODEL_INFO and "ViT" in name: # For ViT models, GELU is standard
            if hasattr(torch.nn.functional, 'gelu'): # Modern PyTorch
                 model.apply(lambda m: setattr(m, 'act_layer', nn.GELU()) if hasattr(m, 'act_layer') and isinstance(m.act_layer, type(lambda x: x * torch.sigmoid(1.702 * x))) else None)
            else: # Older PyTorch might not have GELU directly
                 logging.warning("torch.nn.functional.gelu not found, ViT activation might remain QuickGELU.")


    if jit:
        model = torch.jit.script(model) # Re-script if loaded from state_dict and jit=True

    # Return model and the standard OpenAI CLIP image transform
    input_resolution = 224 # Default
    if name in _MODEL_INFO:
        input_resolution = _MODEL_INFO[name]["visual_input_resolution"]
    elif hasattr(model, 'visual') and hasattr(model.visual, 'input_resolution'): # If loaded from state_dict and has attr
        input_resolution = model.visual.input_resolution


    preprocess = image_transform(
        input_resolution,
        is_train=False, # OpenAI transform is for inference
        mean=None, # Uses OpenAI defaults internally
        std=None
    )

    return model, preprocess
