# -*- coding: utf-8 -*-
# Core model components
from .model import (
    CLIP, CustomTextCLIP,
    CLIPTextCfg, CLIPVisionCfg, # These are defined in model.py
    convert_weights_to_lp,
    convert_to_custom_text_state_dict, # Ensure this is defined in model.py
    # trace_model, # trace_model was removed from main.py call, ensure it's defined if needed elsewhere
    get_cast_dtype, get_input_dtype,
    resize_pos_embed
)

# DA-SigLIP specific model
from .dasiglip_model import DaSiglipModel

# Other model types if used (like CoCa)
from .coca_model import CoCa

# Factory functions for creating models, tokenizers, losses, and transforms
from .factory import (
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    get_tokenizer,
    create_loss,
    list_models,
    get_model_config as get_config_from_factory # Alias for factory's get_model_config
)

# Loss functions
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, DaSiglipLoss

# Pretrained model utilities
from .pretrained import (
    list_pretrained,
    list_pretrained_tags_by_model,
    is_pretrained_cfg,
    get_pretrained_cfg,
    download_pretrained,
    download_pretrained_from_hf,
    HF_CONFIG_NAME
)

# Tokenizers
from .tokenizer import SimpleTokenizer, tokenize, HFTokenizer

# Image transforms
from .transform import image_transform, AugmentationCfg

# OpenAI specific utilities (if still used)
from .openai import load_openai_model, list_openai_models

# Zero-shot utilities
from .zero_shot_classifier import build_zero_shot_classifier, build_zero_shot_classifier_legacy
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES

# Version
from .version import __version__

__all__ = [
    "CLIP", "CustomTextCLIP", "CLIPTextCfg", "CLIPVisionCfg", "DaSiglipModel", "CoCa",
    "convert_weights_to_lp", "convert_to_custom_text_state_dict",
    # "trace_model", # Only include if defined and used
    "get_cast_dtype", "get_input_dtype", "resize_pos_embed",
    "create_model", "create_model_and_transforms", "create_model_from_pretrained",
    "get_tokenizer", "create_loss", "list_models", "get_config_from_factory",
    "ClipLoss", "DistillClipLoss", "CoCaLoss", "DaSiglipLoss",
    "list_pretrained", "list_pretrained_tags_by_model", "is_pretrained_cfg",
    "get_pretrained_cfg", "download_pretrained", "download_pretrained_from_hf", "HF_CONFIG_NAME",
    "SimpleTokenizer", "tokenize", "HFTokenizer",
    "image_transform", "AugmentationCfg",
    "load_openai_model", "list_openai_models",
    "build_zero_shot_classifier", "build_zero_shot_classifier_legacy",
    "OPENAI_IMAGENET_TEMPLATES", "SIMPLE_IMAGENET_TEMPLATES", "IMAGENET_CLASSNAMES",
    "__version__"
]
