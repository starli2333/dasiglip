# -*- coding: utf-8 -*-
import hashlib
import os
import urllib
import warnings
import logging # Added logging
from typing import List, Optional, Union, Dict, Any
from packaging import version # For version comparison if needed

import torch
from tqdm import tqdm

# --- Define HF_CONFIG_NAME at the top level ---
HF_CONFIG_NAME = "config.json"  # Standard Hugging Face model config filename
OPENAI_CLIP_CONFIG_NAME = "config.yaml" # OpenAI CLIPs might use this
# --- End Define ---


# Default cache directory for OpenCLIP
# Note: This might be overridden by environment variables or arguments in main scripts
try:
    # Try to use a standard cache directory
    _default_home = os.path.expanduser("~")
    CACHE_DIR_OPENCLIP = os.environ.get('OPENCLIP_CACHEDIR', os.path.join(_default_home, '.cache', 'openclip'))
except Exception: # Fallback if home directory is not resolvable
    CACHE_DIR_OPENCLIP = os.path.join(os.getcwd(), '.cache', 'openclip')


# This dictionary is usually populated by model-specific pretrained configurations.
# For DA-SigLIP, we primarily rely on Hugging Face for the base model,
# and local checkpoints for the controller.
PRETRAINED: Dict[str, Dict[str, Any]] = {
    # Example (structure can vary):
    # "RN50": {
    #     "openai": {
    #         "url": "...", "filename": "RN50.pt", "sha256": "...",
    #         "hf_hub_id": "openai/clip-vit-base-patch32" # if applicable
    #     }
    # }
}


def _get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Returns the cache directory, defaulting to CACHE_DIR_OPENCLIP."""
    return cache_dir or CACHE_DIR_OPENCLIP


def get_pretrained_url(model_name: str, pretrained: str) -> Optional[str]:
    """Gets the direct download URL for a pretrained model, if defined in PRETRAINED dict."""
    if pretrained not in PRETRAINED.get(model_name, {}):
        return None
    return PRETRAINED[model_name][pretrained].get('url')


def get_pretrained_cfg(model_name: str, pretrained: str) -> Optional[Dict[str, Any]]:
    """Gets the full configuration for a pretrained model, if defined in PRETRAINED dict."""
    if pretrained not in PRETRAINED.get(model_name, {}):
        return None
    return PRETRAINED[model_name][pretrained]


def list_pretrained(model_name: Optional[str] = None) -> List[str]:
    """Lists all model names for which pretrained configs exist, or tags for a specific model."""
    if model_name is None:
        return list(PRETRAINED.keys())
    return list(PRETRAINED.get(model_name, {}).keys())


def list_pretrained_tags_by_model(model_name: str) -> List[str]:
    """Returns a list of available pretrained tags for a given model name."""
    return list(PRETRAINED.get(model_name, {}).keys())


def is_pretrained_cfg(model_name: str, pretrained: str) -> bool:
    """Checks if a pretrained configuration exists for a model and tag."""
    return pretrained in PRETRAINED.get(model_name, {})


def download_pretrained_from_hf(
        model_id: str,
        filename: str = "pytorch_model.bin",
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        use_auth_token: Optional[Union[bool, str]] = None, # Renamed from token for consistency
        revision: Optional[str] = None,
        user_agent: Optional[Union[Dict, str]] = None,
):
    """
    Downloads a pretrained model or config file from Hugging Face Hub.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        logging.error("huggingface_hub is not installed. Please install it with `pip install huggingface-hub`")
        raise

    resolved_cache_dir = _get_cache_dir(cache_dir)
    try:
        downloaded_file = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=resolved_cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=use_auth_token, # hf_hub_download uses 'token'
            revision=revision,
            library_name="open_clip",
            library_version=getattr(version, 'VERSION', 'unknown'), # Assuming version is imported if needed
            user_agent=user_agent,
        )
        return downloaded_file
    except EntryNotFoundError:
        logging.error(f"File '{filename}' not found in Hugging Face Hub repo '{model_id}'.")
        return None
    except Exception as e:
        logging.error(f"Error downloading '{filename}' from HF Hub repo '{model_id}': {e}")
        return None


def download_pretrained(
        cfg: Dict,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
):
    """
    Downloads a pretrained model checkpoint based on its configuration.
    Handles both direct URLs and Hugging Face Hub IDs.
    """
    resolved_cache_dir = _get_cache_dir(cache_dir)
    target_name = cfg.get("filename")

    if not target_name:
        if cfg.get("hf_hub_id"):
            # Default filenames for HF models if not specified
            target_name = "pytorch_model.bin" if not cfg.get("filename", "").endswith(".json") else cfg.get("filename")
        elif cfg.get("url"):
            target_name = os.path.basename(cfg["url"])
        else:
            raise ValueError("Pretrained config must have either 'url' or 'hf_hub_id'.")

    if cfg.get("hf_hub_id"):
        return download_pretrained_from_hf(
            model_id=cfg["hf_hub_id"],
            filename=target_name,
            cache_dir=resolved_cache_dir,
            force_download=force_download,
            revision=cfg.get("hf_revision")
        )
    elif cfg.get("url"):
        url = cfg["url"]
        target_path = os.path.join(resolved_cache_dir, target_name)

        if os.path.exists(target_path) and not force_download:
            logging.debug(f"Pretrained checkpoint '{target_name}' found in cache: {target_path}")
            return target_path

        logging.info(f"Downloading pretrained checkpoint '{target_name}' from {url} to {target_path}")
        os.makedirs(resolved_cache_dir, exist_ok=True)
        try:
            # Using torch.hub.download_url_to_file for progress bar and better handling
            torch.hub.download_url_to_file(url, target_path, hash_prefix=None, progress=True)
            # Verify SHA256 hash if provided
            if "sha256" in cfg:
                logging.debug("Verifying SHA256 hash...")
                with open(target_path, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if digest.lower() != cfg["sha256"].lower(): # Case-insensitive compare
                    os.remove(target_path)
                    raise RuntimeError(f"SHA256 hash mismatch for {target_name} (expected {cfg['sha256']}, got {digest})")
            return target_path
        except Exception as e:
            if os.path.exists(target_path): # Clean up partial download
                os.remove(target_path)
            logging.error(f"Failed to download {target_name} from {url}: {e}")
            raise e
    else:
        raise ValueError("Pretrained config must have either 'url' or 'hf_hub_id'.")

