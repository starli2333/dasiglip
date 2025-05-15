# -*- coding: utf-8 -*-
import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoTokenizer, AutoProcessor, SiglipVisionModel, SiglipTextModel
import datetime

# Import from current package
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, NUM_DEGRADATION_TYPES
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, LayerNorm 
from .coca_model import CoCa
from .dasiglip_model import DaSiglipModel 
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, DaSiglipLoss
from .openai import load_openai_model, _MODEL_INFO
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf, HF_CONFIG_NAME
from .tokenizer import HFTokenizer, tokenize

HF_HUB_PREFIX = 'hf-hub:'
DASIGLIP_HF_PREFIX = 'dasiglip_'

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS
    _MODEL_CONFIGS = {}
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            try:
                model_cfg = json.load(f)
                if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                    _MODEL_CONFIGS[cf.stem] = model_cfg
            except json.JSONDecodeError as e:
                 logging.warning(f"Skipping invalid JSON config file: {cf}. Error: {e}")
            except Exception as e:
                 logging.warning(f"Error loading config file {cf}: {e}")
    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()

def list_models():
    return list(_MODEL_CONFIGS.keys())

def add_model_config(path):
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    if model_name in _MODEL_INFO: 
        return deepcopy(_MODEL_INFO[model_name])
    logging.warning(f"No JSON or OpenAI config found for {model_name}. Config will be loaded from Hugging Face if applicable.")
    return None

def get_tokenizer(model_name):
    if model_name.startswith(DASIGLIP_HF_PREFIX):
        hf_model_name = model_name[len(DASIGLIP_HF_PREFIX):]
        logging.info(f"Loading SigLIP tokenizer for: {hf_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            return tokenizer
        except Exception as e:
            logging.error(f"Failed to load SigLIP tokenizer '{hf_model_name}' from Hugging Face. Error: {e}")
            raise e
    elif model_name.startswith(HF_HUB_PREFIX):
        hf_model_name = model_name[len(HF_HUB_PREFIX):]
        logging.info(f"Loading HF tokenizer for: {hf_model_name}")
        return HFTokenizer(hf_model_name)
    else:
        config = get_model_config(model_name)
        if config and 'text_cfg' in config and 'hf_tokenizer_name' in config['text_cfg']:
            logging.info(f"Loading configured HF tokenizer: {config['text_cfg']['hf_tokenizer_name']}")
            return HFTokenizer(config['text_cfg']['hf_tokenizer_name'])
        else:
            logging.info(f"Using default SimpleTokenizer (OpenAI BPE) for model: {model_name}")
            return tokenize

def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)

    incompatible_keys_info = None # 初始化一个变量来存储加载结果 (None 或 _IncompatibleKeys)
    final_strict_value = strict   # 初始化最终使用的 strict 值

    if isinstance(model, DaSiglipModel):
        logging.info("Loading checkpoint into DaSiglipModel.")
        controller_state_dict = {}
        other_state_dict = {} 

        has_prefixed_keys = any(k.startswith('visual_control.') for k in state_dict.keys())

        for k, v in state_dict.items():
            if has_prefixed_keys:
                if k.startswith('visual_control.'):
                    controller_key = k[len('visual_control.'):]
                    controller_state_dict[controller_key] = v
                elif k == 'logit_scale':
                    other_state_dict[k] = v
            else: 
                if k == 'logit_scale':
                    other_state_dict[k] = v
                elif not k.startswith('siglip_visual.') and not k.startswith('siglip_text.'):
                    controller_state_dict[k] = v

        logging.info(f"Attempting to load {len(controller_state_dict)} keys into visual_control.")
        # 确保使用传递的 strict 参数，或者根据评估需要设置为 False
        current_strict_setting = strict
        final_strict_value = current_strict_setting
        logging.info(f"Using strict={current_strict_setting} for loading visual_control state_dict.")
        
        try:
            incompatible_keys_vc = model.visual_control.load_state_dict(controller_state_dict, strict=current_strict_setting)
            incompatible_keys_info = incompatible_keys_vc # 将结果存入通用变量

            # --- 正确处理 incompatible_keys_vc (来自之前的修复) ---
            if current_strict_setting is False and incompatible_keys_vc is not None: # 检查是否为 _IncompatibleKeys 对象
                 if hasattr(incompatible_keys_vc, 'missing_keys') and incompatible_keys_vc.missing_keys:
                     logging.warning(f"Missing keys when loading visual_control state_dict: {incompatible_keys_vc.missing_keys}")
                 if hasattr(incompatible_keys_vc, 'unexpected_keys') and incompatible_keys_vc.unexpected_keys:
                     logging.warning(f"Unexpected keys when loading visual_control state_dict: {incompatible_keys_vc.unexpected_keys}")
            # --- 结束处理 ---
        except Exception as e:
            logging.error(f"Error during visual_control.load_state_dict: {e}")

        # --- 正确加载 logit_scale (它是 Tensor, 不是 Module) ---
        if 'logit_scale' in state_dict:
             # 直接复制数据
             logit_scale_value_from_ckpt = state_dict['logit_scale']
             # 确保类型匹配或进行转换
             model.logit_scale.data.copy_(logit_scale_value_from_ckpt.to(model.logit_scale.dtype))
             logging.info("Loaded logit_scale from checkpoint.")
        elif 'logit_scale' in logit_scale_state_dict: # 如果之前分离到这里
             logit_scale_value_from_ckpt = logit_scale_state_dict['logit_scale']
             model.logit_scale.data.copy_(logit_scale_value_from_ckpt.to(model.logit_scale.dtype))
             logging.info("Loaded logit_scale from checkpoint.")

        if 'logit_scale' in other_state_dict and hasattr(model, 'logit_scale') and isinstance(model.logit_scale, nn.Parameter):
            model.logit_scale.data.copy_(other_state_dict['logit_scale'])
            logging.info("Loaded logit_scale from checkpoint.")
            if incompatible_keys_info is not None and hasattr(incompatible_keys_info, 'missing_keys'):
                if 'logit_scale' in incompatible_keys_info.missing_keys:
                      incompatible_keys_info.missing_keys.remove('logit_scale')
            if incompatible_keys_info is not None and hasattr(incompatible_keys_info, 'unexpected_keys'):
                if 'logit_scale' in incompatible_keys_info.unexpected_keys:
                       incompatible_keys_info.unexpected_keys.remove('logit_scale')
        elif hasattr(model, 'logit_scale') and 'logit_scale' not in other_state_dict :
            logging.warning("logit_scale not found in checkpoint, but present in model.")
            # 如果 incompatible_keys_info 是 None (例如 strict=True 且 visual_control 加载无错误)
            # 或者它已经是 _IncompatibleKeys 对象
            if incompatible_keys_info is None:
                  # 如果之前没有不匹配，现在 logit_scale 缺失了，需要创建一个 _IncompatibleKeys 对象或类似结构
                  # 为了简化，这里可以只记录一个 warning，或者如果需要更新 missing_keys，
                  # 你可能需要确保 incompatible_keys_info 是一个可修改的列表或字典。
                  # PyTorch 的 _IncompatibleKeys 对象的 missing_keys/unexpected_keys 是列表，可以直接 append/remove
                  # 但如果 incompatible_keys_info 为 None，直接 .missing_keys 会报错
                  # 所以我们需要确保它是一个 _IncompatibleKeys 对象
                  # 更安全的做法是在 load_state_dict 返回 None 时 (strict=True 且无错), 
                  # incompatible_keys_info 也为 None，不应该去修改它。
                  # 如果 strict=False, 它总是 _IncompatibleKeys。
                pass # 或者，如果想跟踪，需要更复杂的逻辑来创建或修改 _IncompatibleKeys
            elif hasattr(incompatible_keys_info, 'missing_keys'):
                if 'logit_scale' not in incompatible_keys_info.missing_keys:
                      incompatible_keys_info.missing_keys.append('logit_scale')
    else: 
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        resize_pos_embed(state_dict, model)

        final_strict_value = strict # 记录最终使用的 strict 值
        incompatible_keys = model.load_state_dict(state_dict, strict=final_strict_value)
        incompatible_keys_info = incompatible_keys # 将结果存入通用变量

        # --- 正确处理 incompatible_keys (来自之前的修复) ---
        if final_strict_value is False and incompatible_keys is not None: # 检查是否为 _IncompatibleKeys 对象
            if hasattr(incompatible_keys, 'missing_keys') and incompatible_keys.missing_keys:
                 logging.warning(f"Missing keys when loading model state_dict: {incompatible_keys.missing_keys}")
            if hasattr(incompatible_keys, 'unexpected_keys') and incompatible_keys.unexpected_keys:
                 logging.warning(f"Unexpected keys when loading model state_dict: {incompatible_keys.unexpected_keys}")
        
    logging.info(f"Finished loading checkpoint '{checkpoint_path}' (strict={final_strict_value}).")
    

def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        dasiglip_num_degrad_types: int = NUM_DEGRADATION_TYPES,
        dasiglip_controller_depth: Optional[int] = None,
        dasiglip_freeze_base: bool = True,
):
    
    logging.info("FACTORY.PY ---- ENTERING create_model ----") # <--- 日志点 0 (函数入口)
    logging.info(f"FACTORY.PY ---- Received model_name: {model_name}, precision: {precision}, pretrained: {pretrained}") # <--- 日志点 1
    
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    has_dasiglip_prefix = model_name.startswith(DASIGLIP_HF_PREFIX)

    if isinstance(device, str):
        device = torch.device(device)

    # 1. Instantiate Model Structure
    if has_dasiglip_prefix:
        siglip_base_model_name = model_name[len(DASIGLIP_HF_PREFIX):]
        logging.info(f"FACTORY.PY ---- Creating DaSiglipModel for base: {siglip_base_model_name}") # <--- 日志点 2
        model = DaSiglipModel(
            model_name=siglip_base_model_name,
            num_degradation_types=dasiglip_num_degrad_types,
            controller_transformer_depth=dasiglip_controller_depth,
            freeze_base=dasiglip_freeze_base,
        )
        pretrained_cfg = {}

    elif has_hf_hub_prefix:
        logging.info(f"Creating model from HF Hub: {model_name}")
        model_id = model_name[len(HF_HUB_PREFIX):]
        config_path = download_pretrained_from_hf(model_id, filename=HF_CONFIG_NAME, cache_dir=cache_dir)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        pretrained_cfg = config.get('preprocess_cfg', {})
        model_cfg_hf = config.get('model_cfg', {})
        underlying_model_name = model_cfg_hf.get('model_name', Path(model_id).name)
        if underlying_model_name in _MODEL_CONFIGS:
             logging.info(f"Found matching JSON config for HF model's underlying type: {underlying_model_name}")
        else:
             logging.warning(f"No local JSON config for HF model's underlying type: {underlying_model_name}. Creation might rely on generic CLIP/CustomTextCLIP.")
        model = create_model(
             model_name=underlying_model_name, pretrained=None, precision=precision, device=device, jit=jit,
             force_quick_gelu=force_quick_gelu, force_custom_text=force_custom_text,
             force_patch_dropout=force_patch_dropout, force_image_size=force_image_size,
             pretrained_image=pretrained_image, pretrained_hf=pretrained_hf,
             cache_dir=cache_dir, output_dict=output_dict, require_pretrained=False,
             dasiglip_num_degrad_types=dasiglip_num_degrad_types,
             dasiglip_controller_depth=dasiglip_controller_depth,
             dasiglip_freeze_base=dasiglip_freeze_base,
        )
        pretrained = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    elif pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading (and creating) pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, precision=precision, device=device, cache_dir=cache_dir)
        pretrained_cfg = _MODEL_INFO.get(model_name, {})

    else:
        model_name_clean = model_name.replace('/', '-')
        model_cfg = get_model_config(model_name_clean)
        if model_cfg is None:
            logging.error(f'Model config for {model_name_clean} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name_clean} not found.')
        logging.info(f'Loaded {model_name_clean} model config.')
        if force_quick_gelu: model_cfg["quick_gelu"] = True
        if force_patch_dropout is not None: model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout
        if force_image_size is not None: model_cfg["vision_cfg"]["image_size"] = force_image_size
        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model: model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else: assert False, 'pretrained image towers currently only supported for timm models'
        is_hf_text_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_text_model
        if custom_text:
            if is_hf_text_model: model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
            if "coca" in model_name.lower(): model = CoCa(**model_cfg)
            else: model = CustomTextCLIP(**model_cfg)
        else:
            model = CLIP(**model_cfg)
        pretrained_cfg = {}

    logging.info(f"FACTORY.PY ---- Model instance created, type: {type(model)}") # <--- 日志点 3

    # 2. Move model to device
    model.to(device)

    # 3. Load Pretrained Weights
    pretrained_loaded = False
    if pretrained and pretrained.lower() != 'openai':
        checkpoint_path = ''
        current_model_name_for_tag = model_name
        if has_hf_hub_prefix and hasattr(model, 'model_name'):
            current_model_name_for_tag = model.model_name

        if not has_dasiglip_prefix:
            cfg_from_tag = get_pretrained_cfg(current_model_name_for_tag, pretrained)
            if cfg_from_tag:
                checkpoint_path = download_pretrained(cfg_from_tag, cache_dir=cache_dir)
                if not has_hf_hub_prefix:
                     pretrained_cfg.update(cfg_from_tag)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained
        elif has_dasiglip_prefix and os.path.exists(pretrained):
             checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained weights from {checkpoint_path}.')
            load_strict = not has_dasiglip_prefix
            load_checkpoint(model, checkpoint_path, strict=load_strict)
            pretrained_loaded = True
        else:
            error_str = f'Pretrained weights ({pretrained}) not found for model {model_name}.'
            available_tags_msg = f'Available pretrained tags ({list_pretrained_tags_by_model(current_model_name_for_tag)}).' if not has_dasiglip_prefix else ''
            error_str += " " + available_tags_msg
            if require_pretrained or (not has_dasiglip_prefix and is_pretrained_cfg(current_model_name_for_tag, pretrained)):
                 logging.error(error_str)
                 raise RuntimeError(error_str)
            elif has_dasiglip_prefix and require_pretrained and pretrained:
                 logging.error(error_str + " DA-SigLIP controller weights are required but not found.")
                 raise RuntimeError(error_str)
            else:
                 logging.warning(f"{error_str} Model initialized with base SigLIP weights (if DA-SigLIP) or randomly.")

    if require_pretrained and not pretrained_loaded:
        raise RuntimeError(
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    logging.info(f"FACTORY.PY ---- Checking precision block, current precision: {precision}") # <--- 日志点 4
    
    # 4. Apply Precision Conversion (AFTER loading checkpoint)
    if precision in ("amp", "amp_bf16", "amp_bfloat16"):
        # 对于标准的混合精度 ("amp")：
        # - 模型参数（尤其是可训练的）应保持 float32。
        # - torch.cuda.amp.autocast 将在运行时自动将操作转换为 float16/bfloat16。
        # - GradScaler 将处理缩放和取消缩放 float16 梯度以更新 float32 参数。
        logging.info(f"Using Mixed Precision training ('{precision}'). Trainable model parameters remain primarily fp32. Autocast handles low precision ops.")
        
        if isinstance(model, DaSiglipModel):
            # 确保 visual_control 和 logit_scale 保持/恢复为 float32
            # 如果它们在模型初始化时已经是 float32，则无需操作。
            # 如果它们因某种原因被更改了，确保它们是 float32。
            model.visual_control.float() # 确保它是 float32
            if hasattr(model, 'logit_scale') and isinstance(model.logit_scale, nn.Parameter):
                model.logit_scale.data = model.logit_scale.data.float() # 确保它是 float32

            logging.info(f"  DaSiglipModel.visual_control parameters set to float32 for '{precision}' mode.")
            try:
                patch_emb_weight = model.visual_control.vision_model.embeddings.patch_embedding.weight
                logging.info(f"  DaSiglipModel visual_control.patch_embedding.weight dtype for '{precision}': {patch_emb_weight.dtype}")
                logging.info(f"  DaSiglipModel logit_scale dtype for '{precision}': {model.logit_scale.dtype}")
            except AttributeError:
                logging.warning("Could not access patch_embedding or logit_scale for dtype logging in 'amp' mode.")

        # 对于其他非 DaSiglip 模型，如果它们有特殊的 LayerNormFp32，可能需要处理
        # (这部分逻辑可以保持，因为它针对的是非 HF、非 OpenAI 模型的特定 LayerNorm)
        elif not model_name.startswith(HF_HUB_PREFIX) and not (pretrained and pretrained.lower() == 'openai'):
            try:
                from .transformer import LayerNormFp32 # 确保这个导入有效
                def _convert_ln_fp32(m):
                    if isinstance(m, LayerNormFp32): # 或者原始的 LayerNorm
                        m.weight.data = m.weight.data.to(torch.float32)
                        if m.bias is not None:
                             m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln_fp32)
                logging.info(f"Applied LayerNorm float32 conversion for non-HF/non-OpenAI model with precision '{precision}'.")
            except ImportError:
                logging.warning("Could not import LayerNormFp32 for custom LayerNorm precision handling.")
        
        # 不需要调用 convert_weights_to_lp(model, ...) 或 model.to(low_precision_dtype) 对于 "amp" 模式

    elif precision in ("fp16", "bf16", "pure_fp16", "pure_bf16"):
        # 对于纯半精度模式或非 "amp" 的半精度模式：
        # - 将模型参数（或特定部分）直接转换为目标半精度。
        # - GradScaler 在这种情况下通常不使用 (scaler 会是 None)。
        target_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
        logging.info(f"Applying FULL low precision '{precision}' (target dtype: {target_dtype}) to model.")
        
        if isinstance(model, DaSiglipModel):
            logging.info(f"  Converting DaSiglipModel.visual_control to {target_dtype}.")
            model.visual_control.to(target_dtype)
            if hasattr(model, 'logit_scale') and isinstance(model.logit_scale, nn.Parameter):
                if model.logit_scale.data.dtype != target_dtype:
                    logging.info(f"  Converting DaSiglipModel.logit_scale to {target_dtype}.")
                    model.logit_scale.data = model.logit_scale.data.to(target_dtype)
            try:
                patch_emb_weight = model.visual_control.vision_model.embeddings.patch_embedding.weight
                logging.info(f"  DaSiglipModel visual_control.patch_embedding.weight dtype after PURE conversion: {patch_emb_weight.dtype}")
            except AttributeError: pass
        else:
            # 对于其他模型，在纯半精度模式下转换整个模型
            model.to(target_dtype)
            logging.info(f"  Converted entire model '{model_name}' to {target_dtype}.")
    
    elif precision == 'fp32':
        logging.info("Model precision set to 'fp32'. No conversion applied by factory.")
        if isinstance(model, DaSiglipModel): # 确保是 fp32
            model.visual_control.float()
            if hasattr(model, 'logit_scale') and isinstance(model.logit_scale, nn.Parameter):
                 model.logit_scale.data = model.logit_scale.data.float()

    else:
        logging.warning(f"Unknown precision: {precision}. Model will remain in its original precision.")



    if not model_name.startswith(DASIGLIP_HF_PREFIX):
         if hasattr(model, 'visual') and model.visual is not None:
             model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
             model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        try:
            model = torch.jit.script(model)
        except Exception as e:
            logging.warning(f"Torch JIT scripting failed for {model_name}: {e}")

    logging.error("FACTORY.PY ---- EXITING create_model ----") # <--- 日志点 16
    return model


def create_loss(args):
    if hasattr(args, 'distill') and args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=getattr(args, 'coca_caption_loss_weight', 1.0),
            clip_loss_weight=getattr(args, 'coca_contrastive_loss_weight', 1.0),
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.model.lower().startswith(DASIGLIP_HF_PREFIX.lower()):
        logging.info("Creating DaSiglipLoss.")
        return DaSiglipLoss(
            lambda_bce_degrad=args.bce_loss_weight,
            lambda_con_degrad=args.degrad_contrastive_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    else:
        logging.info("Creating standard ClipLoss.")
        return ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], Any]] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        dasiglip_num_degrad_types: int = NUM_DEGRADATION_TYPES,
        dasiglip_controller_depth: Optional[int] = None,
        dasiglip_freeze_base: bool = True,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        require_pretrained=False,
        dasiglip_num_degrad_types=dasiglip_num_degrad_types,
        dasiglip_controller_depth=dasiglip_controller_depth,
        dasiglip_freeze_base=dasiglip_freeze_base,
    )

    tokenizer = None
    preprocess_train = None
    preprocess_val = None

    if model_name.startswith(DASIGLIP_HF_PREFIX):
        hf_model_name = model_name[len(DASIGLIP_HF_PREFIX):]
        logging.info(f"Loading SigLIP processor for transforms: {hf_model_name}")
        try:
            processor = AutoProcessor.from_pretrained(hf_model_name, cache_dir=cache_dir)
            tokenizer = processor.tokenizer
            preprocess_train = processor.image_processor
            preprocess_val = processor.image_processor
            if hasattr(model, 'visual') and model.visual is not None and \
               hasattr(processor.image_processor, 'image_mean') and \
               hasattr(processor.image_processor, 'image_std'):
                 if hasattr(model.visual, 'image_mean'):
                     model.visual.image_mean = processor.image_processor.image_mean
                 if hasattr(model.visual, 'image_std'):
                     model.visual.image_std = processor.image_processor.image_std
        except Exception as e:
            logging.error(f"Failed to load SigLIP processor for {hf_model_name}. Error: {e}")
            raise e
    else:
        tokenizer = get_tokenizer(model_name)
        from .transform import image_transform, AugmentationCfg
        
        current_image_size = None
        if hasattr(model, 'visual') and model.visual is not None and hasattr(model.visual, 'image_size'):
            current_image_size = model.visual.image_size
        elif force_image_size is not None:
            current_image_size = force_image_size if isinstance(force_image_size, tuple) else (force_image_size, force_image_size)
        else:
            logging.warning(f"Model {model_name} or its visual attribute/image_size is not properly initialized for transform creation. Using default size 224.")
            current_image_size = (224,224)


        image_mean_used = image_mean or getattr(model.visual, 'image_mean', None) or OPENAI_DATASET_MEAN
        image_std_used = image_std or getattr(model.visual, 'image_std', None) or OPENAI_DATASET_STD

        if isinstance(aug_cfg, dict):
             aug_cfg_obj = AugmentationCfg(**aug_cfg)
        elif isinstance(aug_cfg, AugmentationCfg):
             aug_cfg_obj = aug_cfg
        else:
             aug_cfg_obj = AugmentationCfg()

        preprocess_train = image_transform(
            current_image_size,
            is_train=True,
            mean=image_mean_used,
            std=image_std_used,
            aug_cfg=aug_cfg_obj,
        )
        preprocess_val = image_transform(
            current_image_size,
            is_train=False,
            mean=image_mean_used,
            std=image_std_used,
        )

    return model, preprocess_train, preprocess_val, tokenizer


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        dasiglip_num_degrad_types: int = NUM_DEGRADATION_TYPES,
        dasiglip_controller_depth: Optional[int] = None,
        dasiglip_freeze_base: bool = True,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        require_pretrained=True,
        dasiglip_num_degrad_types=dasiglip_num_degrad_types,
        dasiglip_controller_depth=dasiglip_controller_depth,
        dasiglip_freeze_base=dasiglip_freeze_base,
    )

    if not return_transform:
        return model

    if model_name.startswith(DASIGLIP_HF_PREFIX):
        hf_model_name = model_name[len(DASIGLIP_HF_PREFIX):]
        logging.info(f"Loading SigLIP processor for: {hf_model_name}")
        try:
            processor = AutoProcessor.from_pretrained(hf_model_name, cache_dir=cache_dir)
            return model, processor
        except Exception as e:
            logging.error(f"Failed to load SigLIP processor for {hf_model_name}. Error: {e}")
            raise e
    else:
        from .transform import image_transform
        current_image_size = None
        if hasattr(model, 'visual') and model.visual is not None and hasattr(model.visual, 'image_size'):
            current_image_size = model.visual.image_size
        elif force_image_size is not None:
            current_image_size = force_image_size if isinstance(force_image_size, tuple) else (force_image_size, force_image_size)
        else:
            logging.warning(f"Model {model_name} or its visual attribute/image_size is not properly initialized for transform creation. Using default size 224.")
            current_image_size = (224,224)


        image_mean_used = image_mean or getattr(model.visual, 'image_mean', None) or OPENAI_DATASET_MEAN
        image_std_used = image_std or getattr(model.visual, 'image_std', None) or OPENAI_DATASET_STD
        preprocess = image_transform(
            current_image_size,
            is_train=False,
            mean=image_mean_used,
            std=image_std_used,
        )
        return model, preprocess
