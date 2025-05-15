# -*- coding: utf-8 -*-
import copy
import logging
from typing import Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipTextModel, SiglipVisionConfig, SiglipTextConfig

from .transformer import ControlTransformer # Ensure this is adaptable or placeholder
from .constants import NUM_DEGRADATION_TYPES

class DaSiglipModel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        num_degradation_types: int = NUM_DEGRADATION_TYPES,
        freeze_base: bool = True,
        controller_transformer_depth: Optional[int] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.num_degradation_types = num_degradation_types
        self.controller_transformer_depth = controller_transformer_depth

        logging.info(f"Initializing DaSiglipModel with base: {model_name}")
        try:
            # Load base models in their default precision (usually float32)
            self.siglip_visual = SiglipVisionModel.from_pretrained(model_name)
            self.siglip_text = SiglipTextModel.from_pretrained(model_name)
            logging.info(f"Loaded base SigLIP visual and text models from {model_name}")
        except Exception as e:
            logging.error(f"Failed to load SigLIP model '{model_name}' from Hugging Face. Error: {e}")
            raise e

        self.vision_config = self.siglip_visual.config
        self.text_config = self.siglip_text.config
        controller_config = copy.deepcopy(self.vision_config)

        if self.controller_transformer_depth is not None:
             logging.info(f"Controller custom depth specified: {self.controller_transformer_depth}")
             controller_config.num_hidden_layers = self.controller_transformer_depth

        logging.info(f"Creating visual controller with config: {controller_config}")
        # Initialize visual_control in default precision (FP32)
        self.visual_control = SiglipVisionModel(controller_config)
        self._wrap_controller_transformer()

        controller_output_dim = self.vision_config.hidden_size
        logging.info(f"Controller base output dimension: {controller_output_dim}")
        logging.info(f"Number of degradation types for head: {num_degradation_types}")

        self.degradation_feature_head = nn.Identity()
        self.degradation_logit_head = nn.Linear(controller_output_dim, num_degradation_types)
        logging.info(f"Initialized degradation logit head: Linear({controller_output_dim}, {num_degradation_types})")

        if hasattr(self.siglip_text, 'logit_scale') and self.siglip_text.logit_scale is not None:
            self.logit_scale = nn.Parameter(self.siglip_text.logit_scale.data.clone())
            logging.info("Copied logit_scale from SigLIP text model.")
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
            logging.info("Initialized new logit_scale.")

        if freeze_base:
            self.lock_siglip()

    def cast_controller_precision(self, dtype: torch.dtype):
        """Explicitly casts the visual_control submodule to the specified dtype."""
        if self.visual_control is not None:
            logging.info(f"Attempting to cast DaSiglipModel.visual_control to {dtype} using .to(dtype).")
            self.visual_control.to(dtype)
            # Verification log
            try:
                patch_emb_layer = None
                if hasattr(self.visual_control, 'vision_model') and \
                   hasattr(self.visual_control.vision_model, 'embeddings') and \
                   hasattr(self.visual_control.vision_model.embeddings, 'patch_embedding'):
                    patch_emb_layer = self.visual_control.vision_model.embeddings.patch_embedding
                elif hasattr(self.visual_control, 'embeddings') and \
                     hasattr(self.visual_control.embeddings, 'patch_embedding'):
                    patch_emb_layer = self.visual_control.embeddings.patch_embedding
                
                if patch_emb_layer and hasattr(patch_emb_layer, 'weight'):
                    logging.info(f"Datatype of visual_control's patch_embedding.weight after cast_controller_precision: {patch_emb_layer.weight.dtype}")
                else:
                    logging.warning("Could not find patch_embedding layer in visual_control to log its dtype after cast_controller_precision.")
            except AttributeError:
                logging.warning("Could not access patch_embedding to log its dtype after cast_controller_precision.")
        else:
            logging.warning("visual_control is None, cannot cast its precision.")


    def _wrap_controller_transformer(self):
        target_transformer_module = None
        if hasattr(self.visual_control, 'vision_model') and \
           hasattr(self.visual_control.vision_model, 'encoder') and \
           hasattr(self.visual_control.vision_model.encoder, 'layers'):
            target_transformer_module = self.visual_control.vision_model.encoder
            logging.info("Found transformer module in visual_control.vision_model.encoder")
        elif hasattr(self.visual_control, 'encoder') and \
             hasattr(self.visual_control.encoder, 'layers'):
            target_transformer_module = self.visual_control.encoder
            logging.info("Found transformer module in visual_control.encoder")

        if target_transformer_module is not None:
            try:
                # This is a placeholder. Actual wrapping needs ControlTransformer to be compatible
                # with Hugging Face's SiglipEncoder structure.
                # wrapped_transformer = ControlTransformer(target_transformer_module)
                # if hasattr(self.visual_control, 'vision_model') and hasattr(self.visual_control.vision_model, 'encoder'):
                #     self.visual_control.vision_model.encoder = wrapped_transformer
                # elif hasattr(self.visual_control, 'encoder'):
                #     self.visual_control.encoder = wrapped_transformer
                logging.warning("ControlTransformer wrapping for HF SigLIP models needs careful implementation. Current wrapping is a placeholder and likely ineffective.")
            except Exception as e:
                logging.error(f"Failed to wrap controller's transformer: {e}. Control features might not work correctly.")
        else:
            logging.warning("Could not find a compatible transformer module in visual_control to wrap with ControlTransformer.")


    def lock_siglip(self):
        logging.info("Locking base SigLIP visual and text models.")
        for param in self.siglip_visual.parameters():
            param.requires_grad = False
        for param in self.siglip_text.parameters():
            param.requires_grad = False
        self.siglip_visual.eval()
        self.siglip_text.eval()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.siglip_visual, 'gradient_checkpointing_enable'):
            self.siglip_visual.gradient_checkpointing_enable(gradient_checkpointing=enable)
            logging.info(f"Base SigLIP visual gradient checkpointing {'enabled' if enable else 'disabled'}.")
        if hasattr(self.siglip_text, 'gradient_checkpointing_enable'):
            self.siglip_text.gradient_checkpointing_enable(gradient_checkpointing=enable)
            logging.info(f"Base SigLIP text gradient checkpointing {'enabled' if enable else 'disabled'}.")
        if hasattr(self.visual_control, 'gradient_checkpointing_enable'):
            self.visual_control.gradient_checkpointing_enable(gradient_checkpointing=enable)
            logging.info(f"Controller visual gradient checkpointing {'enabled' if enable else 'disabled'}.")


    def encode_image(self, image_pixel_values: torch.Tensor, control: bool = False, normalize: bool = False):
        if control:
            logging.debug(f"In encode_image (control=True), input image_pixel_values.dtype: {image_pixel_values.dtype}") # <--- 新增日志
            # 确保 visual_control 内部的 patch_embedding 权重类型
            if hasattr(self.visual_control, 'vision_model') and hasattr(self.visual_control.vision_model, 'embeddings'):
                logging.debug(f"  visual_control.patch_embedding.weight.dtype before forward: {self.visual_control.vision_model.embeddings.patch_embedding.weight.dtype}") # <--- 新增日志
            
            try:
                controller_outputs = self.visual_control(
                    pixel_values=image_pixel_values, # Input should match visual_control's expected dtype
                    output_hidden_states=True
                )
            except TypeError as e:
                 logging.error(f"Controller forward pass failed. Error: {e}")
                 raise e
            except RuntimeError as e: # Catch runtime errors like dtype mismatch here
                 logging.error(f"RuntimeError during controller_outputs = self.visual_control(...): {e}")
                 # Log dtypes for debugging
                 logging.error(f"  Input pixel_values dtype: {image_pixel_values.dtype}")
                 if hasattr(self.visual_control, 'vision_model') and \
                    hasattr(self.visual_control.vision_model, 'embeddings') and \
                    hasattr(self.visual_control.vision_model.embeddings, 'patch_embedding'):
                     patch_emb_weight = self.visual_control.vision_model.embeddings.patch_embedding.weight
                     logging.error(f"  visual_control.patch_embedding.weight dtype: {patch_emb_weight.dtype}")
                 raise e


            h_c = controller_outputs.hidden_states if hasattr(controller_outputs, 'hidden_states') else None

            controller_pooled_output = controller_outputs.pooler_output
            if controller_pooled_output is None and hasattr(controller_outputs, 'last_hidden_state'):
                controller_pooled_output = controller_outputs.last_hidden_state[:, 0]
            if controller_pooled_output is None:
                raise ValueError("Controller did not produce a pooler_output or last_hidden_state.")

            f_degrad = self.degradation_feature_head(controller_pooled_output)
            degradation_logits = self.degradation_logit_head(f_degrad)

            with torch.no_grad(): # Base visual model is frozen
                content_outputs = self.siglip_visual(pixel_values=image_pixel_values) # Base model takes original input
                image_content_features = content_outputs.pooler_output
                if image_content_features is None and hasattr(content_outputs, 'last_hidden_state'):
                    image_content_features = content_outputs.last_hidden_state[:, 0]
                if image_content_features is None:
                     raise ValueError("Base visual model did not produce a pooler_output or last_hidden_state.")

            if normalize:
                 image_content_features = F.normalize(image_content_features, dim=-1)
                 f_degrad = F.normalize(f_degrad, dim=-1)

            return image_content_features, f_degrad, degradation_logits
        else:
            with torch.no_grad():
                outputs = self.siglip_visual(pixel_values=image_pixel_values)
                image_features = outputs.pooler_output
                if image_features is None and hasattr(outputs, 'last_hidden_state'):
                    image_features = outputs.last_hidden_state[:, 0]
                if image_features is None:
                    raise ValueError("Base visual model (direct call) did not produce pooler_output or last_hidden_state.")
            if normalize:
                image_features = F.normalize(image_features, dim=-1)
            return image_features, None, None

    def encode_text(self, text_input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, normalize: bool = False):
        with torch.no_grad():
            outputs = self.siglip_text(input_ids=text_input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # 如果没有 pooler_output (不太可能对于 SigLIPTextModel 的标准用法)，
                # 可以考虑使用 CLS token 的 last_hidden_state，但这通常需要额外的投影。
                # SigLIP 的设计是其 head 层处理池化和投影。
                # 因此，pooler_output 应该是首选。
                # 如果直接用 last_hidden_state[:, 0]，可能还需要通过一个线性头。
                # 但 SiglipTextModel 通常在其 head 中处理了这一点。
                logging.warning("SiglipTextModel output did not have 'pooler_output', attempting to use CLS token from last_hidden_state. This might not be the intended SigLIP text embedding.")
                text_features = outputs.last_hidden_state[:, 0] # 取 [CLS] token 的表示
            else:
                raise ValueError("SiglipTextModel output does not contain 'pooler_output' or 'last_hidden_state'. Cannot extract text features.")
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, image_pixel_values: Optional[torch.Tensor] = None, text_dict: Optional[dict] = None):
        output_dict = {}
        if image_pixel_values is not None:
            image_content_features, image_degradation_features, degradation_logits = self.encode_image(
                image_pixel_values, control=True, normalize=True
            )
            output_dict["image_content_features"] = image_content_features
            output_dict["image_degradation_features"] = image_degradation_features
            output_dict["degradation_logits"] = degradation_logits
        else:
             output_dict["image_content_features"] = None
             output_dict["image_degradation_features"] = None
             output_dict["degradation_logits"] = None

        if text_dict is not None:
            caption_tokens = text_dict.get('caption_tokens')
            caption_attention_mask = text_dict.get('caption_attention_mask')
            degradation_target = text_dict.get('degradation_target')
            true_degradation_text_tokens = text_dict.get('true_degradation_text_tokens')
            true_degradation_text_mask = text_dict.get('true_degradation_text_attention_mask')

            if caption_tokens is not None:
                text_content_features = self.encode_text(
                    caption_tokens, attention_mask=caption_attention_mask, normalize=True
                )
                output_dict["text_content_features"] = text_content_features
            else:
                 output_dict["text_content_features"] = None
            if degradation_target is not None:
                 output_dict["degradation_target"] = degradation_target
            else:
                 output_dict["degradation_target"] = None
            if true_degradation_text_tokens is not None:
                text_degradation_features = self.encode_text(
                    true_degradation_text_tokens, attention_mask=true_degradation_text_mask, normalize=True
                )
                output_dict["text_degradation_features"] = text_degradation_features
            else:
                output_dict["text_degradation_features"] = None
        else:
             output_dict["text_content_features"] = None
             output_dict["degradation_target"] = None
             output_dict["text_degradation_features"] = None

        output_dict["logit_scale"] = self.logit_scale.exp()
        return output_dict

