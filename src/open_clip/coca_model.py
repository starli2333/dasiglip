# -*- coding: utf-8 -*-
import logging
from typing import Optional, Tuple, Union, Callable # Added Callable

import torch
import torch.nn.functional as F
from torch import nn, einsum
# from einops import rearrange, repeat # Not used in the simplified version

# Import base components and config dataclasses
from .model import CLIPVisionCfg, CLIPTextCfg, QuickGELU # For default act_layer
from .transformer import VisionTransformer, TextTransformer, LayerNorm


class CoCa(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: Union[dict, CLIPVisionCfg],
            text_cfg: Union[dict, CLIPTextCfg],
            # img_size: int = 224, # Usually part of vision_cfg
            # patch_size: int = 16, # Usually part of vision_cfg
            # CoCa specific parameters
            multimodal_layers: int = 6,
            text_decoder_layers: int = 6, # Number of layers for the autoregressive text decoder
            dim_head: int = 64, # Dimension per attention head in decoders
            heads: Optional[int] = None, # Number of heads in decoders, if None, derived from text_cfg.width
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            **kwargs # To absorb other potential args from factory
    ):
        super().__init__()
        self.output_dict = output_dict

        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        # --- Instantiate Vision Tower ---
        vision_heads = vision_cfg.heads
        if vision_heads is None: # Fallback if heads not directly in cfg
            if vision_cfg.head_width is not None and vision_cfg.head_width > 0:
                vision_heads = vision_cfg.width // vision_cfg.head_width
            else: # Absolute fallback
                vision_heads = vision_cfg.width // 64 
                logging.warning(f"CoCa Vision tower heads not specified, defaulting to {vision_heads}.")
        
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
            global_average_pool=False, # CoCa needs token outputs from ViT
            output_tokens=True,       # CoCa needs token outputs from ViT
            output_dim=embed_dim,     # Vision tower output should match embed_dim for multimodal fusion
            act_layer=vision_cfg.act_layer or (QuickGELU if quick_gelu else nn.GELU),
            norm_layer=vision_cfg.norm_layer or LayerNorm
        )
        logging.info("CoCa: Vision tower instantiated.")

        # --- Instantiate Text Encoder (Unimodal part for contrastive loss) ---
        self.text_unimodal = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers, # This is for the unimodal encoder
            output_dim=embed_dim,   # Projects to embed_dim for contrastive loss
            act_layer=text_cfg.act_layer or (QuickGELU if quick_gelu else nn.GELU),
            norm_layer=text_cfg.norm_layer or LayerNorm
        )
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(text_cfg.context_length, text_cfg.width))
        self.ln_final_unimodal = (text_cfg.norm_layer or LayerNorm)(text_cfg.width)
        logging.info("CoCa: Unimodal text encoder instantiated.")

        # --- CoCa Specific Layers (Multimodal Decoder, Autoregressive Text Decoder) ---
        decoder_embed_dim = embed_dim # Assuming decoders operate at the main embed_dim
        decoder_heads_num = heads if heads is not None else text_cfg.heads # Use text_cfg.heads for decoder

        # Placeholder for Multimodal Decoder (attentional pooler)
        # This typically involves a few layers of cross-attention from text queries to image tokens
        # For simplicity, let's assume it results in pooled image features for the text decoder
        # A more accurate CoCa would have learnable queries that attend to visual tokens.
        self.multimodal_decoder_queries = nn.Parameter(torch.randn(1, text_cfg.context_length, decoder_embed_dim)) # Learnable queries
        self.multimodal_decoder = nn.ModuleList([
            # Example: TransformerEncoderLayer for self-attention among queries,
            # then TransformerDecoderLayer for cross-attention to image tokens.
            # This is a very simplified placeholder.
            nn.TransformerDecoderLayer(
                d_model=decoder_embed_dim, nhead=decoder_heads,
                dim_feedforward=decoder_embed_dim * 4, batch_first=True,
                activation=F.gelu # Explicitly use F.gelu or nn.GELU()
            ) for _ in range(multimodal_layers)
        ])
        logging.info(f"CoCa: Multimodal decoder ({multimodal_layers} layers) placeholder instantiated.")

        # Autoregressive Text Decoder
        self.text_decoder_embeddings = nn.Embedding(text_cfg.vocab_size, decoder_embed_dim) # Separate embeddings for decoder
        self.text_decoder_pos_embeddings = nn.Parameter(torch.empty(text_cfg.context_length, decoder_embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_heads,
            dim_feedforward=decoder_embed_dim * 4, batch_first=True,
            activation=F.gelu
        )
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=text_decoder_layers)
        self.to_logits = nn.Linear(decoder_embed_dim, text_cfg.vocab_size)
        logging.info(f"CoCa: Text decoder ({text_decoder_layers} layers) and to_logits head instantiated.")

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))) # Use torch.log
        self.init_weights()

        if cast_dtype is not None:
            self.to(cast_dtype)

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_decoder_embeddings.weight, std=0.02)
        nn.init.normal_(self.text_decoder_pos_embeddings, std=0.01)

        if hasattr(self.visual, 'init_weights'):
            self.visual.init_weights()
        
        # Initialize TextTransformer parts (unimodal)
        if hasattr(self.text_unimodal, 'init_weights'): # If TextTransformer has its own init
            self.text_unimodal.init_weights()
        else: # Manual init if not
            proj_std = (self.text_unimodal.width ** -0.5) * ((2 * self.text_unimodal.layers) ** -0.5)
            attn_std = self.text_unimodal.width ** -0.5
            fc_std = (2 * self.text_unimodal.width) ** -0.5
            for block in self.text_unimodal.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                if block.attn.in_proj_bias is not None: nn.init.zeros_(block.attn.in_proj_bias)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                if block.attn.out_proj.bias is not None: nn.init.zeros_(block.attn.out_proj.bias)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                if block.mlp.c_fc.bias is not None: nn.init.zeros_(block.mlp.c_fc.bias)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                if block.mlp.c_proj.bias is not None: nn.init.zeros_(block.mlp.c_proj.bias)

            if self.text_unimodal.text_projection is not None and isinstance(self.text_unimodal.text_projection, nn.Linear):
                nn.init.normal_(self.text_unimodal.text_projection.weight, std=self.text_unimodal.width ** -0.5)
                if self.text_unimodal.text_projection.bias is not None: nn.init.zeros_(self.text_unimodal.text_projection.bias)


        # Initialize decoder layers (example)
        for layer in self.multimodal_decoder:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.TransformerDecoderLayer): # More specific init for transformer layers
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

        # Initialize text_decoder (TransformerDecoder)
        for param in self.text_decoder.parameters():
            if param.dim() > 1: nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.to_logits.weight)
        if self.to_logits.bias is not None: nn.init.zeros_(self.to_logits.bias)


    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        # Get image tokens (B, NumImageTokens, Dim)
        image_tokens = self.visual(image) # output_tokens=True in VisionTransformer
        
        # For contrastive loss, CoCa typically uses a set of learnable queries
        # that attend to these image tokens via the multimodal decoder.
        # The output of these queries is then used as the "image_features_contrastive".
        
        # Simplified: Attentional pooling using multimodal_decoder_queries
        # This is a placeholder for a more complex multimodal decoder.
        # Expand queries to batch size
        batch_size = image_tokens.shape[0]
        queries = self.multimodal_decoder_queries.repeat(batch_size, 1, 1) # (B, NumQueries, Dim)
        
        # Pass through multimodal decoder layers (cross-attending to image_tokens)
        # This needs a proper TransformerDecoder structure for cross-attention.
        # The current nn.TransformerDecoderLayer in multimodal_decoder is not set up for this directly.
        # For now, let's use a simple mean of image tokens as a placeholder for contrastive features.
        # A real CoCa would use the output of the multimodal decoder.
        image_features_pooled = image_tokens.mean(dim=1) # (B, Dim) - Placeholder

        return F.normalize(image_features_pooled, dim=-1) if normalize else image_features_pooled

    def encode_text_unimodal(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        cast_dtype = self.text_unimodal.get_cast_dtype() if hasattr(self.text_unimodal, 'get_cast_dtype') else self.token_embedding.weight.dtype
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding[:text.shape[1]].to(cast_dtype) # Slice positional embedding
        x = x.permute(1, 0, 2)
        
        # Build attention mask dynamically based on actual sequence length
        attn_mask = self.build_attention_mask(text.shape[1]).to(text.device)
        x = self.text_unimodal(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final_unimodal(x)
        
        text_long = text if text.dtype == torch.long else text.to(torch.long)
        pooled_features = x[torch.arange(x.shape[0]), text_long.argmax(dim=-1)] # EOT pooling
        
        if hasattr(self.text_unimodal, 'text_projection') and self.text_unimodal.text_projection is not None:
            if isinstance(self.text_unimodal.text_projection, nn.Linear):
                 pooled_features = self.text_unimodal.text_projection(pooled_features)
            elif isinstance(self.text_unimodal.text_projection, nn.Parameter): # If projection is a matrix
                 pooled_features = pooled_features @ self.text_unimodal.text_projection
        
        return F.normalize(pooled_features, dim=-1) if normalize else pooled_features

    def build_attention_mask(self, context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def decode_text(self, image_tokens: torch.Tensor, target_captions: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        """
        Autoregressive text decoding for captioning.
        Args:
            image_tokens (torch.Tensor): Output from the vision encoder (B, NumImageTokens, Dim).
            target_captions (torch.Tensor): Ground truth caption tokens, shifted right (B, SeqLen).
            target_mask (Optional[torch.Tensor]): Attention mask for target_captions.
        Returns:
            torch.Tensor: Logits for next token prediction (B, SeqLen, VocabSize).
        """
        # Embed target captions
        caption_embeds = self.text_decoder_embeddings(target_captions) # (B, SeqLen, DecoderDim)
        caption_embeds = caption_embeds + self.text_decoder_pos_embeddings[:target_captions.size(1)]

        # Generate causal mask for self-attention in text decoder
        seq_len = target_captions.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=target_captions.device)

        # Multimodal fusion (placeholder - needs proper cross-attention)
        # For a real CoCa, image_tokens would be processed by a multimodal decoder (attentional pooler)
        # whose output becomes the 'memory' for the text_decoder's cross-attention.
        # Here, we'll use a simplified mean of image_tokens as memory.
        memory = image_tokens.mean(dim=1).unsqueeze(1).repeat(1, seq_len, 1) # (B, SeqLen, Dim) - very simplified

        # Pass through text decoder
        # TransformerDecoder expects target, memory, tgt_mask, memory_mask
        # tgt_mask is for self-attention within captions (causal)
        # memory_mask is for cross-attention to image features (usually None if all image tokens are attended to)
        decoder_output = self.text_decoder(
            tgt=caption_embeds,
            memory=memory, # This should be the output of the multimodal decoder
            tgt_mask=causal_mask,
            # memory_key_padding_mask=image_padding_mask # If image tokens have padding
        )
        caption_logits = self.to_logits(decoder_output) # (B, SeqLen, VocabSize)
        return caption_logits


    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text_contrastive: Optional[torch.Tensor] = None, # For contrastive loss
            text_captioning_input: Optional[torch.Tensor] = None, # Input to text decoder (e.g., shifted target)
            text_captioning_target_mask: Optional[torch.Tensor] = None, # Attention mask for text_captioning_input
            output_dict: Optional[bool] = None
    ):
        output_dict_flag = output_dict if output_dict is not None else self.output_dict

        image_features_for_contrastive = None
        text_features_for_contrastive = None
        caption_prediction_logits = None

        # --- Image Encoding (common for both paths if image is provided) ---
        image_tokens_visual = None
        if image is not None:
            image_tokens_visual = self.visual(image) # (B, NumImageTokens, Dim)

        # --- Contrastive Path ---
        if image_tokens_visual is not None and text_contrastive is not None:
            # For contrastive loss, pool image_tokens_visual
            # This is a placeholder, CoCa uses a more sophisticated attentional pooler
            image_features_for_contrastive = image_tokens_visual.mean(dim=1) # Simple mean pooling
            image_features_for_contrastive = F.normalize(image_features_for_contrastive, dim=-1)

            text_features_for_contrastive = self.encode_text_unimodal(text_contrastive, normalize=True)

        # --- Captioning Path ---
        if image_tokens_visual is not None and text_captioning_input is not None:
            caption_prediction_logits = self.decode_text(
                image_tokens=image_tokens_visual,
                target_captions=text_captioning_input,
                target_mask=text_captioning_target_mask
            )

        if output_dict_flag:
            return {
                "image_features": image_features_for_contrastive, # For contrastive
                "text_features": text_features_for_contrastive,   # For contrastive
                "logits": caption_prediction_logits,              # For captioning (B, L, V)
                "logit_scale": self.logit_scale.exp()
            }
        
        return image_features_for_contrastive, text_features_for_contrastive, caption_prediction_logits, self.logit_scale.exp()

