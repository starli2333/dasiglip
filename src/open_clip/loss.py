# -*- coding: utf-8 -*-
import logging # Added logging
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    """ Gathers features from all processes. Applies to image and text features separately. """
    if not has_distributed and world_size > 1:
         raise RuntimeError(
             "torch.distributed did not import correctly, please use a PyTorch version with support."
         )

    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    elif world_size > 1: # Standard DDP
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features_list = [torch.zeros_like(image_features) for _ in range(world_size)]
            all_text_features_list = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(all_image_features_list, image_features)
            dist.all_gather(all_text_features_list, text_features)
            if not local_loss: # Ensure grads for local rank
                 all_image_features_list[rank] = image_features
                 all_text_features_list[rank] = text_features
            all_image_features = torch.cat(all_image_features_list, dim=0)
            all_text_features = torch.cat(all_text_features_list, dim=0)

            # Reapply autograd Nodes for gather_with_grad
            # TODO: Find a more elegant way? This works for now.
            # Needs to be applied only during the gather_with_grad case
            # This is adapted from torch.distributed.nn.all_gather from PT 2.0
            # Requires Pytorch 1.11+
            class GatherFeatures(torch.autograd.Function):
                @staticmethod
                def forward(ctx, image_features, text_features):
                    ctx.rank = dist.get_rank()
                    ctx.world_size = dist.get_world_size()
                    gathered_image_features = [torch.zeros_like(image_features) for _ in range(self.world_size)]
                    gathered_text_features = [torch.zeros_like(text_features) for _ in range(self.world_size)]
                    dist.all_gather(gathered_image_features, image_features)
                    dist.all_gather(gathered_text_features, text_features)
                    gathered_image_features[ctx.rank] = image_features # ensure grads for local rank
                    gathered_text_features[ctx.rank] = text_features   # ensure grads for local rank
                    all_image_features = torch.cat(gathered_image_features, dim=0)
                    all_text_features = torch.cat(gathered_text_features, dim=0)
                    return all_image_features, all_text_features

                @staticmethod
                def backward(ctx, grad_image_features, grad_text_features):
                    image_grads = torch.empty_like(grad_image_features)
                    text_grads = torch.empty_like(grad_text_features)
                    dist.reduce_scatter(image_grads, list(grad_image_features.chunk(ctx.world_size, dim=0)), op=dist.ReduceOp.SUM)
                    dist.reduce_scatter(text_grads, list(grad_text_features.chunk(ctx.world_size, dim=0)), op=dist.ReduceOp.SUM)
                    return image_grads[ctx.rank], text_grads[ctx.rank]

            # Apply the custom autograd function
            all_image_features, all_text_features = GatherFeatures.apply(image_features, text_features)

        else: # No grad gathering
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # non-distributed, just return the features themselves
        all_image_features = image_features
        all_text_features = text_features

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    """
    Standard CLIP contrastive loss.
    Calculates cross-entropy loss between image-text similarity scores.
    """
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """ Calculate or retrieve cached ground-truth labels for contrastive loss. """
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        """ Calculate similarity logits between image and text features. """
        if self.world_size > 1:
            # Gather features from all GPUs if distributed training
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                # Local loss calculation (each GPU only compares its images to all texts)
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                # Global loss calculation (all images compared to all texts)
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T # More efficient than redundant calculation
        else:
            # Non-distributed case
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T # Original had image_features.T, likely a typo

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        """ Calculate the CLIP contrastive loss. """
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        # Calculate cross-entropy loss for both directions
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    """
    Loss for CoCa model, combining contrastive loss and captioning loss.
    Inherits from ClipLoss for the contrastive part.
    """
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            # ClipLoss args
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        # CrossEntropyLoss for the captioning part, ignoring padding tokens
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        """
        Calculate combined CoCa loss.

        Args:
            image_features: Features from the image encoder.
            text_features: Features from the text encoder (contrastive part).
            logits: Logits from the text decoder (captioning part). Shape (B, L, Vocab).
            labels: Ground truth caption tokens. Shape (B, L).
            logit_scale: Logit scale parameter.
            output_dict: Whether to return loss components in a dict.
        """
        clip_loss = torch.tensor(0., device=image_features.device)

        # Calculate contrastive loss if weight > 0
        if self.clip_loss_weight > 0:
            clip_loss = super().forward(image_features, text_features, logit_scale) # Call ClipLoss forward
            clip_loss = self.clip_loss_weight * clip_loss

        # Calculate captioning loss
        # Need to reshape logits for CrossEntropyLoss: (B, Vocab, L)
        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1), # Reshape to (B, Vocab, L)
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        # Return individual losses for potential separate logging or analysis
        return clip_loss, caption_loss


# Original DaClipLoss - kept for reference, but replaced by DaSiglipLoss
# class DaClipLoss(ClipLoss):
#     def forward(
#             self,
#             image_features,
#             text_features,
#             image_degra_features,
#             text_degra_features,
#             logit_scale,
#             output_dict=False
#     ):
#         clip_loss = super().forward(image_features, text_features, logit_scale)
#         degra_loss = super().forward(image_degra_features, text_degra_features, logit_scale)

#         if output_dict:
#             return {"contrastive_loss": clip_loss, "degra_loss": degra_loss}

#         return clip_loss, degra_loss


class DaSiglipLoss(nn.Module):
    """
    DA-SigLIP 的损失函数，结合了内容对比损失和退化类型的多标签分类损失。
    可选地加入退化特征与退化文本描述的对比损失。
    """
    def __init__(
        self,
        lambda_bce_degrad=1.0,       # 退化分类损失的权重
        lambda_con_degrad=0.5,      # (可选) 退化特征对比损失的权重
        # ClipLoss 相关参数 (用于对比损失部分)
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        """
        初始化 DaSiglipLoss。

        Args:
            lambda_bce_degrad (float): BCE 退化分类损失的权重。
            lambda_con_degrad (float): (可选) 退化特征对比损失的权重。设为 0 则禁用。
            local_loss (bool): 是否计算 local loss (用于分布式训练)。
            gather_with_grad (bool): 是否在 gather 特征时保留梯度 (用于分布式训练)。
            cache_labels (bool): 是否缓存对比损失的 ground truth 标签。
            rank (int): 当前进程的 rank (用于分布式训练)。
            world_size (int): 总进程数 (用于分布式训练)。
            use_horovod (bool): 是否使用 Horovod 进行分布式训练。
        """
        super().__init__()
        self.lambda_bce_degrad = lambda_bce_degrad
        self.lambda_con_degrad = lambda_con_degrad

        logging.info(f"Initializing DaSiglipLoss with BCE weight: {lambda_bce_degrad}, Degrad Contrastive weight: {lambda_con_degrad}")

        # 初始化用于内容对比损失的 ClipLoss
        # 注意：虽然 SigLIP 本身用 Sigmoid Loss 训练，但这里我们沿用 DA-CLIP 的思路，
        # 对 Controller 输出的特征使用标准的对比损失 (InfoNCE/CrossEntropy) 进行 fine-tune。
        self.content_loss_fn = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        logging.info("Initialized Content Contrastive Loss (ClipLoss)")

        # 初始化用于退化特征对比损失的 ClipLoss (如果权重>0)
        if self.lambda_con_degrad > 0:
            self.degrad_contrastive_loss_fn = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
                use_horovod=use_horovod
            )
            logging.info("Initialized Degradation Contrastive Loss (ClipLoss)")
        else:
            self.degrad_contrastive_loss_fn = None
            logging.info("Degradation Contrastive Loss is disabled (lambda_con_degrad=0).")

        # 初始化多标签分类损失 (Binary Cross Entropy with Logits)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        logging.info("Initialized Degradation Classification Loss (BCEWithLogitsLoss)")

    def forward(self, **outputs):
        """
        计算总损失。

        Args:
            outputs (dict): DaSiglipModel forward 方法的输出字典。
                            必须包含 'image_content_features', 'text_content_features',
                            'degradation_logits', 'degradation_target', 'logit_scale'。
                            可选包含 'image_degradation_features', 'text_degradation_features'。

        Returns:
            dict: 包含各项损失和总损失的字典。
        """
        # 1. 内容对比损失
        image_content_features = outputs.get("image_content_features")
        text_content_features = outputs.get("text_content_features")
        logit_scale = outputs.get("logit_scale")

        # 检查必要的输入是否存在
        if image_content_features is None or text_content_features is None or logit_scale is None:
             # 如果在推理阶段调用（没有文本），则跳过内容损失
             if text_content_features is None and logit_scale is None:
                 content_loss = torch.tensor(0.0, device=image_content_features.device if image_content_features is not None else 'cpu')
                 logging.debug("Skipping content contrastive loss due to missing text features/logit scale (likely inference).")
             else:
                 raise ValueError("Missing required features/logit_scale for content contrastive loss.")
        else:
             # ClipLoss forward 返回的是字典或标量，我们需要标量值
             content_loss_output = self.content_loss_fn(
                 image_features=image_content_features,
                 text_features=text_content_features,
                 logit_scale=logit_scale,
                 output_dict=True # 请求字典输出以便获取 'contrastive_loss'
             )
             content_loss = content_loss_output['contrastive_loss']


        # 2. 退化多标签分类损失
        degradation_logits = outputs.get("degradation_logits")
        degradation_target = outputs.get("degradation_target")

        if degradation_logits is None or degradation_target is None:
             # 可能在推理阶段没有 target
             if degradation_target is None:
                 bce_degrad_loss = torch.tensor(0.0, device=degradation_logits.device if degradation_logits is not None else content_loss.device)
                 logging.debug("Skipping BCE degradation loss due to missing target (likely inference).")
             else:
                 raise ValueError("Missing required logits/target for degradation classification loss.")
        else:
             # 确保 target 是 float 类型
             bce_degrad_loss = self.bce_loss_fn(degradation_logits, degradation_target.float())

        # 3. (可选) 退化特征对比损失
        degrad_contrastive_loss = torch.tensor(0.0, device=content_loss.device) # 初始化为0
        if self.degrad_contrastive_loss_fn is not None:
            image_degradation_features = outputs.get("image_degradation_features")
            text_degradation_features = outputs.get("text_degradation_features")

            if image_degradation_features is not None and text_degradation_features is not None and logit_scale is not None:
                # ClipLoss forward 返回的是字典或标量，我们需要标量值
                degrad_contrastive_loss_output = self.degrad_contrastive_loss_fn(
                    image_features=image_degradation_features,
                    text_features=text_degradation_features,
                    logit_scale=logit_scale,
                    output_dict=True # 请求字典输出
                )
                degrad_contrastive_loss = degrad_contrastive_loss_output['contrastive_loss']
            elif self.lambda_con_degrad > 0: # 只有在权重>0时才警告
                logging.warning("Degradation contrastive loss enabled (lambda > 0), but required features/logit_scale are missing in the output dict.")


        # 4. 总损失
        total_loss = (content_loss +
                      self.lambda_bce_degrad * bce_degrad_loss +
                      self.lambda_con_degrad * degrad_contrastive_loss)

        return {
            "contrastive_loss": content_loss,
            "bce_degrad_loss": bce_degrad_loss,
            "contrastive_degrad_loss": degrad_contrastive_loss, # 可能为 0
            "loss": total_loss  # 'loss' 是训练循环通常寻找的主损失键
        }


class DistillClipLoss(ClipLoss):
    """
    CLIP loss distilled from a teacher model.
    Adds a distillation term to the standard contrastive loss.
    """
    def dist_loss(self, teacher_logits, student_logits):
        """ Kullback-Leibler divergence based distillation loss. """
        # KL divergence between soft labels (teacher softmax) and student logits (log_softmax)
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            # Teacher model outputs
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        # Calculate student logits
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        # Calculate teacher logits (using no_grad as teacher is fixed)
        with torch.no_grad():
            dist_logits_per_image, dist_logits_per_text = \
                self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        # Standard contrastive loss for the student
        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])
        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        # Distillation loss (KL divergence)
        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        # Return individual losses for potential separate logging or weighting
        return contrastive_loss, distill_loss

