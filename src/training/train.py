# -*- coding: utf-8 -*-
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip.factory import create_loss

from .distributed import is_master
from .zero_shot import zero_shot_eval # 假设这个文件存在，尽管可能需要调整以适应DA-SigLIP
from .precision import get_autocast

# 新增：导入 scikit-learn 相关模块
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss_fn, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None): # Renamed loss to loss_fn
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # Distillation part removed as it's not the focus for DA-SigLIP evaluation
    # if args.distill:
    #     assert dist_model is not None
    #     dist_model.eval()

    if 'train' not in data:
         logging.warning("No training data found in 'data' dictionary. Skipping training epoch.")
         return

    # Ensure data['train'] is not None before accessing attributes
    if data['train'] is None or not hasattr(data['train'], 'dataloader'):
        logging.warning("Training data or dataloader is not available. Skipping training epoch.")
        return

    data['train'].set_epoch(epoch) # set epoch in process safe manner via sampler or shared custom dataloader attribute
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq if dataloader.num_batches > 0 else 0
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10)) if dataloader.num_samples > 0 else 1

    # accum_images, accum_texts, accum_features are for more advanced gradient accumulation strategies
    # For DA-SigLIP, we'll focus on accumulating gradients per micro-batch if accum_freq > 1
    if args.accum_freq > 1:
        accum_images_pixel_values, accum_text_dicts = [], [] # Store inputs for re-computation

    losses_m = {} # Dictionary to store AverageMeter for each loss component
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if len(dataloader) == 0: # Handle empty dataloader
        logging.warning("Training dataloader is empty. Skipping training epoch.")
        return


    for i, batch in enumerate(dataloader):
        if batch is None: # Defensive check for None batch
            logging.warning(f"Skipping None batch at training step {i}.")
            continue

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler and scheduler is not None: # Check scheduler existence
            scheduler(step)

        # batch typically contains (image_pixel_values, text_dict)
        images_pixel_values, text_dict = batch
        images_pixel_values = images_pixel_values.to(device=device, dtype=input_dtype, non_blocking=True)

        # Move all tensor items in text_dict to device
        text_dict_device = {}
        for k, v in text_dict.items():
            if isinstance(v, torch.Tensor):
                text_dict_device[k] = v.to(device=device, non_blocking=True)
            else:
                text_dict_device[k] = v # Keep non-tensor items as is

        data_time_m.update(time.time() - end)

        if optimizer: optimizer.zero_grad() # Check optimizer existence

        if args.accum_freq == 1:
            with autocast():
                model_out = model(image_pixel_values=images_pixel_values, text_dict=text_dict_device)
                losses = loss_fn(**model_out) # loss_fn is expected to be DaSiglipLoss instance
                total_loss = losses.get("loss") # DaSiglipLoss returns a dict with 'loss' key
                if total_loss is None: # Fallback if 'loss' key is missing
                     logging.error("Loss function did not return a 'loss' key or it was None.")
                     total_loss = sum(l for l_name, l in losses.items() if isinstance(l, torch.Tensor) and l.requires_grad and 'loss' in l_name.lower())
                     if not isinstance(total_loss, torch.Tensor) or total_loss.numel() == 0 : total_loss = torch.tensor(0.0, device=device, requires_grad=True if optimizer else False)
                     losses["loss"] = total_loss # Ensure 'loss' is in losses for logging

            if total_loss.requires_grad: # Only backward if loss requires grad
                 backward(total_loss, scaler)
        else:
            # First pass for gradient accumulation: recompute forward pass for each micro-batch
            # Store inputs if re-computation is needed (original OpenCLIP does this for features if not recomputing forward)
            # Here, we recompute forward for each micro-batch, so only inputs are needed.
            accum_images_pixel_values.append(images_pixel_values)
            accum_text_dicts.append(text_dict_device)

            if ((i + 1) % args.accum_freq) == 0:
                if optimizer: optimizer.zero_grad() # Zero grad before accumulating new gradients
                for j in range(args.accum_freq):
                    micro_images_pixel_values = accum_images_pixel_values[j]
                    micro_text_dict = accum_text_dicts[j]
                    with autocast():
                        # Recompute forward pass for this micro-batch
                        model_out = model(image_pixel_values=micro_images_pixel_values, text_dict=micro_text_dict)
                        losses = loss_fn(**model_out) # Calculate loss for this micro-batch
                        total_loss = losses.get("loss")
                        if total_loss is None: # Fallback
                            logging.error(f"Micro-batch {j} loss function did not return a 'loss' key.")
                            total_loss = sum(l for l_name, l in losses.items() if isinstance(l, torch.Tensor) and l.requires_grad and 'loss' in l_name.lower())
                            if not isinstance(total_loss, torch.Tensor) or total_loss.numel() == 0 : total_loss = torch.tensor(0.0, device=device, requires_grad=True if optimizer else False)
                            losses["loss"] = total_loss

                        # Normalize loss for accumulation
                        loss_for_backward = total_loss / args.accum_freq
                    
                    if loss_for_backward.requires_grad:
                        backward(loss_for_backward, scaler) # Accumulate gradients

                accum_images_pixel_values, accum_text_dicts = [], [] # Clear cache

        # Optimizer step (performed once per effective batch, i.e., after accum_freq steps if > 1)
        if optimizer: # Check optimizer existence
            if args.accum_freq == 1 or ((i + 1) % args.accum_freq) == 0:
                if scaler is not None:
                    if args.horovod: # Horovod specific synchronization
                        optimizer.synchronize()
                        scaler.unscale_(optimizer) # Unscale before clipping
                        if args.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                        with optimizer.skip_synchronize(): # Skip sync during step if Horovod handles it
                            scaler.step(optimizer)
                    else: # Standard DDP or single GPU
                        if args.grad_clip_norm is not None:
                            scaler.unscale_(optimizer) # Unscale before clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                        scaler.step(optimizer)
                    scaler.update()
                else: # No scaler
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    optimizer.step()

        # Clamp logit_scale
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100)) # Assuming logit_scale exists

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # current_batch_size is the size of the last processed micro-batch
            current_batch_size = len(images_pixel_values) # Or images_pixel_values.shape[0]
            # num_samples processed in this log step (approx)
            num_samples_processed = batch_count * args.batch_size * args.world_size # Effective batch size
            
            if num_batches_per_epoch > 0 :
                percent_complete = 100.0 * batch_count / num_batches_per_epoch
            elif dataloader.num_samples > 0 and num_samples_processed >= dataloader.num_samples : # Handle edge case if num_batches_per_epoch is 0 but samples exist
                percent_complete = 100.0
            else:
                percent_complete = 0.0 # Default if num_batches_per_epoch is 0
            # Update loss meters using the 'losses' dict from the last processed (micro)batch
            # Note: If accum_freq > 1, 'losses' here would be from the last micro-batch of the effective batch.
            # For more accurate averaged loss across micro-batches, one might need to accumulate them.
            # However, typically logging the last micro-batch's loss components is acceptable.
            if 'losses' in locals(): # Ensure losses dict exists
                for key, val in losses.items():
                     if isinstance(val, torch.Tensor) and val.numel() == 1: # Ensure it's a scalar tensor
                         if key not in losses_m: losses_m[key] = AverageMeter()
                         losses_m[key].update(val.item(), current_batch_size) # Use actual micro-batch size 'n'

            logit_scale_scalar = unwrap_model(model).logit_scale.item()
            
            loss_log_parts = []
            for loss_name, loss_m_val in losses_m.items():
                if loss_m_val.count > 0: # Log only if meter has been updated
                    loss_log_parts.append(f"{loss_name.replace('_', ' ').capitalize()}: {loss_m_val.val:#.5g} ({loss_m_val.avg:#.5g})")
            loss_log = " ".join(loss_log_parts) if loss_log_parts else "No loss components logged."


            samples_per_second = args.batch_size * args.world_size / batch_time_m.val if batch_time_m.val > 0 else 0 # Effective samples/sec
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val if batch_time_m.val > 0 else 0 # Effective samples/sec/gpu
            
            current_lr = optimizer.param_groups[0]['lr'] if optimizer and optimizer.param_groups else 0.0


            logging.info(
                f"Train Epoch: {epoch} [{num_samples_processed:>{sample_digits}}/{dataloader.num_samples if dataloader.num_samples else 'N/A'} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#.2f}/s, {samples_per_second_per_gpu:#.2f}/s/gpu "
                f"LR: {current_lr:.3e} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            log_data = {
                "data_time": data_time_m.avg, # Log average data time
                "batch_time": batch_time_m.avg, # Log average batch time
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": current_lr
            }
            # Add loss components to log_data
            for name, val_meter in losses_m.items():
                if val_meter.count > 0:
                    log_data[name] = val_meter.avg # Log average loss component


            if tb_writer is not None:
                for name_log, val_log in log_data.items():
                    tb_writer.add_scalar("train/" + name_log, val_log, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_wandb = {"step": step}
                for name_log, val_log in log_data.items():
                     log_wandb["train/" + name_log] = val_log
                wandb.log(log_wandb)

            # Important: reset batch_time and data_time after logging for correct averagin
            batch_time_m.reset()
            data_time_m.reset()
            # Reset loss meters as well if their .avg is logged per log_every_n_steps
            # If .val is what's primarily observed for instantaneous loss, resetting might not be needed
            # but for accurate overall epoch avg, it should be done carefully or a separate set of meters used.
            # For simplicity, if logging avg, reset them here.
            for loss_name_meter in losses_m: losses_m[loss_name_meter].reset()


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics # Only master process should evaluate

    # Zero-Shot Evaluation (Placeholder, needs adaptation for DA-SigLIP specifics)
    if 'imagenet-val' in data and args.zeroshot_frequency > 0 and \
            (epoch % args.zeroshot_frequency == 0 or epoch == args.epochs):
        logging.warning("Zero-shot evaluation called, but `zero_shot.py` may need adaptation for DA-SigLIP multi-label output if used for degradation.")
        # zero_shot_metrics = zero_shot_eval(model, data, epoch, args) # This might fail or be incorrect
        # metrics.update(zero_shot_metrics)
        logging.info("Skipping zero-shot evaluation for now unless `zero_shot.py` is confirmed compatible or adapted.")


    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # Ensure val data and dataloader exist
        if data['val'] is None or not hasattr(data['val'], 'dataloader'):
            logging.warning("Validation data or dataloader is not available. Skipping validation.")
            return metrics
        dataloader = data['val'].dataloader
        if len(dataloader) == 0:
            logging.warning("Validation dataloader is empty. Skipping validation.")
            return metrics

        num_samples = 0
        samples_per_val = dataloader.num_samples if dataloader.num_samples else 0 # Handle if num_samples is None
        cumulative_losses = {}

        autocast = get_autocast(args.precision)
        input_dtype = get_input_dtype(args.precision)
        loss_fn_val = create_loss(args) # Get the appropriate loss function

        model.eval()

        # Lists to store all predictions and targets for metric calculation
        all_val_logits_list = []
        all_val_targets_list = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if batch is None:
                    logging.warning(f"Skipping None batch during validation at step {i}.")
                    continue

                images_pixel_values, text_dict = batch
                images_pixel_values = images_pixel_values.to(device=args.device, dtype=input_dtype, non_blocking=True)
                text_dict_device = {}
                for k, v in text_dict.items():
                    if isinstance(v, torch.Tensor):
                        text_dict_device[k] = v.to(device=args.device, non_blocking=True)
                    else:
                        text_dict_device[k] = v

                with autocast():
                    model_out = model(image_pixel_values=images_pixel_values, text_dict=text_dict_device)
                    # Ensure loss_fn_val is callable and model_out has necessary keys
                    if callable(loss_fn_val):
                        losses = loss_fn_val(**model_out)
                    else:
                        logging.error("loss_fn_val is not callable during validation.")
                        losses = {} # Empty dict to avoid further errors


                current_batch_size = images_pixel_values.shape[0]
                for k_loss, v_loss in losses.items():
                     if isinstance(v_loss, torch.Tensor) and v_loss.numel() == 1:
                         if k_loss not in cumulative_losses: cumulative_losses[k_loss] = 0.0
                         cumulative_losses[k_loss] += v_loss.item() * current_batch_size
                num_samples += current_batch_size

                # Accumulate degradation_logits and degradation_target for detailed metrics
                if "degradation_logits" in model_out and model_out["degradation_logits"] is not None:
                    all_val_logits_list.append(model_out["degradation_logits"].cpu())
                if "degradation_target" in model_out and model_out["degradation_target"] is not None:
                    all_val_targets_list.append(model_out["degradation_target"].cpu())
                else: # If target is missing, metrics can't be computed properly for this batch
                    if "degradation_logits" in model_out and model_out["degradation_logits"] is not None:
                         logging.debug(f"Batch {i} in validation has degradation_logits but no degradation_target. Metrics might be partial.")


                if is_master(args) and (i % 100) == 0 and samples_per_val > 0 : # Log progress
                     logging.info(f"Eval Epoch: {epoch} [{num_samples}/{samples_per_val}]")
        
        # Calculate average validation losses
        for k_loss_name in cumulative_losses.keys():
            if num_samples > 0:
                metrics[f"val_{k_loss_name}"] = cumulative_losses[k_loss_name] / num_samples
            else:
                metrics[f"val_{k_loss_name}"] = 0.0 # Avoid division by zero
        metrics["epoch"] = float(epoch) # Ensure epoch is float for JSON serialization
        metrics["num_samples_val"] = float(num_samples)

        # --- Calculate and log multi-label degradation classification metrics ---
        if all_val_logits_list and all_val_targets_list and len(all_val_logits_list) == len(all_val_targets_list):
            try:
                all_val_logits_tensor = torch.cat(all_val_logits_list)
                all_val_targets_tensor = torch.cat(all_val_targets_list)

                val_probs = torch.sigmoid(all_val_logits_tensor.float()).numpy()
                
                # Use threshold from args if available, otherwise default to 0.5
                threshold = getattr(args, 'degradation_classification_threshold', 0.5)
                val_preds = (val_probs >= threshold).astype(int)
                val_targets_numpy = all_val_targets_tensor.numpy().astype(int)

                # Ensure targets and preds are not empty and have compatible shapes
                if val_targets_numpy.size == 0 or val_preds.size == 0 or val_targets_numpy.shape != val_preds.shape:
                    logging.warning("Validation targets or predictions are empty or have mismatched shapes. Skipping detailed metrics.")
                else:
                    metrics["val_degrad_accuracy_exact"] = accuracy_score(val_targets_numpy, val_preds)
                    
                    try:
                        # average_precision_score requires targets to be binary {0, 1} or {-1, 1}
                        # Ensure val_targets_numpy is appropriate.
                        metrics["val_degrad_mAP_macro"] = average_precision_score(val_targets_numpy, val_probs, average="macro")
                        metrics["val_degrad_mAP_weighted"] = average_precision_score(val_targets_numpy, val_probs, average="weighted")
                        # Samples average might be memory intensive if many samples/labels
                        # metrics["val_degrad_mAP_samples"] = average_precision_score(val_targets_numpy, val_probs, average="samples") 
                    except ValueError as e:
                        logging.warning(f"Could not compute mAP for degradation: {e}. Check target format and presence of all classes.")

                    # Calculate P, R, F1 with different averaging
                    for avg_type in ['macro', 'micro', 'weighted', 'samples']:
                        try:
                            p, r, f1, _ = precision_recall_fscore_support(
                                val_targets_numpy, val_preds, average=avg_type, zero_division=0
                            )
                            metrics[f"val_degrad_precision_{avg_type}"] = p
                            metrics[f"val_degrad_recall_{avg_type}"] = r
                            metrics[f"val_degrad_f1_{avg_type}"] = f1
                        except Exception as e:
                            logging.warning(f"Could not compute P, R, F1 with average='{avg_type}': {e}")
                    
                    # Per-class F1, Precision, Recall
                    if hasattr(args, 'degradation_types') and args.degradation_types and len(args.degradation_types) == val_preds.shape[1]:
                        try:
                            p_class, r_class, f1_class, s_class = precision_recall_fscore_support(
                                val_targets_numpy, val_preds, average=None, zero_division=0
                            )
                            for idx, class_name in enumerate(args.degradation_types):
                                class_name_safe = class_name.replace(' ', '_') # Make safe for metric name
                                metrics[f"val_degrad_f1_{class_name_safe}"] = f1_class[idx]
                                metrics[f"val_degrad_precision_{class_name_safe}"] = p_class[idx]
                                metrics[f"val_degrad_recall_{class_name_safe}"] = r_class[idx]
                                metrics[f"val_degrad_support_{class_name_safe}"] = float(s_class[idx])
                        except Exception as e:
                             logging.warning(f"Could not compute per-class P,R,F1 metrics: {e}")
                    
                    logging.info("Successfully calculated multi-label degradation classification metrics for validation.")
            except Exception as e:
                logging.error(f"Error during calculation of multi-label degradation metrics: {e}")

        elif is_master(args): # Only master process logs this if lists were empty
             logging.warning("Could not calculate multi-label degradation metrics for validation as collected logits/targets were empty or mismatched.")


    if not metrics: # If metrics dict is still empty
        return metrics

    # Log all metrics
    log_items = []
    for k_metric, v_metric in metrics.items():
        try:
            log_items.append(f"{k_metric}: {float(v_metric):.4f}") # Ensure v_metric is float for formatting
        except (TypeError, ValueError):
            log_items.append(f"{k_metric}: {v_metric}") # Log as is if cannot be cast to float

    logging.info(f"Eval Epoch: {epoch} " + "\t".join(log_items))


    if args.save_logs: # Save metrics to file and optionally to W&B/TensorBoard
        results_file = os.path.join(args.checkpoint_path, "results.jsonl")
        try:
            with open(results_file, "a+") as f:
                # Ensure all metric values are JSON serializable (e.g. convert numpy types to native Python types)
                serializable_metrics = {}
                for k_json, v_json in metrics.items():
                    if isinstance(v_json, (np.generic, np.ndarray)):
                        serializable_metrics[k_json] = v_json.item() if v_json.size == 1 else v_json.tolist()
                    elif isinstance(v_json, torch.Tensor):
                        serializable_metrics[k_json] = v_json.item() if v_json.numel() == 1 else v_json.tolist()
                    else:
                        serializable_metrics[k_json] = v_json
                f.write(json.dumps(serializable_metrics))
                f.write("\n")
        except Exception as e:
            logging.error(f"Failed to write validation results to {results_file}: {e}")

        if tb_writer is not None:
            for name_metric, val_metric in metrics.items():
                 try:
                     # Ensure key for TensorBoard is prefixed appropriately
                     log_key_tb = name_metric if name_metric.startswith("val_") else f"val/{name_metric}"
                     tb_writer.add_scalar(log_key_tb, float(val_metric), epoch)
                 except (TypeError, ValueError):
                     logging.debug(f"Could not log metric '{name_metric}' to TensorBoard (value: {val_metric}).")


        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb_metrics = {}
            for name_metric_wb, val_metric_wb in metrics.items():
                # Ensure W&B keys are simple strings, and values are appropriate
                key_wb = str(name_metric_wb).replace('/', '_') # Sanitize key
                try:
                    if isinstance(val_metric_wb, (np.generic, np.ndarray)):
                        wandb_metrics[key_wb] = val_metric_wb.item() if val_metric_wb.size == 1 else val_metric_wb.tolist()
                    elif isinstance(val_metric_wb, torch.Tensor):
                         wandb_metrics[key_wb] = val_metric_wb.item() if val_metric_wb.numel() == 1 else val_metric_wb.tolist()
                    else:
                        wandb_metrics[key_wb] = val_metric_wb
                except Exception as e_wb_val:
                    logging.debug(f"Could not prepare metric '{key_wb}' for W&B (value: {val_metric_wb}): {e_wb_val}")

            if 'epoch' not in wandb_metrics: wandb_metrics['epoch'] = epoch # Ensure epoch is logged
            try:
                wandb.log(wandb_metrics)
            except Exception as e_wandb:
                logging.error(f"Failed to log metrics to W&B: {e_wandb}")
    return metrics