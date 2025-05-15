# -*- coding: utf-8 -*-
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
import torch.nn as nn # <--- IMPORT ADDED HERE
from torch.cuda.amp import GradScaler
from transformers import AutoProcessor

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# Import from open_clip and training package
from open_clip import create_model_and_transforms, get_tokenizer, create_loss # trace_model removed as it's disabled
from open_clip.factory import DASIGLIP_HF_PREFIX, create_model # Import create_model for base SigLIP
from open_clip.dasiglip_model import DaSiglipModel # Import the new model class

from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args # Uses the modified parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate # train_one_epoch needs modification
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync


import sys
import os

print("MAIN.PY ---- SCRIPT STARTED ----", flush=True)

print("--- Python sys.path ---")
for p in sys.path:
    print(p)
print("-----------------------")
try:
    import open_clip
    print(f"--- open_clip imported from: {open_clip.__file__} ---")
    # 你甚至可以打印 factory.py 的路径
    if hasattr(open_clip, 'factory'):
        print(f"--- open_clip.factory imported from: {open_clip.factory.__file__} ---")
except ImportError:
    print("--- ERROR: Could not import open_clip ---")

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # Keep original implementation
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(result) # Optional: for debugging s3 ls output
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def unwrap_model(model): # Define helper function globally in this file
    if hasattr(model, 'module'): # Check for DDP or DataParallel wrapper
        return model.module
    else:
        return model

def main(initial_cli_args): # Renamed to avoid conflict with 'args' variable
    args = parse_args(initial_cli_args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        # Use the full model name including prefix for logging
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest and args.resume is None: # Check resume is not set
            logging.warning(f"Log file {args.log_path} already exists. Appending logs.")


    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname: # Ensure dirname is not empty
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                logging.error('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1 # Or sys.exit(-1)
            if args.remote_sync_protocol != 's3': # Example, adjust if other protocols are supported
                logging.error('Error. Sync protocol not supported when using resume latest.')
                return -1 # Or sys.exit(-1)
        if is_master(args):
            if args.save_most_recent:
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    resume_from = None
            else:
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from


    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        result = remote_sync( # Initial sync
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Initial remote sync successful.')
        else:
            logging.info('Error: Initial remote sync failed. Exiting.')
            return -1 # Or sys.exit(-1)
        # Start background sync process
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if remote_sync_process: # Check if process started successfully
            remote_sync_process.start()
        else:
            logging.warning("Failed to start remote sync process.")


    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # --- Model Creation ---
    model = None
    processor = None # For SigLIP processor
    tokenizer = None # For non-SigLIP tokenizer

    if args.model.startswith(DASIGLIP_HF_PREFIX):
        logging.info(f"Creating DA-SigLIP model: {args.model}")
        model = DaSiglipModel(
            model_name=args.model[len(DASIGLIP_HF_PREFIX):],
            num_degradation_types=args.num_degradation_types,
            controller_transformer_depth=getattr(args, 'dasiglip_controller_depth', None),
            freeze_base=getattr(args, 'dasiglip_freeze_base', True)
        )
        try:
            current_cache_dir = args.cache_dir if hasattr(args, 'cache_dir') else None
            processor = AutoProcessor.from_pretrained(
                args.model[len(DASIGLIP_HF_PREFIX):],
                cache_dir=current_cache_dir
            )
            tokenizer = processor.tokenizer
            preprocess_train = processor.image_processor
            preprocess_val = processor.image_processor
            logging.info(f"Loaded SigLIP processor for {args.model[len(DASIGLIP_HF_PREFIX):]}")
            # Assign mean/std to model if available in processor and model expects it
            if hasattr(model, 'visual') and hasattr(processor.image_processor, 'image_mean') and hasattr(model.visual, 'image_mean'):
                 model.visual.image_mean = processor.image_processor.image_mean
                 model.visual.image_std = processor.image_processor.image_std
        except Exception as e:
            logging.error(f"Failed to load SigLIP processor. Error: {e}")
            raise e
    else:
        # --- Original logic for creating non-DA-SigLIP models ---
        logging.info(f"Creating standard model: {args.model}")
        current_cache_dir = args.cache_dir if hasattr(args, 'cache_dir') else None
        
    print("MAIN.PY ---- ABOUT TO CALL create_model_and_transforms ----", flush=True)
    try:
        model, preprocess_train, preprocess_val, tokenizer = create_model_and_transforms(
            args.model,
            args.pretrained,
            # ... 其他参数 ...
        )
        print("MAIN.PY ---- SUCCESSFULLY CALLED create_model_and_transforms ----", flush=True)
    except Exception as e:
        print(f"MAIN.PY ---- ERROR DURING OR BEFORE create_model_and_transforms CALL ----", flush=True)
        import traceback
        traceback.print_exc() # 打印完整的错误堆栈
        sys.exit(1)
        
    model, preprocess_train, preprocess_val, tokenizer = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=getattr(args, 'force_quick_gelu', False),
        force_custom_text=getattr(args, 'force_custom_text', False),
        force_patch_dropout=getattr(args, 'force_patch_dropout', None),
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        cache_dir=current_cache_dir,
        output_dict=True,
        dasiglip_num_degrad_types=args.num_degradation_types, # Pass through
        dasiglip_controller_depth=getattr(args, 'dasiglip_controller_depth', None), # Pass through
        dasiglip_freeze_base=getattr(args, 'dasiglip_freeze_base', True), # Pass through
    )

    dist_model = None # Distillation disabled

    random_seed(args.seed, args.rank)

    # --- Removed trace logic as it's not a core requirement and can cause issues ---
    # if hasattr(args, 'trace') and args.trace:
    #     try:
    #         # model = trace_model(model, batch_size=args.batch_size, device=device) # trace_model might need update
    #         logging.info("Model tracing skipped for DA-SigLIP.")
    #     except Exception as e:
    #         logging.warning(f"Model tracing failed: {e}. Proceeding without tracing.")

    if args.lock_image:
        model_to_lock = unwrap_model(model) # Get the actual model module
        if isinstance(model_to_lock, DaSiglipModel):
             model_to_lock.lock_siglip() # This freezes base visual and text
             logging.info("Locked base SigLIP visual and text models.")
        elif hasattr(model_to_lock, 'lock_image_tower'): # Check for non-DA-SigLIP models
             model_to_lock.lock_image_tower(
                 unlocked_groups=args.lock_image_unlocked_groups,
                 freeze_bn_stats=args.lock_image_freeze_bn_stats)
             logging.info("Locked standard image tower.")

    if args.grad_checkpointing:
        model_to_checkpoint = unwrap_model(model)
        if hasattr(model_to_checkpoint, 'set_grad_checkpointing'):
            model_to_checkpoint.set_grad_checkpointing() # Call on the module
            # logging.info("Enabled gradient checkpointing.") # Already logged in DaSiglipModel
        else:
            logging.warning("Model does not support set_grad_checkpointing method.")


    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name_attr in sorted(vars(args)):
                val = getattr(args, name_attr)
                logging.info(f"  {name_attr}: {val}")
                f.write(f"{name_attr}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        logging.info("Model wrapped in DistributedDataParallel.")

    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        params_to_optimize = []
        model_module = unwrap_model(model)

        if isinstance(model_module, DaSiglipModel):
            params_to_optimize.extend(model_module.visual_control.parameters())
            # Check if logit_scale is a Parameter and requires grad
            if hasattr(model_module, 'logit_scale') and isinstance(model_module.logit_scale, nn.Parameter) and model_module.logit_scale.requires_grad:
                 params_to_optimize.append(model_module.logit_scale)
            num_trainable_params = sum(p.numel() for p in params_to_optimize if p.requires_grad)
            logging.info(f"Optimizing DA-SigLIP controller ({num_trainable_params:,}) parameters.")

            if not params_to_optimize:
                 logging.warning("No parameters found to optimize for DaSiglipModel. Check requires_grad settings.")
            else:
                optimizer = optim.AdamW(
                     params_to_optimize,
                     lr=args.lr,
                     betas=(args.beta1, args.beta2),
                     eps=args.eps,
                     weight_decay=args.wd
                )
        else: # Original optimizer setup for non-DA-SigLIP models
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters()) # Use model directly if not DDP, or model_module if DDP
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if not (gain_or_bias_params or rest_params):
                 logging.warning("No parameters found to optimize for standard model.")
            else:
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            logging.info("Optimizing standard model parameters.")


        if args.horovod and optimizer is not None: # Check if optimizer was created
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            incompatible_keys = open_clip.factory.load_checkpoint(model, args.resume, strict=False)
            logging.info(f"Resumed model checkpoint '{args.resume}' (epoch {start_epoch}). Incompatible keys: {incompatible_keys}")
            if optimizer is not None and 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    logging.info("Resumed optimizer state.")
                except Exception as e:
                    logging.warning(f"Could not resume optimizer state: {e}. Starting optimizer from scratch.")
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                logging.info("Resumed scaler state.")
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            open_clip.factory.load_checkpoint(model, args.resume, strict=True)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    data_input_processor = processor if args.model.startswith(DASIGLIP_HF_PREFIX) else (preprocess_train, preprocess_val)
    # Pass tokenizer explicitly for non-SigLIP models if it was created
    current_tokenizer = tokenizer if not args.model.startswith(DASIGLIP_HF_PREFIX) and tokenizer is not None else None
    data = get_data(args, data_input_processor, epoch=start_epoch, tokenizer=current_tokenizer)
    assert len(data), 'At least one train or eval dataset must be specified.'

    scheduler = None
    if 'train' in data and optimizer is not None: # Check optimizer exists
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            sys.exit(1)
        logging.info(f"Using {args.lr_scheduler} LR scheduler.")

    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples if 'train' in data else 0
        args.val_sz = data["val"].dataloader.num_samples if 'val' in data else 0
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if resume_latest else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        if args.log_path and os.path.exists(args.log_path):
             wandb.save(args.log_path)
        params_file_path = os.path.join(args.logs, args.name, "params.txt")
        if os.path.exists(params_file_path):
            wandb.save(params_file_path)
        logging.debug('Finished loading wandb.')

    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(model)

    if 'train' not in data: # If only evaluation
        if optimizer is None: # Ensure optimizer is None if no training data
             logging.info("No training data. Running evaluation only.")
        evaluate(model, data, start_epoch, args, writer)
        return

    if optimizer is None: # Should not happen if train_data is present, but as a safeguard
        logging.error("Optimizer not initialized, but training data is present. Exiting.")
        return -1


    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        if args.save_logs:
            model_module_to_save = unwrap_model(model)
            state_dict_to_save = {}
            if isinstance(model_module_to_save, DaSiglipModel):
                 state_dict_to_save = model_module_to_save.visual_control.state_dict()
                 if hasattr(model_module_to_save, 'logit_scale') and isinstance(model_module_to_save.logit_scale, nn.Parameter):
                     state_dict_to_save['logit_scale'] = model_module_to_save.logit_scale.data.clone()
                 logging.info("Saving DA-SigLIP controller state dict.")
            else:
                 state_dict_to_save = model_module_to_save.state_dict()
                 logging.info("Saving standard model state dict.")

            if not state_dict_to_save: # Check if state_dict is empty
                 logging.warning("State dict to save is empty. Skipping checkpoint save for this epoch.")
                 continue


            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": state_dict_to_save,
            }
            # Only save optimizer and scaler if they exist (i.e., during training)
            if optimizer is not None:
                checkpoint_dict["optimizer"] = optimizer.state_dict()
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()


            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    logging.info(f"Deleting previous checkpoint: {previous_checkpoint}")
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns # Local import
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        logging.error(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    logging.info(f"Copying codebase to {new_code_path}")
    # Determine project root more reliably. Assume main.py is in src/training/
    # Project root would be two levels up from main.py's directory.
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    copytree(project_root, new_code_path, ignore=ignore_patterns(
        'log', 'logs', 'wandb', 'experiments', 'results', '__pycache__',
        '*.pyc', '*.pt', '*.pth', '.git', '.vscode', '.idea', # Common ignores
        'data', 'datasets', # Exclude large data directories
        'pretrained', # Exclude pretrained model directories
        '.env', 'venv', # Exclude virtual environments
        'autodl-tmp' # Common cloud storage mount
        ))
    logging.info("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])

