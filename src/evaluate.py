# evaluate_new.py
import os
import sys
import torch
import logging
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
from typing import Tuple, List, Dict, Optional

# --- 动态确定项目根目录 ---
_CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
_CURRENT_SCRIPT_DIR = os.path.dirname(_CURRENT_SCRIPT_PATH) # src/ 目录
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)       # da-siglip/ (项目根目录)

# --- (可选但推荐) 将项目根目录和 src 目录添加到 sys.path ---
# 这有助于确保 open_clip 等本地模块能被正确导入
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)
# if _CURRENT_SCRIPT_DIR not in sys.path: # 如果 open_clip 在 src/open_clip
#     sys.path.insert(0, _CURRENT_SCRIPT_DIR)


try:
    import open_clip
    from open_clip.dasiglip_model import DaSiglipModel
    from open_clip.constants import DEGRADATION_TYPES, DEGRADATION_TO_ID, NUM_DEGRADATION_TYPES
    from open_clip.factory import load_checkpoint
    from transformers import AutoProcessor
except ImportError as e:
    logging.error(f"Failed to import open_clip or transformers components. Ensure PYTHONPATH is set correctly or the package is installed. Error: {e}")
    sys.exit(1)

# --- 配置参数 (所有路径都将视为相对于项目根目录) ---
# 请将 CHECKPOINT_PATH_REL_TO_ROOT 更新为您的实际 checkpoint 路径
CHECKPOINT_PATH_REL_TO_ROOT = 'logs_dasiglip/dasiglip_google_siglip-base-patch16-224-20250512_141842/checkpoints/epoch_50.pt'
BASE_SIGLIP_MODEL = "google/siglip-base-patch16-224"

# CSV 文件路径 (相对于项目根目录)
# 例如, 如果 daclip_val.csv 在 src/datasets/ 目录下:
VAL_CSV_PATH_REL_TO_ROOT = 'datasets/universal/daclip_val.csv'

THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- 结束配置 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_degradations_from_csv_title(title_str: str) -> Tuple[np.ndarray, int]:
    """
    从 CSV 的 title 列解析退化标签并转换为多热编码，同时返回退化数量。
    预期格式 "图像内容文本描述:退化类型1_退化类型2_..." 或 "图像内容文本描述:退化类型1"
    """
    target_vector = np.zeros(NUM_DEGRADATION_TYPES, dtype=int)
    num_true_degradations = 0
    
    # 只分割一次，以处理描述中可能也包含冒号的情况
    parts = title_str.split(':', 1) 
    
    if len(parts) > 1 and parts[1].strip(): # 确保冒号后有内容
        degradation_part = parts[1].strip()
        # 多个退化类型用下划线 '_' 分隔
        # 单个退化类型则没有下划线 (e.g., "motion-blurry")
        deg_names_in_title = degradation_part.split('_')
        
        for deg_name in deg_names_in_title:
            clean_deg_name = deg_name.strip()
            if not clean_deg_name: # 跳过空的退化名 (例如 "type1__type2" 中的空隙)
                continue
            if clean_deg_name in DEGRADATION_TO_ID:
                target_vector[DEGRADATION_TO_ID[clean_deg_name]] = 1
                num_true_degradations += 1
            else:
                logging.debug(f"Unknown degradation string '{clean_deg_name}' parsed from title: '{title_str}'")
    # else:
        # logging.debug(f"No degradation part found after colon, or it's empty, in title: '{title_str}'")
        
    return target_vector, num_true_degradations

def calculate_and_log_metrics(targets: np.ndarray, preds: np.ndarray, group_name: str, threshold_used: float):
    """计算并记录给定目标和预测的指标"""
    if len(targets) == 0:
        logging.info(f"No samples found for group: '{group_name}'. Skipping metrics calculation.")
        return

    logging.info(f"\n--- Evaluation Summary for '{group_name}' (Support: {len(targets)} samples, Threshold: {threshold_used}) ---")
    try:
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(targets, preds, average='micro', zero_division=0)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
        subset_acc = accuracy_score(targets, preds)
        h_loss = hamming_loss(targets, preds)

        logging.info(f"  Subset Accuracy (Exact Match): {subset_acc:.4f}")
        logging.info(f"  Hamming Loss: {h_loss:.4f}")
        logging.info(f"  Micro Average ---- Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
        logging.info(f"  Macro Average ---- Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")

        logging.info("\n  Per-Class F1, Precision, Recall:")
        for i, deg_type in enumerate(DEGRADATION_TYPES):
            class_targets = targets[:, i]
            class_preds = preds[:, i]
            class_support = np.sum(class_targets)

            if class_support == 0 and np.sum(class_preds) == 0:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                p, r, f1, _ = precision_recall_fscore_support(class_targets, class_preds, average='binary', pos_label=1, zero_division=0)
            
            logging.info(f"    {deg_type:<20}: P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} (Support: {int(class_support)})")

    except Exception as e:
        logging.error(f"Could not calculate sklearn multi-label metrics for {group_name}: {e}")
        import traceback
        traceback.print_exc()

def evaluate_main(checkpoint_rel_path, base_model_name, val_csv_rel_path, threshold=0.5, device="cuda"):
    project_root = _CURRENT_SCRIPT_DIR

    actual_checkpoint_path = os.path.join(project_root, checkpoint_rel_path)
    actual_val_csv_path = os.path.join(project_root, val_csv_rel_path)

    if not os.path.exists(actual_checkpoint_path):
        logging.error(f"Checkpoint not found at: {actual_checkpoint_path} (resolved from relative: {checkpoint_rel_path})")
        return
    if not os.path.exists(actual_val_csv_path):
        logging.error(f"Validation CSV not found at: {actual_val_csv_path} (resolved from relative: {val_csv_rel_path})")
        return

    logging.info(f"Loading base SigLIP model '{base_model_name}' and processor...")
    try:
        processor = AutoProcessor.from_pretrained(base_model_name)
    except Exception as e:
        logging.error(f"Failed to load base SigLIP processor: {e}")
        return

    logging.info(f"Creating DA-SigLIP model structure...")
    model = DaSiglipModel(
        model_name=base_model_name,
        num_degradation_types=NUM_DEGRADATION_TYPES,
        freeze_base=True # 评估时基础模型应冻结
        # controller_transformer_depth 等参数应与训练时一致，如果 DaSiglipModel 构造函数需要它们
    ).to(device)

    logging.info(f"Loading controller checkpoint from: {actual_checkpoint_path}")
    try:
        load_checkpoint(model, actual_checkpoint_path, strict=False) # 使用 open_clip.factory 中的 load_checkpoint
        logging.info("Controller checkpoint loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load controller checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    model.eval()

    all_preds_overall, all_targets_overall = [], []
    all_preds_single_gt, all_targets_single_gt = [], []
    all_preds_multi_gt, all_targets_multi_gt = [], []

    logging.info(f"Loading validation data from CSV: {actual_val_csv_path}")
    try:
        df_val = pd.read_csv(actual_val_csv_path, sep='\t') # 假设CSV是制表符分隔
    except Exception as e:
        logging.error(f"Failed to read CSV file {actual_val_csv_path}. Error: {e}")
        return

    logging.info(f"Found {len(df_val)} samples in CSV. Starting evaluation...")

    for index, row in tqdm(df_val.iterrows(), total=df_val.shape[0], desc="Evaluating from CSV"):
        relative_img_path_from_csv = str(row['filepath']).strip()
        title_str = str(row['title']).strip()

        img_path = os.path.join(project_root, relative_img_path_from_csv)

        if not os.path.exists(img_path):
            logging.warning(f"Image path not found: {img_path}. Skipping.")
            continue
        try:
            pil_image = Image.open(img_path).convert('RGB')
            image_inputs = processor(images=pil_image, return_tensors="pt").to(device)
        except Exception as e:
            logging.warning(f"Could not process image {img_path}. Error: {e}. Skipping.")
            continue

        target_vector, num_true_degradations = parse_degradations_from_csv_title(title_str)

        degradation_logits = None
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
            outputs = model.encode_image(image_inputs['pixel_values'], control=True, normalize=False)
            
            # 处理 model.encode_image 返回元组的情况 (根据上次的错误)
            if isinstance(outputs, tuple):
                if len(outputs) >= 3: # 假设 degradation_logits 是第三个元素
                    degradation_logits = outputs[2]
                else:
                    logging.error(f"encode_image returned a tuple with too few elements ({len(outputs)}) for {img_path}. Skipping.")
                    continue
            elif isinstance(outputs, dict) and 'degradation_logits' in outputs:
                degradation_logits = outputs['degradation_logits']
            else: # 如果不是预期的元组或字典，或者字典中没有 'degradation_logits'
                logging.error(f"Unexpected output format or missing 'degradation_logits' from model.encode_image for {img_path}. Type: {type(outputs)}. Skipping.")
                continue
            
            if degradation_logits is None: # 再次确认
                 logging.warning(f"Failed to extract degradation_logits for {img_path}. Skipping.")
                 continue

            probabilities = torch.sigmoid(degradation_logits).squeeze(0).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)

            all_preds_overall.append(predictions)
            all_targets_overall.append(target_vector)

            if num_true_degradations == 1:
                all_preds_single_gt.append(predictions)
                all_targets_single_gt.append(target_vector)
            elif num_true_degradations > 1:
                all_preds_multi_gt.append(predictions)
                all_targets_multi_gt.append(target_vector)
            # elif num_true_degradations == 0:
            #     logging.debug(f"Sample {img_path} has no degradation labels in CSV title.")


    # --- 计算并记录指标 ---
    calculate_and_log_metrics(np.array(all_targets_overall), np.array(all_preds_overall), "Overall Dataset", threshold)
    calculate_and_log_metrics(np.array(all_targets_single_gt), np.array(all_preds_single_gt), "Single-Type Ground Truth Images", threshold)
    calculate_and_log_metrics(np.array(all_targets_multi_gt), np.array(all_preds_multi_gt), "Multi-Type Ground Truth Images", threshold)

    logging.info("Evaluation finished.")

if __name__ == "__main__":
    if not CHECKPOINT_PATH_REL_TO_ROOT or "your_dasiglip_experiment" in CHECKPOINT_PATH_REL_TO_ROOT :
        logging.error("CRITICAL: CHECKPOINT_PATH_REL_TO_ROOT is not set or is set to the default placeholder. Please update it in the script.")
    elif not VAL_CSV_PATH_REL_TO_ROOT:
        logging.error("CRITICAL: VAL_CSV_PATH_REL_TO_ROOT is not set. Please update it in the script.")
    else:
        evaluate_main(
            checkpoint_rel_path=CHECKPOINT_PATH_REL_TO_ROOT,
            base_model_name=BASE_SIGLIP_MODEL,
            val_csv_rel_path=VAL_CSV_PATH_REL_TO_ROOT,
            threshold=THRESHOLD,
            device=DEVICE
        )