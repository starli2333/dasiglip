#!/bin/bash

# 脚本用于在单个 GPU 上训练 DA-SigLIP 控制器

# --- 可配置参数 ---
# !! 修改为你选择的 SigLIP 模型标识符 (前面加上 dasiglip_ 前缀) !!
MODEL_NAME="dasiglip_google/siglip-base-patch16-224"

# !! 修改为你预训练的 *控制器* 权重的路径 (如果需要恢复或继续训练) !!
# 如果从头开始训练控制器，请将其设置为空字符串: PRETRAINED_CONTROLLER=""
PRETRAINED_CONTROLLER="" # 例如: "logs/dasiglip-base-p16-224-YYYYMMDD_HHMMSS/checkpoints/epoch_X.pt"

# !! 修改为你的训练数据路径 (确保 CSV 包含 'filepath', 'caption', 'degradation' 列) !!
# !! 已更新为使用您提供的 daclip_train.csv，请确保路径正确 !!
TRAIN_DATA="datasets/universal/daclip_train.csv" # <--- 请确保此文件与脚本在同一目录或提供完整路径
# !! 修改为你的验证数据路径 (确保 CSV 包含 'filepath', 'caption', 'degradation' 列) !!
VAL_DATA="datasets/universal/daclip_val.csv" # <--- 请替换为您的验证 CSV 路径

# !! 确保这些键名与您的 CSV 文件中的列名完全匹配 !!
CSV_IMG_KEY="filepath"
CSV_CAPTION_KEY="caption" # 假设 'caption' 列现在只包含图像的纯文本描述
CSV_DEGRADATION_KEY="degradation" # 'degradation' 列包含单一退化类型字符串

LOG_DIR="./logs_dasiglip" # 建议为 DA-SigLIP 使用新的日志目录
EXP_NAME="${MODEL_NAME//\//_}-$(date +%Y%m%d_%H%M%S)" # 基于模型名称和时间戳生成实验名称

BATCH_SIZE=128      # 单 GPU 批次大小 (根据显存调整, SigLIP base 可能需要减小)
EPOCHS=50           # 训练轮数 (训练控制器可能不需要太多轮)
LR=5e-5             # 学习率 (针对 SigLIP base 控制器微调，可能需要调整)
WD=0.1              # 权重衰减
WARMUP_STEPS=500    # 预热步数 (可以适当减少)
PRECISION="amp"     # 混合精度: "amp", "amp_bf16", "fp32"

# DA-SigLIP 特有的损失权重
BCE_LOSS_WEIGHT=1.0             # BCE 退化分类损失的权重
DEGRAD_CONTRASTIVE_WEIGHT=0.0   # (可选) 退化特征对比损失的权重 (设为0则禁用)

WORKERS=8          # 数据加载进程数
SAVE_FREQUENCY=5   # Checkpoint 保存频率 (每多少个 epoch 保存一次)
VAL_FREQUENCY=1    # 验证频率 (每多少个 epoch 验证一次)
LOG_EVERY_N_STEPS=100 # 每 N 步记录一次日志

REPORT_TO="tensorboard" # 日志后端 ('tensorboard', 'wandb', 或者两者 'wandb,tensorboard')
# WANDB_PROJECT_NAME="dasiglip-controller-training" # 如果使用 wandb

# --- 执行训练 ---
# 使用 python -m training.main 确保模块可以被正确找到
# 确保当前工作目录是 da-clip/src/
cd "$(dirname "$0")" # 切换到脚本所在目录 (即 da-clip/src/)

echo "开始训练 DA-SigLIP 控制器: ${EXP_NAME}"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "验证数据: ${VAL_DATA}"
echo "CSV 图像键: ${CSV_IMG_KEY}, 标题键: ${CSV_CAPTION_KEY}, 退化键: ${CSV_DEGRADATION_KEY}"
echo "批次大小: ${BATCH_SIZE}, 学习率: ${LR}, 轮数: ${EPOCHS}"

python -m training.main \
    --model "${MODEL_NAME}" \
    --pretrained "${PRETRAINED_CONTROLLER}" \
    --train-data="${TRAIN_DATA}" \
    --val-data="${VAL_DATA}" \
    --dataset-type csv \
    --csv-img-key "${CSV_IMG_KEY}" \
    --csv-caption-key "${CSV_CAPTION_KEY}" \
    --csv-degradation-key "${CSV_DEGRADATION_KEY}" \
    --csv-separator ","
    --logs "${LOG_DIR}" \
    --name "${EXP_NAME}" \
    --workers ${WORKERS} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --wd ${WD} \
    --warmup ${WARMUP_STEPS} \
    --precision ${PRECISION} \
    --bce-loss-weight ${BCE_LOSS_WEIGHT} \
    --degrad-contrastive-weight ${DEGRAD_CONTRASTIVE_WEIGHT} \
    --save-frequency ${SAVE_FREQUENCY} \
    --val-frequency ${VAL_FREQUENCY} \
    --log-every-n-steps ${LOG_EVERY_N_STEPS} \
    --report-to "${REPORT_TO}" \
    # --wandb-project-name "${WANDB_PROJECT_NAME}" # 如果使用 wandb
    # --resume latest # 如果需要从最新的 checkpoint 恢复
    # --lock-image # 对于控制器训练，通常不需要冻结基础 SigLIP 视觉模型 (DaSiglipModel 内部会处理)
    # --grad-checkpointing # 如果显存不足，可以尝试启用梯度检查点

echo "DA-SigLIP 控制器训练脚本执行完毕: ${EXP_NAME}"

