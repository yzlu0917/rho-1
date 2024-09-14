#!/bin/bash

# 检查是否传入了输出文件夹名字作为参数
if [ "$#" -ne 1 ]; then
    echo "Error: Output folder name is required."
    echo "Usage: $0 <output-folder-name>"
    exit 1
fi

# 赋值操作
OUTPUT_FOLDER_NAME="$1"

MODEL_NAME="/cephfs/shared/hf_cache/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"
DATA_PATH="/cephfs/shared/lichao/pretrain/data/pretrain_data/train"
EVAL_PATH="/cephfs/shared/lichao/pretrain/data/pretrain_data/test"
ANNEAL_PATH="/cephfs/shared/lichao/pretrain/data/pretrain_data/anneal"

CURRENT_DATETIME=$(date "+%Y-%m-%d-%H-%M-%S")

SAVE_PATH="/cephfs/shared/lichao/pretrain/checkpoints/${OUTPUT_FOLDER_NAME}-${CURRENT_DATETIME}"

LOG_FILE="${SAVE_PATH}/training.log"

mkdir -p "$SAVE_PATH"

{
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_IB_TIMEOUT=21
  export NCCL_DEBUG=INFO
  export NCCL_TIMEOUT=3600
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export PYTHONPATH=$PYTHONPATH:/cephfs/lichao/codes/edge-llm

  echo "Starting training at $(date)"
  echo "Saving logs to $LOG_FILE"

  deepspeed --force_multi --hostfile hostfile pretrain/pretrain.py \
    --deepspeed ds_config/ds_config_stage2.json \
    --model_name_or_path "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --eval_path "$EVAL_PATH" \
    --anneal_path "$ANNEAL_PATH" \
    --output_dir "$SAVE_PATH" \
    --model_max_length 2048 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --weight_decay 0.02 \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --lr_scheduler_type "tri_stage" \
    --warmup 0.1 \
    --learning_rate 2e-4 \
    --min_lr_rate 0.01 \
    --lora_r 0 \
    --zero_stage 1 \
    --offload_adam false \
    --offload_params false \
    --gradient_checkpointing \
    --wandb_enabled \
    --wandb_project_name "pretrain" \
    --resume_from_checkpoint "/cephfs/shared/lichao/pretrain/checkpoints/math_1_5b_mixed_5-2024-08-20-11-33-06"

  echo "Training finished at $(date)"

} 2>&1 | tee -a "$LOG_FILE"