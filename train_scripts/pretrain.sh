#!/bin/bash

# # 检查是否传入了输出文件夹名字作为参数
# if [ "$#" -ne 1 ]; then
#     echo "Error: Output folder name is required."
#     echo "Usage: $0 <output-folder-name>"
#     exit 1
# fi

# 赋值操作
OUTPUT_FOLDER_NAME="rho1"

MODEL_NAME="/cephfs/shared/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773"
DATA_PATH="/cephfs/shared/yanghanbin/data/coq/train_data_proof_state_transition_outcomes_v6.jsonl"
EVAL_PATH="/cephfs/shared/yanghanbin/data/coq/test_data_proof_state_transition_outcomes_v6.jsonl"
# DATA_PATH='/cephfs/shared/luyanzhen/data/train_ps_v6.jsonl'
# EVAL_PATH='/cephfs/shared/luyanzhen/data/test_ps_v6.jsonl'

CURRENT_DATETIME=$(date "+%Y-%m-%d-%H-%M-%S")

SAVE_PATH="/cephfs/shared/luyanzhen/coq-pretrain/checkpoints/${OUTPUT_FOLDER_NAME}-${CURRENT_DATETIME}"

LOG_FILE="${SAVE_PATH}/training.log"

mkdir -p "$SAVE_PATH"

{
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_IB_TIMEOUT=21
  export NCCL_DEBUG=INFO
  export NCCL_TIMEOUT=3600
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export PYTHONPATH=$PYTHONPATH:/cephfs/shared/luyanzhen/codes/rho1

  echo "Starting training at $(date)"
  echo "Saving logs to $LOG_FILE"

  deepspeed pretrain/train.py \
    --deepspeed ds_config/ds_config_stage2.json \
    --pretrained_dir "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --eval_path "$EVAL_PATH" \
    --output_dir "$SAVE_PATH" \
    --model_max_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --weight_decay 0.02 \
    --num_train_epochs 2 \
    --scheduler "tri_stage" \
    --warmup 0.1 \
    --weight_decay 0.001 \
    --decay 0.5 \
    --learning_rate 2e-4 \
    --zero_stage 1 \
    --offload_adam false \
    --offload_params false \
    --gradient_checkpointing \
    --wandb_enabled \
    --wandb_project_name "pretrain_coq" \

  echo "Training finished at $(date)"

} 2>&1 | tee -a "$LOG_FILE"