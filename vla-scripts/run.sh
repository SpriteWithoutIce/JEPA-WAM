LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="/home/jwhe/linyihan/datasets/modified_libero_rlds"
QWEN_PATH="/home/jwhe/linyihan/CKPT/Qwen2.5-0.5B"
VJEPA_CKPT="/home/jwhe/linyihan/CKPT/vjepa2_1_vitl_384.pt"
RUNS_DIR="./runs"
# CUDA_VISIBLE_DEVICES=2,3

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
    --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90 \
    --vla.base_vlm /home/jwhe/linyihan/prismatic-vlm/runs/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}" \
    --llm_checkpoint_path "${QWEN_PATH}" \
    --data_root_dir "${LIBERO_DATA}" \
    --run_root_dir ./runs \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 64 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 2e-4 \
    --vla.max_steps 45000 \
    --vla.shuffle_buffer_size 10000 \
    --vla.use_lora True \
    --vla.freeze_vision_backbone True \
    --vla.lora_rank 32 \
    --vla.lora_alpha 64 \
    --vla.lora_dropout 0.1 \
    --vla.action_head_type l1 \
    --vla.future_obs_window_size 8 \
    --vla.use_aux_head True \
    --use_wrist_image False \
    --save_interval 5000 \
    --seed 7 \
    --use_wandb True \
    --debug_batch_shapes False \
    2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"

