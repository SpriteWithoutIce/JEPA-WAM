LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="/home/jwhe/linyihan/datasets/modified_libero_rlds"
QWEN_PATH="/home/jwhe/linyihan/CKPT/Qwen2.5-0.5B"
VJEPA_CKPT="/home/jwhe/linyihan/CKPT/vjepa2-vitl-fpc64-256"
RUNS_DIR="./runs"

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
    --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}" \
    --llm_checkpoint_path "${QWEN_PATH}" \
    --data_root_dir "${LIBERO_DATA}" \
    --run_root_dir "${RUNS_DIR}" \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 64 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 2e-4 \
    --vla.max_steps 60000 \
    --vla.action_head_type l1 \
    --use_wrist_image True \
    --debug_batch_shapes True \
    --debug_embedding_viz_interval 500 \
    --debug_embedding_viz_samples 1 \
    --vla.shuffle_buffer_size 10000 \
    --save_interval 5000 \
    --seed 7 \
    --use_wandb True \
    --pretrained_checkpoint /home/jwhe/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n0+b8+x7--20260422_111024/checkpoints/step-035000-epoch-08-loss=0.2463.pt \
    --is_resume True \
    --resume_step 35000 \
    --resume_epoch 8 \
    --vla.enable_gradient_checkpointing True \
    2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
