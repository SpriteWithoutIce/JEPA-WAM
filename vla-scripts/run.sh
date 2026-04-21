
LIBERO_DATA="/mnt/data/linyihan/datasets/modified_libero_rlds"
QWEN_PATH="/mnt/data/linyihan/ckpt/Qwen2.5-0.5B"
VJEPA_CKPT="/mnt/data/linyihan/ckpt/vjepa2-vitl-fpc64-256"
RUNS_DIR="./runs"

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
    --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90 \
    --vla.data_mix libero_4_task_suites_no_noops \
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}" \
    --llm_checkpoint_path "${QWEN_PATH}" \
    --data_root_dir "${LIBERO_DATA}" \
    --run_root_dir "${RUNS_DIR}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 4 \
    --vla.per_device_batch_size 2 \
    --vla.learning_rate 2e-5 \
    --vla.max_steps 100000 \
    --save_interval 2500 \
    --seed 7 \
