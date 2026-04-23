python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /home/jwhe/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n0+b16+x7--20260422_165455/checkpoints/step-060000-epoch-14-loss=0.2303.pt \
    --llm_checkpoint_path /home/jwhe/linyihan/CKPT/Qwen2.5-0.5B \
    --task_suite_name libero_goal \
    --num_trials_per_task 10 \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True