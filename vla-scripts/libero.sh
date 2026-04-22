python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /home/jwhe/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b4+x7--20260421_180452/checkpoints/step-030000-epoch-03-loss=0.3369.pt \
    --llm_checkpoint_path /home/jwhe/linyihan/CKPT/Qwen2.5-0.5B \
    --task_suite_name libero_object \
    --num_trials_per_task 2 \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True