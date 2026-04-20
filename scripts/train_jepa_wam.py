"""Training script for JEPA-WAM."""

import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from jepa_wam.conf.config import TrainConfig
from jepa_wam.datasets.batch_transform import JEPAWBatchTransform
from jepa_wam.datasets.collator import JEPAWCollator
from jepa_wam.datasets.dataset import JEPAWRLDSDataset
from jepa_wam.models.jepa_vla import JepaVLA
from jepa_wam.training.metrics import Metrics
from jepa_wam.training.trainer import JEPAWTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    cfg = TrainConfig()

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    run_dir = cfg.run_root_dir / (cfg.run_id or "jepa_wam_run")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Tokenizer (same as LLM)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    batch_transform = JEPAWBatchTransform(
        base_tokenizer=tokenizer,
        data_cfg=cfg.data,
    )
    dataset = JEPAWRLDSDataset(
        data_root_dir=cfg.data.data_root_dir,
        data_mix=cfg.data.data_mix,
        batch_transform=batch_transform,
        data_cfg=cfg.data,
        train=True,
    )

    # Save dataset statistics
    # (optional) import json; (run_dir / "dataset_statistics.json").write_text(json.dumps(dataset.dataset_statistics))

    # Collator
    collator = JEPAWCollator(
        tokenizer=tokenizer,
        img_size=cfg.vision.img_size,
        action_horizon=cfg.action_horizon,
        action_dim=cfg.action_dim,
        proprio_dim=cfg.proprio_dim,
    )

    # Model
    print("Building JepaVLA model...")
    model = JepaVLA(cfg)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    # Metrics & Trainer
    metrics = Metrics(run_dir=run_dir)
    trainer = JEPAWTrainer(model=model, cfg=cfg, device_id=device.index or 0)

    # Train
    print("Starting training...")
    trainer.train(
        dataset=dataset,
        collator=collator,
        metrics=metrics,
        save_interval=cfg.save_interval,
        run_dir=run_dir,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
