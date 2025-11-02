#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for embedding model finetuning
Optimized for RTX 2080 8GB GPU
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration optimized for RTX 2080 8GB"""

    # Model settings
    model_name: str = "intfloat/multilingual-e5-small"

    # Dataset settings
    dataset_root: str = "/mnt/d/datasets/KorQuAD"
    max_samples: Optional[int] = None  # None = use all data, or set a number for testing
    train_split: float = 0.9  # 90% train, 10% validation

    # LoRA settings - optimized for 8GB GPU
    lora_r: int = 8  # Rank of LoRA matrices
    lora_alpha: int = 16  # LoRA scaling factor
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # Will auto-detect for e5 model

    # Training settings - optimized for 8GB GPU
    batch_size: int = 16  # Batch size per device
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_seq_length: int = 256  # Shorter sequences = less memory

    # Output settings
    output_dir: str = "./models/finetuned-e5-small-korquad"
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 1000

    # TensorBoard logging settings
    log_dir: str = "./logs/tensorboard"
    log_every_n_steps: int = 10  # Log loss every N steps
    log_gradients: bool = True  # Log gradient statistics

    # Evaluation settings
    eval_samples: int = 1000  # Number of validation samples for evaluation

    # System settings
    num_workers: int = 4  # Number of workers for DataLoader multiprocessing
    fp16: bool = True  # Use mixed precision for memory efficiency
    gradient_checkpointing: bool = True

    def __post_init__(self):
        """Initialize derived configuration values"""
        # Auto-detect target modules for e5 model (typically attention layers)
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value"]

    def to_dict(self):
        """Convert config to dictionary for logging/saving"""
        return {
            "model_name": self.model_name,
            "dataset_root": self.dataset_root,
            "max_samples": self.max_samples,
            "train_split": self.train_split,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "save_steps": self.save_steps,
            "log_dir": self.log_dir,
            "log_every_n_steps": self.log_every_n_steps,
            "log_gradients": self.log_gradients,
            "eval_samples": self.eval_samples,
            "num_workers": self.num_workers,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def print_summary(self):
        """Print a formatted summary of the configuration"""
        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"\n📦 Model Settings:")
        print(f"  Model: {self.model_name}")
        print(f"  Max sequence length: {self.max_seq_length}")

        print(f"\n📊 Dataset Settings:")
        print(f"  Dataset root: {self.dataset_root}")
        print(f"  Train/Val split: {self.train_split:.0%}/{1-self.train_split:.0%}")
        if self.max_samples:
            print(f"  Max samples: {self.max_samples:,}")

        print(f"\n🔧 LoRA Settings:")
        print(f"  Rank (r): {self.lora_r}")
        print(f"  Alpha: {self.lora_alpha}")
        print(f"  Dropout: {self.lora_dropout}")
        print(f"  Target modules: {self.target_modules}")

        print(f"\n🚀 Training Settings:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Warmup steps: {self.warmup_steps}")
        print(f"  FP16: {self.fp16}")
        print(f"  Gradient checkpointing: {self.gradient_checkpointing}")

        print(f"\n💾 Output Settings:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        print(f"  Save steps: {self.save_steps}")

        print(f"\n📊 TensorBoard Settings:")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Log every N steps: {self.log_every_n_steps}")
        print(f"  Log gradients: {self.log_gradients}")

        print(f"\n⚙️ System Settings:")
        print(f"  Data loader workers: {self.num_workers}")
        print("=" * 60 + "\n")
