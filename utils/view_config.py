#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to view training configuration from a saved model
"""

import os
import json
import sys
from pathlib import Path


def view_training_config(model_path):
    """
    Display training configuration from a saved model

    Args:
        model_path: Path to model directory (e.g., ./logs/tensorboard/run_20251103_143022/model)
    """
    config_path = os.path.join(model_path, "training_config.json")

    if not os.path.exists(config_path):
        print(f"❌ Error: Training config not found at {config_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def print_config(config, title="Training Configuration"):
    """Pretty print configuration"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    # Group configs by category
    categories = {
        "Model Settings": ["model_name", "max_seq_length"],
        "Dataset Settings": ["dataset_root", "max_samples", "train_split", "eval_samples"],
        "LoRA Settings": ["lora_r", "lora_alpha", "lora_dropout", "target_modules"],
        "Training Settings": ["batch_size", "num_epochs", "learning_rate", "warmup_steps", "fp16", "gradient_checkpointing"],
        "Output Settings": ["save_steps"],
        "TensorBoard Settings": ["log_dir", "log_every_n_steps", "log_gradients", "log_gradient_histograms"],
        "System Settings": ["num_workers"]
    }

    for category, keys in categories.items():
        print(f"\n📌 {category}:")
        for key in keys:
            if key in config:
                value = config[key]
                # Format value
                if isinstance(value, list):
                    value = ", ".join(map(str, value))
                print(f"  {key:25s} = {value}")

    print("\n" + "=" * 70 + "\n")


def compare_configs(model_paths):
    """
    Compare training configurations from multiple models

    Args:
        model_paths: List of model directory paths
    """
    configs = []
    names = []

    for path in model_paths:
        config = view_training_config(path)
        if config:
            configs.append(config)
            # Extract run name from path
            run_name = Path(path).parent.name if Path(path).name == "model" else Path(path).name
            names.append(run_name)

    if not configs:
        print("❌ No valid configs found")
        return

    print("\n" + "=" * 100)
    print(" Configuration Comparison")
    print("=" * 100)

    # Find all unique keys
    all_keys = set()
    for config in configs:
        all_keys.update(config.keys())

    # Print header
    print(f"\n{'Parameter':<30}", end="")
    for name in names:
        print(f"{name:<30}", end="")
    print()
    print("-" * (30 + 30 * len(names)))

    # Print each parameter
    for key in sorted(all_keys):
        print(f"{key:<30}", end="")
        values = []
        for config in configs:
            value = config.get(key, "N/A")
            if isinstance(value, list):
                value = str(value)
            value_str = str(value)[:28]  # Truncate long values
            values.append(value_str)
            print(f"{value_str:<30}", end="")

        # Highlight if values differ
        if len(set(values)) > 1:
            print(" ⚠️  DIFFERENT", end="")

        print()

    print("=" * 100 + "\n")


def list_available_models():
    """List all available trained models"""
    tensorboard_dir = "./logs/tensorboard"

    if not os.path.exists(tensorboard_dir):
        print(f"❌ No models found: {tensorboard_dir} does not exist")
        return []

    run_dirs = [
        os.path.join(tensorboard_dir, d)
        for d in os.listdir(tensorboard_dir)
        if os.path.isdir(os.path.join(tensorboard_dir, d)) and d.startswith("run_")
    ]

    if not run_dirs:
        print(f"❌ No training runs found in {tensorboard_dir}")
        return []

    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print("\n" + "=" * 70)
    print(" Available Trained Models")
    print("=" * 70)

    model_paths = []
    for i, run_dir in enumerate(run_dirs, 1):
        model_path = os.path.join(run_dir, "model")
        config_path = os.path.join(model_path, "training_config.json")

        run_name = os.path.basename(run_dir)
        has_config = "✓" if os.path.exists(config_path) else "✗"

        print(f"{i}. {run_name}  [{has_config} config]")

        if os.path.exists(config_path):
            model_paths.append(model_path)

    print("=" * 70 + "\n")

    return model_paths


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: list available models
        model_paths = list_available_models()

        if model_paths:
            print("Usage:")
            print("  python view_config.py <model_path>              # View single config")
            print("  python view_config.py <path1> <path2> ...       # Compare multiple configs")
            print("\nExample:")
            print(f"  python view_config.py {model_paths[0]}")

    elif len(sys.argv) == 2:
        # Single model: view config
        model_path = sys.argv[1]
        config = view_training_config(model_path)

        if config:
            print_config(config)

    else:
        # Multiple models: compare configs
        model_paths = sys.argv[1:]
        compare_configs(model_paths)
