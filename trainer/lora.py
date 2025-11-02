#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) utilities for efficient model finetuning
"""

from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType

from config import TrainingConfig


def setup_lora_model(base_model: SentenceTransformer, config: TrainingConfig) -> SentenceTransformer:
    """
    Configure LoRA for the base model

    LoRA (Low-Rank Adaptation) enables efficient finetuning by adding trainable
    low-rank matrices to the model while keeping the original weights frozen.
    This dramatically reduces memory usage and trainable parameters.

    For sentence-transformers, we apply LoRA to the underlying transformer model.

    Args:
        base_model: SentenceTransformer model to apply LoRA to
        config: Training configuration containing LoRA parameters

    Returns:
        base_model: SentenceTransformer with LoRA applied

    Example:
        >>> model = SentenceTransformer("intfloat/multilingual-e5-small")
        >>> config = TrainingConfig(lora_r=8, lora_alpha=16)
        >>> model = setup_lora_model(model, config)
    """
    print("\nConfiguring LoRA for efficient training...")

    # Get the base transformer model from sentence-transformers
    # sentence-transformers wraps the transformer in a module list
    auto_model = base_model[0].auto_model

    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=config.lora_r,                      # Rank of low-rank matrices
        lora_alpha=config.lora_alpha,          # Scaling factor
        target_modules=config.target_modules,  # Modules to apply LoRA to
        lora_dropout=config.lora_dropout,      # Dropout for LoRA layers
        bias="none",                           # Don't apply LoRA to bias
        task_type=TaskType.FEATURE_EXTRACTION  # Task type for embedding models
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(auto_model, lora_config)

    # Replace the base model with LoRA-enhanced version
    base_model[0].auto_model = peft_model

    # Print trainable parameters statistics
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_percentage = 100 * trainable_params / total_params

    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_percentage:.2f}%)")
    print(f"Memory savings: ~{100 - trainable_percentage:.1f}% reduction in trainable params")

    return base_model


def print_lora_info(model: SentenceTransformer):
    """
    Print detailed information about LoRA configuration

    Args:
        model: SentenceTransformer model with LoRA applied
    """
    peft_model = model[0].auto_model

    if not hasattr(peft_model, 'peft_config'):
        print("No LoRA configuration found in model")
        return

    print("\n" + "=" * 60)
    print("LoRA Configuration Details")
    print("=" * 60)

    config = peft_model.peft_config.get('default')
    if config:
        print(f"Rank (r): {config.r}")
        print(f"Alpha: {config.lora_alpha}")
        print(f"Dropout: {config.lora_dropout}")
        print(f"Target modules: {config.target_modules}")
        print(f"Task type: {config.task_type}")

    print("\nTrainable modules:")
    for name, param in peft_model.named_parameters():
        if param.requires_grad and 'lora' in name:
            print(f"  {name}: {param.shape}")

    print("=" * 60 + "\n")
