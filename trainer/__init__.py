#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer package for embedding model finetuning

Includes:
- EmbeddingTrainer: Main trainer class with TensorBoard logging
- LoRA utilities: setup_lora_model, print_lora_info
"""

from .trainer import EmbeddingTrainer
from .lora import setup_lora_model, print_lora_info

__all__ = ['EmbeddingTrainer', 'setup_lora_model', 'print_lora_info']
