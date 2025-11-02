#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for finetuning multilingual-e5-small on KorQuAD 2.1 dataset
Uses LoRA/PEFT for efficient training on 8GB GPU
"""

import os
import sys
import torch
from typing import List, Tuple
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

# Import project modules
from config import TrainingConfig
from dataloader.loader import load_entire_dataset


def prepare_training_data(dataset: dict, config: TrainingConfig) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Convert KorQuAD dataset to training examples for embedding model

    KorQuAD structure:
    - articles: list of articles
        - title: article title
        - context: article text
        - qas: list of QA pairs
            - question: question text
            - answer: answer text

    We create pairs: (question, context) as positive pairs
    """
    print("\nPreparing training data from KorQuAD...")

    examples = []
    articles = dataset['articles']

    # Limit samples if specified (for testing)
    if config.max_samples:
        total_qas = 0
        limited_articles = []
        for article in articles:
            limited_articles.append(article)
            total_qas += len(article.get('qas', []))
            if total_qas >= config.max_samples:
                break
        articles = limited_articles

    # Convert to training examples
    for article in tqdm(articles, desc="Processing articles"):
        context = article.get('context', '')
        title = article.get('title', '')

        # Combine title and context for better retrieval
        full_context = f"{title}\n{context}" if title else context

        for qa in article.get('qas', []):
            question = qa.get('question', '')

            if question and full_context:
                # E5 models use instruction prefixes
                # Query gets "query: " prefix, passage gets "passage: " prefix
                example = InputExample(
                    texts=[f"query: {question}", f"passage: {full_context}"]
                )
                examples.append(example)

    print(f"Created {len(examples)} training examples")

    # Split into train and validation
    split_idx = int(len(examples) * config.train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    return train_examples, val_examples


def setup_lora_model(base_model: SentenceTransformer, config: TrainingConfig):
    """
    Configure LoRA for the base model

    For sentence-transformers, we need to apply LoRA to the underlying transformer
    """
    print("\nConfiguring LoRA for efficient training...")

    # Get the base transformer model from sentence-transformers
    auto_model = base_model[0].auto_model

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(auto_model, lora_config)

    # Replace the base model with LoRA-enhanced version
    base_model[0].auto_model = peft_model

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return base_model


def create_evaluation_data(val_examples: List[InputExample], config: TrainingConfig):
    """
    Create evaluation dataset in the format required by InformationRetrievalEvaluator

    Args:
        val_examples: List of validation examples
        config: Training configuration

    Returns:
        queries: dict {query_id: query_text}
        corpus: dict {doc_id: doc_text}
        relevant_docs: dict {query_id: set(relevant_doc_ids)}
    """
    queries = {}
    corpus = {}
    relevant_docs = {}

    # Limit validation samples for faster evaluation
    val_examples = val_examples[:config.eval_samples]

    for idx, example in enumerate(val_examples):
        query_id = f"q_{idx}"
        doc_id = f"d_{idx}"

        # Remove prefixes for evaluation
        query_text = example.texts[0].replace("query: ", "")
        doc_text = example.texts[1].replace("passage: ", "")

        queries[query_id] = query_text
        corpus[doc_id] = doc_text
        relevant_docs[query_id] = {doc_id}

    return queries, corpus, relevant_docs


def main():
    """Main training function"""

    # Initialize configuration
    config = TrainingConfig()

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Print header
    print("\n" + "=" * 60)
    print("KorQuAD Embedding Model Finetuning")
    print("=" * 60)

    # Print device info
    print(f"\n💻 Device Information:")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Print configuration summary
    config.print_summary()

    # Step 1: Load KorQuAD dataset
    print("\n[1/5] Loading KorQuAD dataset...")
    dataset = load_entire_dataset(config.dataset_root, verbose=True)

    if dataset is None:
        print("Failed to load dataset!")
        sys.exit(1)

    print(f"\nDataset statistics:")
    print(f"  Total files: {dataset['total_files']}")
    print(f"  Total articles: {dataset['total_articles']}")
    print(f"  Total QA pairs: {dataset['total_qas']}")

    # Step 2: Prepare training data
    print("\n[2/5] Preparing training data...")
    train_examples, val_examples = prepare_training_data(dataset, config)

    # Step 3: Load base model
    print("\n[3/5] Loading base model...")
    model = SentenceTransformer(config.model_name)

    # Set max sequence length
    model.max_seq_length = config.max_seq_length

    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        model[0].auto_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Step 4: Apply LoRA
    print("\n[4/5] Applying LoRA...")
    model = setup_lora_model(model, config)

    # Step 5: Setup training
    print("\n[5/5] Setting up training...")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Use MultipleNegativesRankingLoss - excellent for embedding training
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Setup evaluator
    queries, corpus, relevant_docs = create_evaluation_data(val_examples, config)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="korquad-validation",
        show_progress_bar=True
    )

    # Calculate total steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.num_epochs

    print(f"\n📈 Training Steps:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        optimizer_params={'lr': config.learning_rate},
        output_path=config.output_dir,
        evaluation_steps=config.save_steps,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=config.fp16  # Automatic Mixed Precision
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")

    # Save LoRA weights separately for easy loading
    lora_output_path = os.path.join(config.output_dir, "lora_weights")
    model[0].auto_model.save_pretrained(lora_output_path)
    print(f"LoRA weights saved to: {lora_output_path}")

    # Final evaluation
    print("\nRunning final evaluation...")
    final_score = evaluator(model)
    print(f"Final validation score: {final_score}")

    return model


if __name__ == "__main__":
    try:
        model = main()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
