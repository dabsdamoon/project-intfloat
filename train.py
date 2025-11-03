#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for finetuning multilingual-e5-small on KorQuAD 2.1 dataset
Uses LoRA/PEFT for efficient training on 8GB GPU
"""

import os
import sys
import torch
import re
from typing import List, Tuple
from tqdm import tqdm
from html import unescape

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Import project modules
from config import TrainingConfig
from dataloader.loader import load_entire_dataset
from trainer import EmbeddingTrainer, setup_lora_model


def clean_html_text(text: str) -> str:
    """
    Clean HTML markup from text while preserving content.

    Args:
        text: Raw text that may contain HTML

    Returns:
        Cleaned text with HTML removed
    """
    if not text:
        return ""

    # Remove DOCTYPE and html/head/body tags
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<html[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</html>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<body[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</body>', '', text, flags=re.IGNORECASE)

    # Remove script and style tags with their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove all HTML tags but keep the content
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities (e.g., &nbsp; → space, &lt; → <)
    text = unescape(text)

    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
    text = text.strip()

    return text


def collate_input_examples(batch):
    """
    Custom collate function for InputExample objects.

    Instead of trying to stack InputExample objects (which fails with default collate),
    we just return them as a list. The model's _prepare_batch() will handle tokenization.

    This allows us to use num_workers > 0 for parallel data loading.
    """
    return batch


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
        for qa in article.get('qas', []):
            question = qa.get('question', '')
            answer_obj = qa.get('answer', {})

            # Extract answer text from answer object
            # KorQuAD answer structure: {'text': 'answer text', 'answer_start': position}
            if isinstance(answer_obj, dict):
                answer = answer_obj.get('text', '')
            elif isinstance(answer_obj, str):
                answer = answer_obj
            else:
                answer = ''

            if not question or not answer:
                continue

            # Clean HTML from answer text (Wikipedia data may contain HTML markup)
            answer = clean_html_text(answer)

            # Skip if answer is empty after cleaning
            if not answer:
                continue

            # Use Question → Answer pairs (verified, precise)
            # This avoids false negative problem in contrastive learning
            # and provides strongest supervision signal
            example = InputExample(
                texts=[f"query: {question}", f"passage: {answer}"]
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

    # Create log directory for TensorBoard
    # Note: Model directory will be created automatically within each run
    os.makedirs(config.log_dir, exist_ok=True)

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
    print("\n[1/6] Loading KorQuAD dataset...")
    dataset = load_entire_dataset(config.dataset_root, verbose=True)

    if dataset is None:
        print("Failed to load dataset!")
        sys.exit(1)

    print(f"\nDataset statistics:")
    print(f"  Total files: {dataset['total_files']}")
    print(f"  Total articles: {dataset['total_articles']}")
    print(f"  Total QA pairs: {dataset['total_qas']}")

    # Step 2: Prepare training data
    print("\n[2/6] Preparing training data...")
    train_examples, val_examples = prepare_training_data(dataset, config)

    # Step 3: Load base model
    print("\n[3/6] Loading base model...")
    model = SentenceTransformer(config.model_name)

    # Set max sequence length
    model.max_seq_length = config.max_seq_length

    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        model[0].auto_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Step 4: Apply LoRA
    print("\n[4/6] Applying LoRA...")
    model = setup_lora_model(model, config)

    # Step 5: Setup training
    print("\n[5/6] Setting up training...")

    # Create DataLoader with custom collate function
    # The collate function just returns InputExample objects as a list,
    # allowing multiprocessing while deferring tokenization to the model
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_input_examples  # Custom collate to handle InputExample
    )

    # Use MultipleNegativesRankingLoss - excellent for embedding training
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluation data
    queries, corpus, relevant_docs = create_evaluation_data(val_examples, config)

    # Initialize trainer
    print("\n[6/6] Initializing trainer...")
    trainer = EmbeddingTrainer(model=model, config=config)

    # Create evaluator
    evaluator = trainer._create_evaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="korquad-validation"
    )

    # Start training with TensorBoard logging
    best_score = trainer.train(
        train_dataloader=train_dataloader,
        train_loss=train_loss,
        evaluator=evaluator
    )

    # Final evaluation
    final_score = trainer.evaluate(evaluator)

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
