#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer class for embedding model finetuning with TensorBoard logging
"""

import os
import torch
from typing import List
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from config import TrainingConfig


class EmbeddingTrainer:
    """
    Trainer class for finetuning embedding models with LoRA/PEFT
    Includes comprehensive TensorBoard logging for monitoring training progress
    """

    def __init__(self, model: SentenceTransformer, config: TrainingConfig):
        """
        Initialize the trainer

        Args:
            model: SentenceTransformer model (with LoRA already applied)
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.writer = None
        self.global_step_counter = [0]
        self.best_score = -1

    def _setup_tensorboard(self):
        """Setup TensorBoard writer with timestamped run directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.config.log_dir, f'run_{timestamp}')
        self.writer = SummaryWriter(log_dir=log_dir)

        print(f"\n📊 TensorBoard initialized")
        print(f"  Log directory: {log_dir}")
        print(f"  To view: tensorboard --logdir={self.config.log_dir}")

        # Log configuration to TensorBoard
        config_text = "\n".join([f"{k}: {v}" for k, v in self.config.to_dict().items()])
        self.writer.add_text('config', config_text, 0)

        return log_dir

    def _setup_gradient_logging(self):
        """
        Setup hooks to log gradient statistics to TensorBoard

        Registers backward hooks on key modules (query, key, value, dense layers)
        to track gradient norms and distributions
        """
        def log_gradients(module, grad_input, grad_output):
            """Hook function to log gradient statistics"""
            if not self.config.log_gradients or self.global_step_counter[0] % self.config.log_every_n_steps != 0:
                return

            # Log gradient norms for this module
            if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                grad = module.weight.grad

                # Check if gradient has valid elements
                if grad.numel() == 0:
                    return  # Skip empty gradients

                # Check for NaN or inf values
                if not torch.isfinite(grad).all():
                    return  # Skip invalid gradients

                # Log gradient norm
                grad_norm = grad.norm().item()
                self.writer.add_scalar(
                    f'gradients/{module.__class__.__name__}_weight_norm',
                    grad_norm,
                    self.global_step_counter[0]
                )

                # Log gradient histogram (only if gradient has non-zero variance)
                # This prevents "empty histogram" errors from all-zero gradients
                if grad.std() > 1e-10:  # Check for non-trivial variance
                    self.writer.add_histogram(
                        f'gradients/{module.__class__.__name__}_weight',
                        grad,
                        self.global_step_counter[0]
                    )

        # Register hooks for key modules
        auto_model = self.model[0].auto_model
        for name, module in auto_model.named_modules():
            if any(target in name for target in ['query', 'key', 'value', 'dense']):
                module.register_full_backward_hook(log_gradients)

        print("  Gradient logging hooks registered")

    def _create_evaluator(self, queries, corpus, relevant_docs, name: str = "korquad-validation"):
        """
        Create an evaluator that logs metrics to TensorBoard

        Args:
            queries: Evaluation queries dict {query_id: query_text}
            corpus: Evaluation corpus dict {doc_id: doc_text}
            relevant_docs: Relevant documents mapping dict {query_id: set(doc_ids)}
            name: Name for the evaluator

        Returns:
            InformationRetrievalEvaluator instance
        """
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=name,
            show_progress_bar=True,
            write_csv=True
        )

        # Store writer reference for potential custom logging
        evaluator.tensorboard_writer = self.writer

        return evaluator

    def _setup_optimizer_and_scheduler(self, train_dataloader):
        """
        Setup optimizer and learning rate scheduler with warmup

        Args:
            train_dataloader: Training data loader

        Returns:
            optimizer, scheduler tuple
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate
        )

        # Setup learning rate scheduler with warmup
        total_steps = len(train_dataloader) * self.config.num_epochs
        warmup_steps = self.config.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler

    def _training_step(self, batch, train_loss, optimizer, scaler):
        """
        Execute a single training step

        Args:
            batch: Training batch (list of InputExample objects)
            train_loss: Loss function (MultipleNegativesRankingLoss)
            optimizer: Optimizer
            scaler: Gradient scaler for mixed precision (or None)

        Returns:
            loss_value: Loss tensor
        """
        optimizer.zero_grad()

        # Extract texts from InputExample objects
        # Each InputExample has .texts = [query, passage]
        texts = [example.texts for example in batch]

        # Separate queries and passages
        queries = [text[0] for text in texts]
        passages = [text[1] for text in texts]

        # Tokenize inputs
        query_features = self.model.tokenize(queries)
        passage_features = self.model.tokenize(passages)

        # Move features to device
        device = next(self.model.parameters()).device
        query_features = {key: val.to(device) for key, val in query_features.items()}
        passage_features = {key: val.to(device) for key, val in passage_features.items()}

        # Forward pass with mixed precision
        if self.config.fp16 and scaler:
            with autocast():
                # Compute embeddings using the model's forward pass
                query_embeddings = self.model(query_features)['sentence_embedding']
                passage_embeddings = self.model(passage_features)['sentence_embedding']

                # Compute contrastive loss using in-batch negatives
                loss_value = self._compute_contrastive_loss(query_embeddings, passage_embeddings)
        else:
            # Compute embeddings using the model's forward pass
            query_embeddings = self.model(query_features)['sentence_embedding']
            passage_embeddings = self.model(passage_features)['sentence_embedding']

            # Compute contrastive loss using in-batch negatives
            loss_value = self._compute_contrastive_loss(query_embeddings, passage_embeddings)

        # Backward pass
        if self.config.fp16 and scaler:
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_value.backward()
            optimizer.step()

        return loss_value

    def _compute_contrastive_loss(self, query_embeddings, passage_embeddings):
        """
        Compute contrastive loss with in-batch negatives

        This implements the same loss as MultipleNegativesRankingLoss:
        - Each query should match its corresponding passage
        - All other passages in the batch are treated as negatives

        Args:
            query_embeddings: Tensor of shape (batch_size, embedding_dim)
            passage_embeddings: Tensor of shape (batch_size, embedding_dim)

        Returns:
            loss: Scalar loss tensor
        """
        # Normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)

        # Compute similarity scores: (batch_size, batch_size)
        # scores[i][j] = similarity between query i and passage j
        scores = torch.matmul(query_embeddings, passage_embeddings.t())

        # Scale by temperature (default 20 in sentence-transformers)
        scores = scores * 20

        # Labels: diagonal elements are positive pairs
        # query 0 should match passage 0, query 1 should match passage 1, etc.
        labels = torch.arange(len(scores), device=scores.device)

        # Cross-entropy loss treats each row as a classification problem
        # We want to maximize scores[i][i] and minimize scores[i][j] for j != i
        loss = torch.nn.functional.cross_entropy(scores, labels)

        return loss

    def _log_metrics(self, loss_value, scheduler, epoch, batch_idx, total_batches, global_step):
        """
        Log metrics to TensorBoard

        Args:
            loss_value: Current loss value
            scheduler: Learning rate scheduler
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total number of batches
            global_step: Global step counter
        """
        if global_step % self.config.log_every_n_steps == 0:
            self.writer.add_scalar('train/loss', loss_value.item(), global_step)
            self.writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
            self.writer.add_scalar('train/epoch', epoch + (batch_idx / total_batches), global_step)

    def _evaluate_and_save(self, evaluator, global_step):
        """
        Evaluate model and save if best score

        Args:
            evaluator: Evaluation function
            global_step: Current global step

        Returns:
            eval_score: Evaluation score
        """
        print(f"\n\nEvaluating at step {global_step}...")
        self.model.eval()
        eval_score = evaluator(self.model)
        self.model.train()

        # Log evaluation metrics
        self.writer.add_scalar('eval/score', eval_score, global_step)

        # Save best model
        if eval_score > self.best_score:
            self.best_score = eval_score
            print(f"New best score: {self.best_score:.4f} - Saving model...")
            self.model.save(self.config.output_dir)

            # Save LoRA weights separately
            lora_output_path = os.path.join(self.config.output_dir, "lora_weights")
            self.model[0].auto_model.save_pretrained(lora_output_path)

        return eval_score

    def train(self, train_dataloader: DataLoader, train_loss, evaluator):
        """
        Main training loop with TensorBoard logging

        Args:
            train_dataloader: DataLoader for training data
            train_loss: Loss function
            evaluator: Evaluation function

        Returns:
            best_score: Best validation score achieved during training
        """
        # Setup TensorBoard
        log_dir = self._setup_tensorboard()
        self._setup_gradient_logging()

        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(train_dataloader)

        # Setup gradient scaler for mixed precision
        scaler = GradScaler() if self.config.fp16 else None

        # Training setup
        self.model.train()
        global_step = 0

        # Calculate total steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.config.num_epochs

        print(f"\n📈 Training Steps:")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Total steps: {total_steps:,}")

        print("\n" + "=" * 60)
        print("Starting training with TensorBoard logging...")
        print("=" * 60 + "\n")

        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                loss_value = self._training_step(batch, train_loss, optimizer, scaler)
                scheduler.step()

                # Update step counter
                global_step += 1
                self.global_step_counter[0] = global_step
                epoch_loss += loss_value.item()

                # Log metrics to TensorBoard
                self._log_metrics(
                    loss_value, scheduler, epoch, batch_idx,
                    len(train_dataloader), global_step
                )

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss_value.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })

                # Evaluate and save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._evaluate_and_save(evaluator, global_step)

            # Log epoch metrics
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
            print(f"\nEpoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")

        # Training completed
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Model saved to: {self.config.output_dir}")
        print(f"Best validation score: {self.best_score:.4f}")

        # Close TensorBoard writer
        self.writer.close()
        print(f"\nTensorBoard logs saved to: {log_dir}")

        return self.best_score

    def evaluate(self, evaluator):
        """
        Run evaluation on the model

        Args:
            evaluator: Evaluation function

        Returns:
            eval_score: Evaluation score
        """
        print("\nRunning final evaluation...")
        self.model.eval()
        eval_score = evaluator(self.model)

        if self.writer:
            self.writer.add_scalar('eval/final_score', eval_score, 0)

        print(f"Final validation score: {eval_score}")
        return eval_score
