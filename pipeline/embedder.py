#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding utilities for original and finetuned models
"""

import torch
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for embedding models (original and finetuned)"""

    def __init__(self, model_path: str, model_type: str = "original"):
        """
        Initialize embedding model

        Args:
            model_path: Path to model (Hugging Face model name or local path)
            model_type: "original" or "finetuned"
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = SentenceTransformer(model_path)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        print(f"Loaded {model_type} model from: {model_path}")
        if torch.cuda.is_available():
            print(f"  Model on: cuda")
        else:
            print(f"  Model on: cpu")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """
        Encode texts to embeddings

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Tensor of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a single query with appropriate prefix

        Args:
            query: Query text

        Returns:
            Tensor embedding
        """
        # E5 models use "query:" prefix for queries
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            [prefixed_query],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embedding[0]  # Return single embedding
