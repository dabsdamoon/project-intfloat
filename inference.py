#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for using the finetuned E5 model with LoRA
Demonstrates how to load the model and perform question-answering retrieval
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util


class E5KorQuADInference:
    """
    Inference class for the finetuned E5 model on KorQuAD

    Usage:
        model = E5KorQuADInference(model_path="./models/finetuned-e5-small-korquad")
        results = model.search("스위스의 수도는?", passages)
    """

    def __init__(self, model_path: str):
        """
        Initialize the inference model

        Args:
            model_path: Path to the finetuned model directory
        """
        print(f"Loading finetuned model from {model_path}...")

        # Load the finetuned SentenceTransformer model
        # This automatically loads the LoRA weights that were saved
        self.model = SentenceTransformer(model_path)

        # Set to evaluation mode
        self.model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Model loaded on CPU")

        print("Model ready for inference!")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding vector

        Args:
            query: Query text (question)

        Returns:
            Embedding vector (numpy array)
        """
        # Add the "query: " prefix (same as training)
        formatted_query = f"query: {query}"

        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode(
                formatted_query,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization (same as training)
            )

        return embedding

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple queries into embedding vectors

        Args:
            queries: List of query texts
            batch_size: Batch size for encoding

        Returns:
            Embedding matrix (numpy array of shape [num_queries, embedding_dim])
        """
        # Add the "query: " prefix to all queries
        formatted_queries = [f"query: {q}" for q in queries]

        # Generate embeddings in batches
        with torch.no_grad():
            embeddings = self.model.encode(
                formatted_queries,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

        return embeddings

    def encode_passage(self, passage: str) -> np.ndarray:
        """
        Encode a single passage into an embedding vector

        Args:
            passage: Passage text (answer/document)

        Returns:
            Embedding vector (numpy array)
        """
        # Add the "passage: " prefix (same as training)
        formatted_passage = f"passage: {passage}"

        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode(
                formatted_passage,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        return embedding

    def encode_passages(self, passages: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple passages into embedding vectors

        Args:
            passages: List of passage texts
            batch_size: Batch size for encoding

        Returns:
            Embedding matrix (numpy array of shape [num_passages, embedding_dim])
        """
        # Add the "passage: " prefix to all passages
        formatted_passages = [f"passage: {p}" for p in passages]

        # Generate embeddings in batches
        with torch.no_grad():
            embeddings = self.model.encode(
                formatted_passages,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

        return embeddings

    def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and multiple passages

        Args:
            query_embedding: Query embedding vector (1D array)
            passage_embeddings: Passage embedding matrix (2D array)

        Returns:
            Similarity scores (1D array)
        """
        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(passage_embeddings, query_embedding)
        return similarities

    def search(
        self,
        query: str,
        passages: List[str],
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Search for the most relevant passages given a query

        Args:
            query: Query text (question)
            passages: List of passage texts (candidate answers)
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing:
                - rank: Result rank (1-indexed)
                - passage: Passage text
                - score: Similarity score
                - index: Original index in passages list
        """
        # Encode query
        query_emb = self.encode_query(query)

        # Encode passages
        passage_embs = self.encode_passages(passages)

        # Compute similarities
        similarities = self.compute_similarity(query_emb, passage_embs)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Format results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'passage': passages[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            })

        return results

    def batch_search(
        self,
        queries: List[str],
        passages: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, any]]]:
        """
        Batch search for multiple queries

        Args:
            queries: List of query texts
            passages: List of passage texts (same corpus for all queries)
            top_k: Number of top results per query

        Returns:
            List of result lists (one per query)
        """
        # Encode all queries and passages
        query_embs = self.encode_queries(queries)
        passage_embs = self.encode_passages(passages)

        # Compute similarity matrix: [num_queries, num_passages]
        similarity_matrix = np.dot(query_embs, passage_embs.T)

        # Get top-k results for each query
        all_results = []
        for query_idx, similarities in enumerate(similarity_matrix):
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for rank, passage_idx in enumerate(top_indices, 1):
                results.append({
                    'rank': rank,
                    'passage': passages[passage_idx],
                    'score': float(similarities[passage_idx]),
                    'index': int(passage_idx)
                })

            all_results.append(results)

        return all_results


def example_single_query():
    """Example: Single query search"""
    print("\n" + "=" * 60)
    print("Example 1: Single Query Search")
    print("=" * 60)

    # Initialize model
    model = E5KorQuADInference(model_path="./models/finetuned-e5-small-korquad")

    # Example query and passages
    query = "스위스의 수도는 어디인가?"

    passages = [
        "베른은 스위스의 수도이다.",
        "취리히는 스위스에서 가장 큰 도시이다.",
        "제네바는 스위스 서부에 위치한 도시이다.",
        "파리는 프랑스의 수도이다.",
        "베를린은 독일의 수도이다."
    ]

    # Search
    print(f"\nQuery: {query}")
    print(f"\nSearching through {len(passages)} passages...\n")

    results = model.search(query, passages, top_k=3)

    # Display results
    for result in results:
        print(f"Rank {result['rank']}: (Score: {result['score']:.4f})")
        print(f"  {result['passage']}")
        print()


def example_batch_search():
    """Example: Batch query search"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Query Search")
    print("=" * 60)

    # Initialize model
    model = E5KorQuADInference(model_path="./models/finetuned-e5-small-korquad")

    # Multiple queries
    queries = [
        "스위스의 수도는?",
        "프랑스의 수도는?",
        "독일의 수도는?"
    ]

    passages = [
        "베른은 스위스의 수도이다.",
        "취리히는 스위스에서 가장 큰 도시이다.",
        "파리는 프랑스의 수도이다.",
        "베를린은 독일의 수도이다.",
        "런던은 영국의 수도이다."
    ]

    # Batch search
    print(f"\nSearching {len(queries)} queries through {len(passages)} passages...\n")

    all_results = model.batch_search(queries, passages, top_k=2)

    # Display results
    for query_idx, (query, results) in enumerate(zip(queries, all_results)):
        print(f"Query {query_idx + 1}: {query}")
        for result in results:
            print(f"  Rank {result['rank']}: (Score: {result['score']:.4f})")
            print(f"    {result['passage']}")
        print()


def example_similarity_computation():
    """Example: Direct similarity computation"""
    print("\n" + "=" * 60)
    print("Example 3: Direct Similarity Computation")
    print("=" * 60)

    # Initialize model
    model = E5KorQuADInference(model_path="./models/finetuned-e5-small-korquad")

    # Encode query and passages separately
    query = "한국의 수도는?"
    passages = [
        "서울은 한국의 수도이다.",
        "부산은 한국의 제2의 도시이다."
    ]

    print(f"\nQuery: {query}")
    print("\nPassages:")
    for i, p in enumerate(passages):
        print(f"  {i + 1}. {p}")

    # Encode
    query_emb = model.encode_query(query)
    passage_embs = model.encode_passages(passages)

    print(f"\nQuery embedding shape: {query_emb.shape}")
    print(f"Passage embeddings shape: {passage_embs.shape}")

    # Compute similarity
    similarities = model.compute_similarity(query_emb, passage_embs)

    print("\nSimilarity scores:")
    for i, score in enumerate(similarities):
        print(f"  Passage {i + 1}: {score:.4f}")


if __name__ == "__main__":
    """
    Run all examples

    Make sure you have trained the model first using train.py
    The model should be saved at: ./models/finetuned-e5-small-korquad
    """

    # Check if model exists
    model_path = "./models/finetuned-e5-small-korquad"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        exit(1)

    # Run examples
    try:
        example_single_query()
        example_batch_search()
        example_similarity_computation()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
