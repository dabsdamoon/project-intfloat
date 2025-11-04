#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module using OpenAI embeddings to compare retrieval quality.
"""

import os
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIEvaluator:
    """
    Evaluator that uses OpenAI embeddings to assess retrieval quality.
    Compares how well retrieved documents match the query.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI evaluator.

        Args:
            model: OpenAI embedding model to use
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def evaluate_retrieval(
        self,
        query: str,
        original_results: List[Dict[str, Any]],
        finetuned_results: List[Dict[str, Any]],
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality using OpenAI embeddings.

        Args:
            query: Search query
            original_results: Results from original model
            finetuned_results: Results from finetuned model
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with evaluation metrics
        """
        # Get query embedding
        if progress_callback:
            progress_callback(0.1, "Embedding query with OpenAI...")
        query_embedding = self.get_embedding(query)

        # Evaluate original model results
        if progress_callback:
            progress_callback(0.2, f"Embedding {len(original_results)} documents from original model...")
        original_scores = []
        for idx, result in enumerate(original_results, 1):
            if progress_callback:
                progress_callback(0.2 + (0.3 * idx / len(original_results)),
                                f"Original model: {idx}/{len(original_results)} documents")
            doc_text = result['answer']
            doc_embedding = self.get_embedding(doc_text)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            original_scores.append(similarity)

        # Evaluate finetuned model results
        if progress_callback:
            progress_callback(0.5, f"Embedding {len(finetuned_results)} documents from finetuned model...")
        finetuned_scores = []
        for idx, result in enumerate(finetuned_results, 1):
            if progress_callback:
                progress_callback(0.5 + (0.3 * idx / len(finetuned_results)),
                                f"Finetuned model: {idx}/{len(finetuned_results)} documents")
            doc_text = result['answer']
            doc_embedding = self.get_embedding(doc_text)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            finetuned_scores.append(similarity)

        # Calculate metrics
        if progress_callback:
            progress_callback(0.8, "Calculating similarity metrics...")
        original_avg = np.mean(original_scores) if original_scores else 0.0
        finetuned_avg = np.mean(finetuned_scores) if finetuned_scores else 0.0

        original_max = max(original_scores) if original_scores else 0.0
        finetuned_max = max(finetuned_scores) if finetuned_scores else 0.0

        # Determine winner
        if progress_callback:
            progress_callback(0.9, "Comparing results...")
        if original_avg > finetuned_avg:
            winner = "Original"
            improvement = ((original_avg - finetuned_avg) / finetuned_avg * 100) if finetuned_avg > 0 else 0
        elif finetuned_avg > original_avg:
            winner = "Finetuned"
            improvement = ((finetuned_avg - original_avg) / original_avg * 100) if original_avg > 0 else 0
        else:
            winner = "Tie"
            improvement = 0.0

        return {
            "query": query,
            "original": {
                "avg_similarity": float(original_avg),
                "max_similarity": float(original_max),
                "scores": [float(s) for s in original_scores],
                "num_results": len(original_scores)
            },
            "finetuned": {
                "avg_similarity": float(finetuned_avg),
                "max_similarity": float(finetuned_max),
                "scores": [float(s) for s in finetuned_scores],
                "num_results": len(finetuned_scores)
            },
            "comparison": {
                "winner": winner,
                "improvement_pct": float(improvement),
                "avg_diff": float(abs(original_avg - finetuned_avg))
            },
            "evaluation_model": self.model
        }

    def format_evaluation_results(self, eval_results: Dict[str, Any]) -> str:
        """
        Format evaluation results as markdown string.

        Args:
            eval_results: Evaluation results dictionary

        Returns:
            Formatted markdown string
        """
        original = eval_results["original"]
        finetuned = eval_results["finetuned"]
        comparison = eval_results["comparison"]

        output = "### 📊 OpenAI Embedding Evaluation\n\n"
        output += f"**Evaluation Model**: `{eval_results['evaluation_model']}`\n\n"
        output += f"**Query**: {eval_results['query']}\n\n"
        output += "---\n\n"

        # Metrics table
        output += "#### Similarity Scores (Query ↔ Retrieved Documents)\n\n"
        output += "| Metric | Original Model | Finetuned Model |\n"
        output += "|--------|---------------|----------------|\n"
        output += f"| **Average Similarity** | {original['avg_similarity']:.4f} | {finetuned['avg_similarity']:.4f} |\n"
        output += f"| **Max Similarity** | {original['max_similarity']:.4f} | {finetuned['max_similarity']:.4f} |\n"
        output += f"| **Results Count** | {original['num_results']} | {finetuned['num_results']} |\n\n"

        # Winner announcement
        winner = comparison["winner"]
        improvement = comparison["improvement_pct"]

        output += "#### 🏆 Evaluation Result\n\n"

        if winner == "Tie":
            output += "**Result**: Both models performed equally well.\n\n"
        else:
            output += f"**Winner**: **{winner} Model** 🎯\n\n"
            output += f"**Performance Improvement**: {improvement:.2f}%\n\n"
            output += f"**Average Difference**: {comparison['avg_diff']:.4f}\n\n"

        # Interpretation
        output += "#### 💡 Interpretation\n\n"
        output += "The evaluation uses OpenAI's embedding model as a neutral reference to measure "
        output += "how semantically similar the retrieved documents are to the query. "
        output += "Higher similarity scores indicate better retrieval quality.\n\n"

        # Individual scores
        output += "<details>\n"
        output += "<summary>📈 Detailed Scores by Rank</summary>\n\n"
        output += "**Original Model:**\n"
        for i, score in enumerate(original['scores'], 1):
            output += f"- Rank {i}: {score:.4f}\n"
        output += "\n**Finetuned Model:**\n"
        for i, score in enumerate(finetuned['scores'], 1):
            output += f"- Rank {i}: {score:.4f}\n"
        output += "</details>\n"

        return output


def evaluate_models(
    query: str,
    original_results: List[Dict[str, Any]],
    finetuned_results: List[Dict[str, Any]],
    progress_callback=None
) -> str:
    """
    Convenience function to evaluate models and return formatted results.

    Args:
        query: Search query
        original_results: Results from original model
        finetuned_results: Results from finetuned model
        progress_callback: Optional callback for progress updates

    Returns:
        Formatted evaluation results as markdown
    """
    try:
        if progress_callback:
            progress_callback(0.05, "Initializing OpenAI evaluator...")
        evaluator = OpenAIEvaluator()
        eval_results = evaluator.evaluate_retrieval(
            query, original_results, finetuned_results, progress_callback
        )
        if progress_callback:
            progress_callback(0.95, "Formatting results...")
        return evaluator.format_evaluation_results(eval_results)
    except Exception as e:
        return f"### ❌ Evaluation Error\n\n{str(e)}\n\nPlease ensure OPENAI_API_KEY is set in your .env file."
