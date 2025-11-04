#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever for querying ChromaDB and comparing original vs finetuned embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.embedder import EmbeddingModel


class RAGRetriever:
    """RAG retriever for comparing original and finetuned embeddings"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        original_model_path: str = "intfloat/multilingual-e5-small",
        finetuned_model_path: str = "./logs/tensorboard/run_20251103_083449/model",
        original_collection_name: str = "original_embeddings",
        finetuned_collection_name: str = "finetuned_embeddings"
    ):
        """
        Initialize retriever with both embedding models and databases

        Args:
            db_path: Path to ChromaDB storage
            original_model_path: Path to original model
            finetuned_model_path: Path to finetuned model
            original_collection_name: Name of original embeddings collection
            finetuned_collection_name: Name of finetuned embeddings collection
        """
        self.db_path = db_path

        # Initialize embedding models
        print("Loading models...")
        self.original_embedder = EmbeddingModel(
            model_path=original_model_path,
            model_type="original"
        )
        self.finetuned_embedder = EmbeddingModel(
            model_path=finetuned_model_path,
            model_type="finetuned"
        )

        # Initialize ChromaDB client
        print(f"\nConnecting to ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get collections
        try:
            self.original_collection = self.client.get_collection(original_collection_name)
            self.finetuned_collection = self.client.get_collection(finetuned_collection_name)
            print(f"✅ Loaded collections:")
            print(f"   - {original_collection_name}: {self.original_collection.count()} documents")
            print(f"   - {finetuned_collection_name}: {self.finetuned_collection.count()} documents")
        except Exception as e:
            raise ValueError(
                f"Failed to load collections from {db_path}. "
                f"Please run build_database.py first. Error: {e}"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        model_type: str = "both"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for relevant documents

        Args:
            query: Query text
            top_k: Number of results to return
            model_type: "original", "finetuned", or "both"

        Returns:
            Dictionary with results for each model type:
            {
                "original": [
                    {
                        "rank": 1,
                        "document": "query: answer",
                        "query": "query text",
                        "answer": "answer text",
                        "score": 0.95,
                        "distance": 0.05
                    },
                    ...
                ],
                "finetuned": [...]
            }
        """
        results = {}

        if model_type in ["original", "both"]:
            results["original"] = self._search_collection(
                query=query,
                embedder=self.original_embedder,
                collection=self.original_collection,
                top_k=top_k
            )

        if model_type in ["finetuned", "both"]:
            results["finetuned"] = self._search_collection(
                query=query,
                embedder=self.finetuned_embedder,
                collection=self.finetuned_collection,
                top_k=top_k
            )

        return results

    def _search_collection(
        self,
        query: str,
        embedder: EmbeddingModel,
        collection: chromadb.Collection,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search in a single collection

        Args:
            query: Query text
            embedder: Embedding model to use
            collection: ChromaDB collection
            top_k: Number of results

        Returns:
            List of result dictionaries
        """
        # Encode query
        query_embedding = embedder.encode_query(query)
        query_embedding_list = query_embedding.cpu().numpy().tolist()

        # Query collection
        chroma_results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        results = []
        for rank, (doc, metadata, distance) in enumerate(zip(
            chroma_results['documents'][0],
            chroma_results['metadatas'][0],
            chroma_results['distances'][0]
        ), start=1):
            # Convert distance to similarity score
            # ChromaDB cosine distance = 1 - cosine_similarity
            # So similarity = 1 - distance
            similarity = 1.0 - distance

            results.append({
                "rank": rank,
                "document": doc,
                "query": metadata['query'],
                "answer": metadata['answer'],
                "score": similarity,
                "distance": distance
            })

        return results

    def compare_search(self, query: str, top_k: int = 5) -> None:
        """
        Search and print comparison between original and finetuned

        Args:
            query: Query text
            top_k: Number of results to show
        """
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        results = self.search(query, top_k=top_k, model_type="both")

        # Original results
        print("\n" + "🔹" * 40)
        print("ORIGINAL MODEL RESULTS")
        print("🔹" * 40)
        self._print_results(results["original"])

        # Finetuned results
        print("\n" + "🔸" * 40)
        print("FINETUNED MODEL RESULTS")
        print("🔸" * 40)
        self._print_results(results["finetuned"])

        print("\n" + "=" * 80 + "\n")

    def _print_results(self, results: List[Dict[str, Any]]) -> None:
        """Print formatted results"""
        for r in results:
            print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f}")
            print(f"  Query:  {r['query']}")
            print(f"  Answer: {r['answer']}")


def main():
    """Example usage of the retriever"""
    import argparse

    parser = argparse.ArgumentParser(description="Query RAG databases")
    parser.add_argument(
        "--query",
        type=str,
        default="스위스의 수도는?",
        help="Query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Path to ChromaDB"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="./logs/tensorboard/run_20251103_083449/model",
        help="Path to finetuned model"
    )
    parser.add_argument(
        "--original-collection",
        type=str,
        default="original_embeddings",
        help="Name of original embeddings collection"
    )
    parser.add_argument(
        "--finetuned-collection",
        type=str,
        default="finetuned_embeddings",
        help="Name of finetuned embeddings collection"
    )
    args = parser.parse_args()

    # Initialize retriever
    retriever = RAGRetriever(
        db_path=args.db_path,
        finetuned_model_path=args.finetuned_model,
        original_collection_name=args.original_collection,
        finetuned_collection_name=args.finetuned_collection
    )

    # Perform comparison search
    retriever.compare_search(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
