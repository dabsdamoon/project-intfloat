#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ChromaDB vector databases for original and finetuned models
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.embedder import EmbeddingModel
from dataloader.loader import load_entire_dataset
from config import TrainingConfig
from train import clean_html_text
from utils.text_chunker import chunk_file


def load_eval_data(dataset_root: str, config: TrainingConfig) -> List[Tuple[str, str]]:
    """
    Load evaluation dataset as (query, answer) pairs

    Args:
        dataset_root: Path to KorQuAD dataset
        config: Training configuration

    Returns:
        List of (query, answer) tuples
    """
    print("\n" + "=" * 60)
    print("Loading KorQuAD Dataset")
    print("=" * 60)

    # Load dataset
    dataset = load_entire_dataset(dataset_root, verbose=True)
    if not dataset:
        raise ValueError(f"Failed to load dataset from {dataset_root}")

    articles = dataset['articles']

    # Extract Q-A pairs
    qa_pairs = []
    for article in tqdm(articles, desc="Extracting Q-A pairs"):
        for qa in article.get('qas', []):
            question = qa.get('question', '')
            answer_obj = qa.get('answer', {})

            # Extract answer text
            if isinstance(answer_obj, dict):
                answer = answer_obj.get('text', '')
            elif isinstance(answer_obj, str):
                answer = answer_obj
            else:
                answer = ''

            if not question or not answer:
                continue

            # Clean HTML
            answer = clean_html_text(answer)
            if not answer:
                continue

            qa_pairs.append((question, answer))

    print(f"\nTotal Q-A pairs: {len(qa_pairs)}")

    # Use same split as training
    split_idx = int(len(qa_pairs) * config.train_split)
    eval_pairs = qa_pairs[split_idx:]

    # Limit to eval_samples
    eval_pairs = eval_pairs[:config.eval_samples]

    print(f"Eval Q-A pairs: {len(eval_pairs)}")

    return eval_pairs


def load_wiki_data(
    text_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    max_chunks: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Load and chunk wiki text file

    Args:
        text_file: Path to cleaned wiki text file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        max_chunks: Maximum number of chunks to return (None for all)

    Returns:
        List of (title, content) tuples where title is section heading
    """
    print("\n" + "=" * 60)
    print("Loading Wiki Text")
    print("=" * 60)
    print(f"File: {text_file}")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    # Get document name from file
    doc_name = Path(text_file).stem

    # Chunk the file
    chunks = chunk_file(
        text_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy="standard",  # Can be changed to "heading" if text has clear headings
        metadata={'document': doc_name}
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # Convert chunks to (title, content) tuples for consistency with QA format
    # Use chunk_index as title
    chunk_pairs = []
    for chunk_data in chunks:
        text = chunk_data['text']
        metadata = chunk_data['metadata']

        # Create title from chunk info
        chunk_idx = metadata.get('chunk_index', 0)
        heading = metadata.get('heading', '')

        if heading:
            title = f"{doc_name} - {heading} (Part {chunk_idx + 1})"
        else:
            title = f"{doc_name} - Chunk {chunk_idx + 1}"

        # Text is the content
        content = text

        chunk_pairs.append((title, content))

    # Limit chunks if specified
    if max_chunks and max_chunks > 0:
        chunk_pairs = chunk_pairs[:max_chunks]
        print(f"Limited to: {len(chunk_pairs)} chunks")

    return chunk_pairs


def build_chromadb(
    collection_name: str,
    qa_pairs: List[Tuple[str, str]],
    embedder: EmbeddingModel,
    db_path: str = "./chroma_db"
) -> chromadb.Collection:
    """
    Build ChromaDB collection with embeddings

    Args:
        collection_name: Name of the collection
        qa_pairs: List of (query, answer) tuples
        embedder: Embedding model
        db_path: Path to ChromaDB storage

    Returns:
        ChromaDB collection
    """
    print(f"\n{'=' * 60}")
    print(f"Building ChromaDB: {collection_name}")
    print(f"{'=' * 60}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete collection if exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Prepare data
    print("\nPreparing embeddings...")
    documents = []
    metadatas = []
    ids = []

    for idx, (query, answer) in enumerate(tqdm(qa_pairs, desc="Formatting data")):
        # Document text: "query: answer" format
        doc_text = f"{query}: {answer}"
        documents.append(doc_text)

        # Metadata to store original query and answer separately
        metadatas.append({
            "query": query,
            "answer": answer,
            "index": idx
        })

        ids.append(f"doc_{idx}")

    # Generate embeddings
    print(f"\nGenerating embeddings with {embedder.model_type} model...")

    # For documents, use "passage:" prefix (E5 convention)
    prefixed_docs = [f"passage: {doc}" for doc in documents]

    embeddings = embedder.encode(
        prefixed_docs,
        batch_size=32,
        show_progress=True
    )

    # Convert to list of lists for ChromaDB
    embeddings_list = embeddings.cpu().numpy().tolist()

    # Add to collection
    print("\nAdding to ChromaDB...")
    batch_size = 1000
    for i in tqdm(range(0, len(documents), batch_size), desc="Inserting batches"):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings_list[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

    print(f"\n✅ Collection '{collection_name}' created with {len(documents)} documents")

    return collection


def main():
    """Main function to build a single ChromaDB collection"""
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB collection for embedding model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["korquad", "wiki"],
        default="korquad",
        help="Data source mode: 'korquad' for KorQuAD dataset, 'wiki' for wiki text file"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/d/datasets/KorQuAD",
        help="Path to KorQuAD dataset (for korquad mode)"
    )
    parser.add_argument(
        "--wiki-file",
        type=str,
        default="data/text_cleaned.txt",
        help="Path to wiki text file (for wiki mode)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for wiki text (wiki mode only)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for wiki text (wiki mode only)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to use (wiki mode only, None for all)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to embedding model (HuggingFace name or local path)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["original", "finetuned"],
        help="Type of model: 'original' or 'finetuned'"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="ChromaDB collection name (e.g., 'original_embeddings')"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Path to store ChromaDB"
    )
    args = parser.parse_args()

    # Load data based on mode
    if args.mode == "wiki":
        data_pairs = load_wiki_data(
            text_file=args.wiki_file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks
        )
        data_type = "Wiki Text Chunks"
    else:  # korquad mode
        config = TrainingConfig()
        data_pairs = load_eval_data(args.dataset_root, config)
        data_type = "KorQuAD Q-A Pairs"

    # Build database for specified model
    print("\n" + "=" * 60)
    print(f"Building Collection: {args.collection_name}")
    print(f"Data Type: {data_type}")
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Model Path: {args.model_path}")
    print("=" * 60)

    embedder = EmbeddingModel(
        model_path=args.model_path,
        model_type=args.model_type
    )

    build_chromadb(
        collection_name=args.collection_name,
        qa_pairs=data_pairs,
        embedder=embedder,
        db_path=args.db_path
    )

    print("\n" + "=" * 60)
    print("✅ Collection build complete!")
    print("=" * 60)
    print(f"Database location: {args.db_path}")
    print(f"Collection: {args.collection_name} ({len(data_pairs)} documents)")
    print(f"Mode: {args.mode}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
