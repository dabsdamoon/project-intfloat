#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorQuAD Dataset Analysis Script
Analyzes Korean Question Answering Dataset (KorQuAD) JSON files
"""

import json
import statistics
from pathlib import Path
from multiprocessing import cpu_count
from tqdm import tqdm

# Import the loader module
from dataloader import loader

# Configuration
DATASET_ROOT = Path("/mnt/d/datasets/KorQuAD")


def load_json_file(file_path):
    """Load JSON file with proper UTF-8 encoding for Korean text"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_statistics_from_dataset(dataset):
    """Compute comprehensive statistics from accumulated dataset

    Args:
        dataset: Dict returned from loader.accumulate_all_data()

    Returns:
        dict: Statistics including question/answer length distributions
    """
    articles = dataset['articles']

    question_lengths = []
    answer_lengths = []
    answer_start_positions = []

    for article in articles:
        qas_list = article.get('qas', [])

        for qa in qas_list:
            question = qa.get('question', '')
            answer = qa.get('answer', {})

            question_lengths.append(len(question))
            answer_text = answer.get('text', '')
            answer_lengths.append(len(answer_text))
            answer_start = answer.get('answer_start', 0)
            answer_start_positions.append(answer_start)

    return {
        'question_lengths': question_lengths,
        'answer_lengths': answer_lengths,
        'answer_start_positions': answer_start_positions,
        'min_question_length': min(question_lengths) if question_lengths else 0,
        'max_question_length': max(question_lengths) if question_lengths else 0,
        'avg_question_length': statistics.mean(question_lengths) if question_lengths else 0,
        'median_question_length': statistics.median(question_lengths) if question_lengths else 0,
        'min_answer_length': min(answer_lengths) if answer_lengths else 0,
        'max_answer_length': max(answer_lengths) if answer_lengths else 0,
        'avg_answer_length': statistics.mean(answer_lengths) if answer_lengths else 0,
        'median_answer_length': statistics.median(answer_lengths) if answer_lengths else 0,
    }


def analyze_single_file(file_path):
    """Analyze a single KorQuAD JSON file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path.name}")
    print(f"{'='*80}")

    data = load_json_file(file_path)
    if not data:
        return None

    # Basic statistics
    version = data.get('version', 'Unknown')
    articles = data.get('data', [])

    print(f"\nVersion: {version}")
    print(f"Number of articles: {len(articles)}")

    # Analyze questions and answers
    total_qas = 0
    question_lengths = []
    answer_lengths = []
    answer_start_positions = []

    for article in articles:
        qas_list = article.get('qas', [])
        total_qas += len(qas_list)

        for qa in qas_list:
            question = qa.get('question', '')
            answer = qa.get('answer', {})

            question_lengths.append(len(question))

            answer_text = answer.get('text', '')
            answer_lengths.append(len(answer_text))

            answer_start = answer.get('answer_start', 0)
            answer_start_positions.append(answer_start)

    print(f"Total Q&A pairs: {total_qas}")

    if question_lengths:
        print(f"\nQuestion Statistics:")
        print(f"  - Min length: {min(question_lengths)} chars")
        print(f"  - Max length: {max(question_lengths)} chars")
        print(f"  - Average length: {statistics.mean(question_lengths):.2f} chars")
        print(f"  - Median length: {statistics.median(question_lengths):.2f} chars")

    if answer_lengths:
        print(f"\nAnswer Statistics:")
        print(f"  - Min length: {min(answer_lengths)} chars")
        print(f"  - Max length: {max(answer_lengths)} chars")
        print(f"  - Average length: {statistics.mean(answer_lengths):.2f} chars")
        print(f"  - Median length: {statistics.median(answer_lengths):.2f} chars")

    # Sample examples
    print(f"\n{'─'*80}")
    print("Sample Examples:")
    print(f"{'─'*80}")

    if articles and len(articles) > 0:
        sample_article = articles[0]
        print(f"\nArticle Title: {sample_article.get('title', 'N/A')}")
        print(f"URL: {sample_article.get('url', 'N/A')}")

        sample_qas = sample_article.get('qas', [])
        if sample_qas:
            for i, qa in enumerate(sample_qas[:3], 1):  # Show first 3 Q&As
                print(f"\n  Example {i}:")
                print(f"    Question: {qa.get('question', 'N/A')}")
                answer = qa.get('answer', {})
                print(f"    Answer: {answer.get('text', 'N/A')}")
                print(f"    Answer Start Position: {answer.get('answer_start', 'N/A')}")

    return {
        'file': file_path.name,
        'version': version,
        'articles': len(articles),
        'total_qas': total_qas,
        'avg_question_length': statistics.mean(question_lengths) if question_lengths else 0,
        'avg_answer_length': statistics.mean(answer_lengths) if answer_lengths else 0
    }


def analyze_directory(directory_path):
    """Analyze all JSON files in a directory"""
    json_files = list(directory_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return []

    print(f"\nFound {len(json_files)} JSON files in {directory_path.name}")

    results = []
    for json_file in tqdm(sorted(json_files), total=len(json_files)):
        result = analyze_single_file(json_file)
        if result:
            results.append(result)

    return results


def analyze_all_directories(num_workers=None):
    """Analyze all KorQuAD training directories using multiprocessing

    Uses the loader module to load all files in parallel, then computes
    statistics from the entire accumulated dataset.

    Args:
        num_workers: Number of worker processes (default: CPU count)
    """
    print(f"\n{'='*80}")
    print("KorQuAD Dataset Full Analysis (Multiprocessing)")
    print(f"{'='*80}")
    print(f"Dataset Root: {DATASET_ROOT}\n")

    # Load entire dataset using multiprocessing loader
    dataset = loader.load_entire_dataset(
        dataset_root=DATASET_ROOT,
        num_workers=num_workers,
        verbose=True
    )

    if not dataset:
        print("Failed to load dataset!")
        return

    # Compute statistics from the whole dataset
    print("Computing statistics from entire dataset...")
    stats = compute_statistics_from_dataset(dataset)

    # Display summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal JSON files analyzed: {dataset['total_files']}")
    print(f"Total articles: {dataset['total_articles']:,}")
    print(f"Total Q&A pairs: {dataset['total_qas']:,}")

    print(f"\nQuestion Statistics:")
    print(f"  - Min length: {stats['min_question_length']} chars")
    print(f"  - Max length: {stats['max_question_length']} chars")
    print(f"  - Average length: {stats['avg_question_length']:.2f} chars")
    print(f"  - Median length: {stats['median_question_length']:.2f} chars")

    print(f"\nAnswer Statistics:")
    print(f"  - Min length: {stats['min_answer_length']} chars")
    print(f"  - Max length: {stats['max_answer_length']} chars")
    print(f"  - Average length: {stats['avg_answer_length']:.2f} chars")
    print(f"  - Median length: {stats['median_answer_length']:.2f} chars")

    print(f"\n{'='*80}\n")

    return dataset, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze KorQuAD dataset JSON files with comprehensive statistics (with multiprocessing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Full analysis using all CPU cores
  %(prog)s --workers 4             # Full analysis using 4 workers
  %(prog)s --file path.json        # Analyze specific JSON file
        """
    )

    parser.add_argument(
        '--file',
        type=str,
        metavar='PATH',
        help='Path to specific JSON file to analyze'
    )

    parser.add_argument(
        '--workers',
        type=int,
        metavar='N',
        default=4,
        help=f'Number of worker processes (default: {cpu_count()} - CPU count)'
    )

    args = parser.parse_args()

    # Execute based on arguments
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            analyze_single_file(file_path)
        else:
            print(f"Error: File not found: {file_path}")
            exit(1)
    else:
        analyze_all_directories(num_workers=args.workers)

    exit(0)
