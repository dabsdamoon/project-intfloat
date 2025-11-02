#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorQuAD Dataset Loader Module
Provides multiprocessing-enabled loading of KorQuAD JSON files
"""

import json
from pathlib import Path
from multiprocessing import Pool, cpu_count


def load_json_file(file_path):
    """Load a single JSON file with proper UTF-8 encoding for Korean text

    Args:
        file_path: Path to JSON file

    Returns:
        dict: Loaded JSON data, or None if error occurs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_file_worker(file_path):
    """Worker function to load a single file

    Args:
        file_path: Path to JSON file

    Returns:
        tuple: (file_path, data) where data is the loaded JSON or None
    """
    data = load_json_file(file_path)
    return (file_path, data)


def load_files_parallel(file_paths, num_workers=None):
    """Load multiple JSON files in parallel using multiprocessing

    Args:
        file_paths: List of file paths to load
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        list: List of tuples (file_path, data) for successfully loaded files
    """
    if num_workers is None:
        num_workers = cpu_count()

    if not file_paths:
        return []

    # Use multiprocessing Pool to load files in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(load_file_worker, file_paths)

    # Filter out failed loads (where data is None)
    successful_loads = [(path, data) for path, data in results if data is not None]

    return successful_loads


def accumulate_all_data(loaded_files):
    """Accumulate all articles and QA pairs from loaded files into single dataset

    Args:
        loaded_files: List of tuples (file_path, data) from load_files_parallel

    Returns:
        dict: Accumulated dataset with structure:
            {
                'files': list of file paths,
                'versions': list of versions,
                'articles': list of all articles from all files,
                'total_files': count,
                'total_articles': count,
                'total_qas': count
            }
    """
    all_articles = []
    all_versions = []
    file_paths = []
    total_qas = 0

    for file_path, data in loaded_files:
        file_paths.append(file_path)
        version = data.get('version', 'Unknown')
        all_versions.append(version)

        articles = data.get('data', [])
        all_articles.extend(articles)

        # Count QAs in this file
        for article in articles:
            total_qas += len(article.get('qas', []))

    return {
        'files': file_paths,
        'versions': all_versions,
        'articles': all_articles,
        'total_files': len(file_paths),
        'total_articles': len(all_articles),
        'total_qas': total_qas
    }


def get_all_json_files(dataset_root):
    """Get all JSON files from all KorQuAD training directories

    Args:
        dataset_root: Path to KorQuAD dataset root directory

    Returns:
        list: Sorted list of Path objects for all JSON files
    """
    dataset_root = Path(dataset_root)
    json_files = []

    for train_dir in sorted(dataset_root.glob("KorQuAD_2.1_train_*")):
        if train_dir.is_dir():
            json_files.extend(train_dir.glob("*.json"))

    return sorted(json_files)


def load_entire_dataset(dataset_root, num_workers=None, verbose=True):
    """Load entire KorQuAD dataset using multiprocessing

    Args:
        dataset_root: Path to KorQuAD dataset root directory
        num_workers: Number of worker processes (default: CPU count)
        verbose: Print progress information

    Returns:
        dict: Accumulated dataset with all articles and metadata
    """
    if verbose:
        print(f"Collecting JSON file paths from {dataset_root}...")

    json_files = get_all_json_files(dataset_root)

    if not json_files:
        print(f"No JSON files found in {dataset_root}")
        return None

    if verbose:
        print(f"Found {len(json_files)} JSON files")
        if num_workers is None:
            num_workers = cpu_count()
        print(f"Loading files using {num_workers} worker processes...\n")

    # Load all files in parallel
    loaded_files = load_files_parallel(json_files, num_workers)

    if verbose:
        print(f"Successfully loaded {len(loaded_files)}/{len(json_files)} files")
        print("Accumulating data...\n")

    # Accumulate all data into single dataset
    dataset = accumulate_all_data(loaded_files)

    return dataset
