"""
KorQuAD Dataset Loader Package
"""

from .loader import (
    load_json_file,
    load_file_worker,
    load_files_parallel,
    accumulate_all_data,
    get_all_json_files,
    load_entire_dataset
)

__all__ = [
    'load_json_file',
    'load_file_worker',
    'load_files_parallel',
    'accumulate_all_data',
    'get_all_json_files',
    'load_entire_dataset'
]
