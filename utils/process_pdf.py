#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined PDF text extraction and cleaning pipeline.
Extracts text from PDF and applies cleaning operations in one step.
"""

import os
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from clean_text import TextCleaner


def process_pdf(
    pdf_path: str,
    output_file: str = None,
    output_dir: str = "data",
    verbose: bool = True
):
    """
    Extract text from PDF and clean it in one step.

    Args:
        pdf_path: Path to PDF file
        output_file: Path to output cleaned text file (default: data/text_cleaned.txt)
        output_dir: Directory to save output (used if output_file not specified)
        verbose: Print progress messages

    Returns:
        str: Path to the cleaned output file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set default output file
    if output_file is None:
        output_file = os.path.join(output_dir, "text_cleaned.txt")

    if verbose:
        print("=" * 60)
        print("PDF Processing Pipeline")
        print("=" * 60)
        print(f"Input PDF: {pdf_path}")
        print(f"Output file: {output_file}")
        print()

    # Step 1: Extract text from PDF
    if verbose:
        print("Step 1: Extracting text from PDF...")
        print("-" * 60)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return None

    # Extract text from all pages
    all_text = []
    page_count = len(doc)

    if verbose:
        print(f"Processing {page_count} pages...")

    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        all_text.append(f"--- Page {page_num + 1} ---\n{text}\n")

        if verbose:
            print(f"  Page {page_num + 1}/{page_count}", end='\r')

    doc.close()

    if verbose:
        print(f"\n  Extracted {len(''.join(all_text))} characters")

    # Combine all text
    raw_text = "\n".join(all_text)

    # Step 2: Clean the text
    if verbose:
        print()
        print("Step 2: Cleaning extracted text...")
        print("-" * 60)

    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_all(raw_text)

    original_size = len(raw_text)
    cleaned_size = len(cleaned_text)
    reduction = original_size - cleaned_size
    reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

    if verbose:
        print(f"  Original size: {original_size:,} characters")
        print(f"  Cleaned size:  {cleaned_size:,} characters")
        print(f"  Reduction:     {reduction:,} characters ({reduction_pct:.1f}%)")

    # Step 3: Save cleaned text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    if verbose:
        print()
        print("=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)
        print(f"Cleaned text saved to: {output_file}")

    return output_file


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Extract text from PDF and clean it in one step"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output cleaned text file (default: data/text_cleaned.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save output if output-file not specified (default: data)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1

    # Process PDF
    output_path = process_pdf(
        pdf_path=args.pdf_path,
        output_file=args.output_file,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    if output_path is None:
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
