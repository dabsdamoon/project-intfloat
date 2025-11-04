#!/bin/bash

# Script to extract text from PDF and clean it in one step
# Uses combined processor: utils/process_pdf.py
# Final output: data/text_cleaned.txt

set -e  # Exit on error

# PDF file path
PDF_PATH="tmp/볼드모트 - 나무위키.pdf"

# Output file
OUTPUT_FILE="data/text_cleaned.txt"

# Run the combined PDF processor
python3 utils/process_pdf.py "$PDF_PATH" --output-file "$OUTPUT_FILE"

echo ""
echo "Next steps:"
echo "  Build database: ./run_build_wiki_database.sh"
echo "  Launch demo:    ./run_demo_wiki.sh"
echo ""
