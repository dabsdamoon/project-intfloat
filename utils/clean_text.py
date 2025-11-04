import re
import argparse
from pathlib import Path


class TextCleaner:
    """
    A utility class for cleaning extracted text from web pages (especially wiki-style pages).
    Removes common artifacts like page markers, footers, navigation elements, and excessive whitespace.
    """

    def __init__(self):
        self.patterns = {
            # Page markers (e.g., "--- Page 1 ---")
            'page_markers': r'^---\s*Page\s+\d+\s*---\s*$',

            # Date/time stamps (e.g., "25. 11. 4. 오후 5:57")
            'datetime_stamps': r'^\d{2}\.\s+\d{1,2}\.\s+\d{1,2}\.\s+(오전|오후)\s+\d{1,2}:\d{2}\s*$',

            # URLs
            'urls': r'^https?://[^\s]+$',

            # Page numbers (e.g., "1/54", "2/54")
            'page_numbers': r'^\d+/\d+\s*$',

            # Wiki navigation elements (e.g., "[ 펼치기 · 접기 ]")
            'wiki_nav': r'\[\s*[^]]*펼치기\s*·\s*접기\s*[^]]*\]',

            # Reference numbers in square brackets (e.g., [1], [20])
            'reference_numbers': r'\[\d+\]',

            # Standalone site name patterns
            'site_names': r'^.*(나무위키|위키백과|Wikipedia).*$',

            # Multiple consecutive blank lines
            'multiple_blanks': r'\n{3,}',
        }

    def remove_page_markers(self, text):
        """Remove page separator markers like '--- Page 1 ---'"""
        return re.sub(self.patterns['page_markers'], '', text, flags=re.MULTILINE)

    def remove_datetime_stamps(self, text):
        """Remove date/time stamps"""
        return re.sub(self.patterns['datetime_stamps'], '', text, flags=re.MULTILINE)

    def remove_urls(self, text):
        """Remove standalone URLs"""
        return re.sub(self.patterns['urls'], '', text, flags=re.MULTILINE)

    def remove_page_numbers(self, text):
        """Remove page number indicators like '1/54'"""
        return re.sub(self.patterns['page_numbers'], '', text, flags=re.MULTILINE)

    def remove_wiki_navigation(self, text):
        """Remove wiki navigation elements like expand/collapse buttons"""
        return re.sub(self.patterns['wiki_nav'], '', text)

    def remove_reference_numbers(self, text):
        """Remove reference numbers in square brackets like [1], [2]"""
        return re.sub(self.patterns['reference_numbers'], '', text)

    def remove_site_names(self, text):
        """Remove lines containing only site names"""
        return re.sub(self.patterns['site_names'], '', text, flags=re.MULTILINE)

    def remove_duplicate_lines(self, text):
        """Remove consecutive duplicate lines"""
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = None

        for line in lines:
            # Keep the line if it's different from the previous one
            if line.strip() != prev_line:
                cleaned_lines.append(line)
                prev_line = line.strip()

        return '\n'.join(cleaned_lines)

    def normalize_whitespace(self, text):
        """Normalize excessive whitespace"""
        # Replace multiple consecutive blank lines with just two newlines
        text = re.sub(self.patterns['multiple_blanks'], '\n\n', text)

        # Remove trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]

        return '\n'.join(lines)

    def remove_leading_trailing_blanks(self, text):
        """Remove leading and trailing blank lines"""
        return text.strip()

    def clean_all(self, text, options=None):
        """
        Apply all cleaning operations to the text.

        Args:
            text (str): The text to clean
            options (dict): Optional dict to enable/disable specific cleaners
                           If None, all cleaners are enabled by default

        Returns:
            str: Cleaned text
        """
        if options is None:
            options = {
                'page_markers': True,
                'datetime_stamps': True,
                'urls': True,
                'page_numbers': True,
                'wiki_navigation': True,
                'reference_numbers': True,
                'site_names': True,
                'duplicate_lines': True,
                'normalize_whitespace': True,
            }

        cleaning_steps = [
            ('page_markers', self.remove_page_markers),
            ('datetime_stamps', self.remove_datetime_stamps),
            ('urls', self.remove_urls),
            ('page_numbers', self.remove_page_numbers),
            ('wiki_navigation', self.remove_wiki_navigation),
            ('reference_numbers', self.remove_reference_numbers),
            ('site_names', self.remove_site_names),
            ('duplicate_lines', self.remove_duplicate_lines),
            ('normalize_whitespace', self.normalize_whitespace),
        ]

        cleaned_text = text
        for step_name, step_func in cleaning_steps:
            if options.get(step_name, True):
                cleaned_text = step_func(cleaned_text)

        cleaned_text = self.remove_leading_trailing_blanks(cleaned_text)

        return cleaned_text


def clean_file(input_path, output_path=None, options=None):
    """
    Clean a text file and save the result.

    Args:
        input_path (str): Path to input file
        output_path (str): Path to output file (if None, creates *_cleaned.txt)
        options (dict): Cleaning options to pass to TextCleaner

    Returns:
        tuple: (original_size, cleaned_size, output_path)
    """
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    original_size = len(original_text)

    # Clean text
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_all(original_text, options)

    cleaned_size = len(cleaned_text)

    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"

    # Save cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    return original_size, cleaned_size, output_path


def main():
    parser = argparse.ArgumentParser(description='Clean extracted text from web pages')
    parser.add_argument('input_file', type=str, help='Path to input text file')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to output file (default: input_file_cleaned.txt)')
    parser.add_argument('--keep-references', action='store_true',
                        help='Keep reference numbers [1], [2], etc.')
    parser.add_argument('--keep-urls', action='store_true',
                        help='Keep URLs')

    args = parser.parse_args()

    # Build options based on arguments
    options = {
        'reference_numbers': not args.keep_references,
        'urls': not args.keep_urls,
    }

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Clean the file
    print(f"Cleaning file: {args.input_file}")
    original_size, cleaned_size, output_path = clean_file(
        args.input_file,
        args.output_file,
        options
    )

    # Report results
    reduction = original_size - cleaned_size
    reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

    print(f"\nCleaning completed!")
    print(f"Original size: {original_size:,} characters")
    print(f"Cleaned size:  {cleaned_size:,} characters")
    print(f"Reduction:     {reduction:,} characters ({reduction_pct:.1f}%)")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
