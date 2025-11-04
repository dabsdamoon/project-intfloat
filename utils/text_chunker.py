"""
Text chunking utilities for splitting long documents into smaller chunks.
Supports various chunking strategies for RAG applications.
"""

import re
from typing import List, Dict, Tuple, Optional


class TextChunker:
    """
    Utility class for chunking text documents.
    Supports multiple chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator for splitting (default: double newline)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Chunk text using the configured strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if not text.strip():
            return []

        # Split by separator first
        sections = text.split(self.separator)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If section fits in current chunk, add it
            if len(current_chunk) + len(section) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + section
                else:
                    current_chunk = section
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunk_meta = metadata.copy() if metadata else {}
                    chunk_meta['chunk_index'] = chunk_index
                    chunk_meta['chunk_size'] = len(current_chunk)

                    chunks.append({
                        'text': current_chunk,
                        'metadata': chunk_meta
                    })
                    chunk_index += 1

                # If section is too large, split it further
                if len(section) > self.chunk_size:
                    sub_chunks = self._split_large_section(section)
                    for sub_chunk in sub_chunks:
                        chunk_meta = metadata.copy() if metadata else {}
                        chunk_meta['chunk_index'] = chunk_index
                        chunk_meta['chunk_size'] = len(sub_chunk)

                        chunks.append({
                            'text': sub_chunk,
                            'metadata': chunk_meta
                        })
                        chunk_index += 1

                    # Start new chunk with overlap
                    if sub_chunks:
                        overlap_text = sub_chunks[-1][-self.chunk_overlap:] if len(sub_chunks[-1]) > self.chunk_overlap else sub_chunks[-1]
                        current_chunk = overlap_text
                    else:
                        current_chunk = ""
                else:
                    # Add overlap from previous chunk
                    if chunks and self.chunk_overlap > 0:
                        prev_text = chunks[-1]['text']
                        overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text
                        current_chunk = overlap_text + self.separator + section
                    else:
                        current_chunk = section

        # Add final chunk
        if current_chunk.strip():
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta['chunk_index'] = chunk_index
            chunk_meta['chunk_size'] = len(current_chunk)

            chunks.append({
                'text': current_chunk,
                'metadata': chunk_meta
            })

        return chunks

    def _split_large_section(self, text: str) -> List[str]:
        """
        Split a large section that exceeds chunk_size.

        Args:
            text: Large text section

        Returns:
            List of smaller chunks
        """
        # Try splitting by sentences first
        sentences = re.split(r'([.!?。！？]+\s*)', text)

        # Reconstruct sentences with their punctuation
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed.append(sentences[i])

        if len(sentences) % 2 == 1:
            reconstructed.append(sentences[-1])

        # Combine sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in reconstructed:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single sentence is too long, split it by character
                if len(sentence) > self.chunk_size:
                    for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                        chunk_part = sentence[i:i + self.chunk_size]
                        chunks.append(chunk_part)
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_by_heading(
        self,
        text: str,
        heading_pattern: str = r'^#+\s+(.+)$',
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Chunk text by markdown-style headings.

        Args:
            text: Text to chunk
            heading_pattern: Regex pattern to identify headings
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with heading information
        """
        lines = text.split('\n')
        chunks = []
        current_heading = None
        current_content = []
        chunk_index = 0

        for line in lines:
            # Check if line is a heading
            heading_match = re.match(heading_pattern, line, re.MULTILINE)

            if heading_match:
                # Save previous chunk if exists
                if current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        chunk_meta = metadata.copy() if metadata else {}
                        chunk_meta['heading'] = current_heading or "Introduction"
                        chunk_meta['chunk_index'] = chunk_index
                        chunk_meta['chunk_size'] = len(content_text)

                        chunks.append({
                            'text': content_text,
                            'metadata': chunk_meta
                        })
                        chunk_index += 1

                # Start new section
                current_heading = heading_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final chunk
        if current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta['heading'] = current_heading or "Content"
                chunk_meta['chunk_index'] = chunk_index
                chunk_meta['chunk_size'] = len(content_text)

                chunks.append({
                    'text': content_text,
                    'metadata': chunk_meta
                })

        return chunks


def chunk_file(
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str = "standard",
    metadata: Optional[Dict] = None
) -> List[Dict[str, any]]:
    """
    Chunk a text file.

    Args:
        file_path: Path to text file
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy ("standard" or "heading")
        metadata: Base metadata to attach to chunks

    Returns:
        List of chunks with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Add file metadata
    base_metadata = metadata.copy() if metadata else {}
    base_metadata['source_file'] = file_path

    if strategy == "heading":
        chunks = chunker.chunk_by_heading(text, metadata=base_metadata)
    else:
        chunks = chunker.chunk_text(text, metadata=base_metadata)

    return chunks


if __name__ == '__main__':
    # Example usage
    sample_text = """
1. 개요
볼드모트는 해리 포터 시리즈의 메인 악역입니다.

2. 특징
그는 강력한 마법사였으며 어둠의 마법을 사용했습니다.

3. 능력
볼드모트는 다양한 마법 능력을 보유하고 있었습니다.
"""

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text, metadata={'document': 'example'})

    print("Standard Chunking:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Text: {chunk['text'][:50]}...")
        print(f"  Metadata: {chunk['metadata']}")

    print("\n" + "=" * 60)

    heading_chunks = chunker.chunk_by_heading(sample_text, metadata={'document': 'example'})
    print("\nHeading-based Chunking:")
    for i, chunk in enumerate(heading_chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Heading: {chunk['metadata'].get('heading', 'N/A')}")
        print(f"  Text: {chunk['text'][:50]}...")
