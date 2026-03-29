"""
Table-Aware Chunker
───────────────────
Splits markdown into chunks while keeping tables intact.

Problem with naive chunking:
  A RecursiveCharacterTextSplitter doesn't know what a markdown table
  is — it'll happily split a table in the middle of a row, losing
  structure and making the data useless for retrieval.

How this works:
  1. Scan the markdown and split it into segments — alternating between
     "prose" blocks and "table" blocks.
  2. Each table block includes the paragraph immediately above it
     (the description/caption) so the table retains its context.
  3. Prose blocks are chunked normally with RecursiveCharacterTextSplitter.
  4. Table blocks are kept whole. If a table exceeds the chunk size,
     it's split by rows with the header row repeated in every chunk.
  5. All chunks are reassembled in document order.
"""

import re
import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_file: str


# ── Markdown table detection ───────────────────────────────────────

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _extract_segments(markdown: str) -> list[dict]:
    """
    Walk through lines and split the document into typed segments:
      {"type": "prose",  "text": "..."}
      {"type": "table",  "text": "...", "description": "..."}

    The description is the last non-empty paragraph before a table.
    """
    lines = markdown.split("\n")
    segments = []
    current_prose_lines = []
    current_table_lines = []
    in_table = False

    for line in lines:
        if _is_table_row(line):
            if not in_table:
                # Starting a new table — pull the description from
                # the trailing prose (last paragraph before the table)
                description = _extract_last_paragraph(current_prose_lines)

                # Save prose before this table (minus the description)
                remaining_prose = _remove_last_paragraph(current_prose_lines)
                if remaining_prose.strip():
                    segments.append({"type": "prose", "text": remaining_prose})

                current_table_lines = []
                in_table = True

            current_table_lines.append(line)

        else:
            if in_table:
                # Table just ended
                table_text = "\n".join(current_table_lines)
                segments.append({
                    "type": "table",
                    "text": table_text,
                    "description": description,
                })
                current_table_lines = []
                current_prose_lines = []
                in_table = False

            current_prose_lines.append(line)

    # Flush remaining content
    if in_table and current_table_lines:
        table_text = "\n".join(current_table_lines)
        segments.append({
            "type": "table",
            "text": table_text,
            "description": description,
        })
    elif current_prose_lines:
        prose = "\n".join(current_prose_lines)
        if prose.strip():
            segments.append({"type": "prose", "text": prose})

    return segments


def _extract_last_paragraph(lines: list[str]) -> str:
    """
    Pull the last non-empty paragraph from a list of lines.
    A paragraph is a contiguous block of non-empty lines at the end.
    """
    # Strip trailing empty lines
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()

    if not trimmed:
        return ""

    # Walk backwards to find paragraph start
    para_lines = []
    for line in reversed(trimmed):
        if not line.strip():
            break
        para_lines.append(line)

    para_lines.reverse()
    return "\n".join(para_lines)


def _remove_last_paragraph(lines: list[str]) -> str:
    """
    Return all lines except the last paragraph.
    """
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()

    # Remove the last paragraph
    while trimmed and trimmed[-1].strip():
        trimmed.pop()

    return "\n".join(trimmed)


# ── Table splitting (for oversized tables) ─────────────────────────

def _split_large_table(
    table_text: str,
    description: str,
    chunk_size: int,
) -> list[str]:
    """
    If a table + description fits within chunk_size, return it as one chunk.
    Otherwise, split by rows, repeating the header + separator in each chunk.
    """
    full_text = f"{description}\n\n{table_text}".strip() if description else table_text

    if len(full_text) <= chunk_size:
        return [full_text]

    rows = table_text.split("\n")

    # Find header and separator rows (first two lines of a markdown table)
    header = rows[0] if rows else ""
    separator = rows[1] if len(rows) > 1 and _SEPARATOR_RE.match(rows[1]) else ""
    data_start = 2 if separator else 1
    data_rows = rows[data_start:]

    if not data_rows:
        return [full_text]

    # Build the prefix that goes in every chunk
    table_header = f"{header}\n{separator}" if separator else header
    prefix = f"{description}\n\n{table_header}" if description else table_header

    chunks = []
    current_rows = []
    current_len = len(prefix)

    for row in data_rows:
        row_len = len(row) + 1  # +1 for newline
        if current_len + row_len > chunk_size and current_rows:
            chunk = prefix + "\n" + "\n".join(current_rows)
            chunks.append(chunk)
            current_rows = []
            current_len = len(prefix)

        current_rows.append(row)
        current_len += row_len

    # Flush remaining rows
    if current_rows:
        chunk = prefix + "\n" + "\n".join(current_rows)
        chunks.append(chunk)

    return chunks


# ── Public API ──────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split markdown into chunks with table-awareness.

    - Prose sections are split with RecursiveCharacterTextSplitter.
    - Tables are kept intact with their description prepended.
    - Oversized tables are split by rows with headers repeated.
    """
    segments = _extract_segments(text)
    logger.info(
        "Segmented document into %d parts (%d tables, %d prose)",
        len(segments),
        sum(1 for s in segments if s["type"] == "table"),
        sum(1 for s in segments if s["type"] == "prose"),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks: list[str] = []

    for segment in segments:
        if segment["type"] == "prose":
            prose_chunks = splitter.split_text(segment["text"])
            raw_chunks.extend(prose_chunks)

        elif segment["type"] == "table":
            table_chunks = _split_large_table(
                segment["text"],
                segment.get("description", ""),
                chunk_size,
            )
            raw_chunks.extend(table_chunks)

    # DEBUG
    with open("chunk.txt", "w") as f: 
        for i, chunk in enumerate(raw_chunks):
            f.write(f"\nChunk: {i:02d}\n")
            f.write(chunk)

    return [
        Chunk(text=c, chunk_index=i, source_file=source_file)
        for i, c in enumerate(raw_chunks)
        if c.strip()
    ]