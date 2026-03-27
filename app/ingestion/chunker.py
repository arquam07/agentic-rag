from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_file: str


def chunk_text(text: str, source_file: str,
               chunk_size: int = 512,
               chunk_overlap: int = 64) -> list[Chunk]:
    """
    Split markdown text into overlapping chunks.
    Returns a list of Chunk dataclasses.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    return [
        Chunk(text=c, chunk_index=i, source_file=source_file)
        for i, c in enumerate(raw_chunks)
        if c.strip()          # drop any whitespace-only chunks
    ]