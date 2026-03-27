import tempfile
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Accept raw file bytes, write to a temp file, run Docling,
    return clean markdown string.
    """
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False          # flip to True for scanned PDFs
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(tmp_path)
        markdown = result.document.export_to_markdown()
        return markdown

    finally:
        os.unlink(tmp_path)