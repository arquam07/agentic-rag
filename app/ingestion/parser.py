"""
Document Parser — uses Docling to convert files to markdown.

Two tiers:
  • basic   — fast, no image handling, tables + text only
  • premium — uses Docling's vision model to generate image
              descriptions and embed them in the markdown
"""

import tempfile
import os
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

logger = logging.getLogger(__name__)


def _build_converter(tier: str) -> DocumentConverter:
    """
    Build a Docling converter based on the tier.

    basic:   no OCR, no image descriptions, table structure enabled
    premium: OCR on, image descriptions via vision model, table structure on
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True

    if tier == "premium":
        pipeline_options.do_ocr = True
        pipeline_options.generate_picture_images = True

        # Enable image description via Docling's vision pipeline
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions

        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = PictureDescriptionApiOptions()

        logger.info("Premium tier: OCR + image descriptions enabled")
    else:
        pipeline_options.do_ocr = False
        logger.info("Basic tier: text + tables only")

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_document(file_bytes: bytes, filename: str, tier: str = "basic") -> str:
    """
    Accept raw file bytes, write to a temp file, run Docling,
    return clean markdown string.

    Args:
        file_bytes: raw file content
        filename:   original filename (used for extension detection)
        tier:       "basic" or "premium"
    """
    if tier not in ("basic", "premium"):
        raise ValueError(f"Unknown tier '{tier}'. Use 'basic' or 'premium'.")

    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        converter = _build_converter(tier)
        result = converter.convert(tmp_path)
        markdown = result.document.export_to_markdown()
        return markdown
    finally:
        os.unlink(tmp_path)