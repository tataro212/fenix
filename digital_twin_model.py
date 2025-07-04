"""
Digital Twin Document Model - Unified Data Schema

This module consolidates the three existing data model systems into a single,
authoritative "Digital Twin" representation that preserves document structure,
layout, and content relationships throughout the translation pipeline.

Architectural Principles:
1. Single Source of Truth: One canonical data structure for document representation
2. Structure Preservation: Maintains spatial relationships and layout information
3. Type Safety: Pydantic models ensure data integrity across pipeline stages
4. Translation Aware: Built-in support for bilingual content preservation
5. Performance Optimized: Efficient serialization for caching and parallel processing
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from enum import Enum
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Type aliases for clarity
BoundingBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
PageDimensions = Tuple[float, float]  # (width, height)

class BlockType(str, Enum):
    """Enumeration of all possible document block types"""
    # Text blocks
    TEXT = "text"
    PARAGRAPH = "paragraph" 
    HEADING = "heading"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    QUOTE = "quote"
    
    # Visual blocks
    IMAGE = "image"
    FIGURE = "figure"
    TABLE = "table"
    CHART = "chart"
    
    # Structural blocks
    PAGE_BREAK = "page_break"
    TOC_ENTRY = "toc_entry"
    EQUATION = "equation"
    CODE_BLOCK = "code_block"

class StructuralRole(str, Enum):
    """Semantic roles for enhanced document understanding"""
    CONTENT = "content"          # Main document content
    NAVIGATION = "navigation"    # TOC, indexes, page numbers
    METADATA = "metadata"        # Headers, footers, author info
    ILLUSTRATION = "illustration" # Images, figures, charts
    DATA = "data"               # Tables, equations
    ANNOTATION = "annotation"   # Footnotes, captions, comments

class BaseBlock(BaseModel):
    """Base class for all document blocks with essential properties"""
    
    block_id: str = Field(..., description="Unique identifier for this block")
    block_type: BlockType = Field(..., description="Type classification of this block")
    structural_role: StructuralRole = Field(default=StructuralRole.CONTENT, description="Semantic role in document")
    
    # Spatial properties
    bbox: BoundingBox = Field(..., description="Bounding box coordinates (x0, y0, x1, y1)")
    page_number: int = Field(..., ge=1, description="1-based page number")
    z_order: int = Field(default=0, description="Stacking order for overlapping elements")
    
    # Content properties
    original_text: str = Field(default="", description="Original extracted text content")
    translated_text: str = Field(default="", description="Translated text content")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence score")
    
    # Processing metadata
    extraction_method: str = Field(default="pymupdf", description="Method used to extract this block")
    processing_notes: List[str] = Field(default_factory=list, description="Processing annotations and warnings")
    
    @field_validator('block_id')
    @classmethod
    def validate_block_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("block_id cannot be empty")
        return v.strip()
    
    def generate_content_hash(self) -> str:
        """Generate hash for content-based deduplication"""
        content = f"{self.block_type.value}:{self.original_text}:{self.bbox}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def has_translation(self) -> bool:
        """Check if block has been translated"""
        return bool(self.translated_text and self.translated_text != self.original_text)
    
    def get_display_text(self, prefer_translation: bool = True) -> str:
        """Get appropriate text for display"""
        if prefer_translation and self.has_translation():
            return self.translated_text
        return self.original_text

class TextBlock(BaseBlock):
    """Represents textual content blocks with formatting information"""
    
    # Typography and formatting
    font_family: str = Field(default="", description="Font family name")
    font_size: float = Field(default=12.0, ge=0.0, description="Font size in points")
    font_weight: str = Field(default="normal", description="Font weight (normal, bold, etc.)")
    font_style: str = Field(default="normal", description="Font style (normal, italic, etc.)")
    text_color: Optional[str] = Field(default=None, description="Text color in hex format")
    
    # Structural properties
    heading_level: Optional[int] = Field(default=None, ge=1, le=6, description="Heading level (1-6)")
    list_level: Optional[int] = Field(default=None, ge=1, description="List nesting level")
    is_bold: bool = Field(default=False, description="Whether text is bold")
    is_italic: bool = Field(default=False, description="Whether text is italic")
    
    # Content analysis
    word_count: int = Field(default=0, ge=0, description="Number of words in text")
    language_detected: Optional[str] = Field(default=None, description="Detected language code")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.word_count and self.original_text:
            self.word_count = len(self.original_text.split())

class ImageBlock(BaseBlock):
    """Represents image content blocks with file system links"""
    
    # Image file properties
    image_path: str = Field(..., description="Path to saved image file")
    image_format: str = Field(default="png", description="Image format (png, jpg, etc.)")
    image_size: Optional[Tuple[int, int]] = Field(default=None, description="Image dimensions in pixels")
    image_hash: Optional[str] = Field(default=None, description="SHA256 hash of image data")
    
    # Visual properties
    alt_text: str = Field(default="", description="Alternative text description")
    caption_text: str = Field(default="", description="Associated caption text")
    
    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        if not v:
            raise ValueError("image_path cannot be empty")
        return str(Path(v).as_posix())  # Normalize path separators
    
    def image_exists(self) -> bool:
        """Check if image file exists on filesystem"""
        return Path(self.image_path).exists()

class TableBlock(BaseBlock):
    """Represents table content with structured data"""
    
    # Table structure
    rows: List[List[str]] = Field(default_factory=list, description="Table data as rows of cells")
    headers: Optional[List[str]] = Field(default=None, description="Table header row")
    
    # Table properties
    num_rows: int = Field(default=0, ge=0, description="Number of data rows")
    num_cols: int = Field(default=0, ge=0, description="Number of columns")
    has_header: bool = Field(default=False, description="Whether table has header row")
    
    # Formatting
    table_style: str = Field(default="grid", description="Table visual style")
    markdown_content: str = Field(default="", description="Markdown representation of table")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.rows:
            self.num_rows = len(self.rows)
            self.num_cols = max(len(row) for row in self.rows) if self.rows else 0
            self.has_header = bool(self.headers)
    
    @field_validator('rows')
    @classmethod
    def validate_rows(cls, v: List[List[str]]) -> List[List[str]]:
        if not v:
            return []
        
        # Ensure all rows have same number of columns
        max_cols = max(len(row) for row in v) if v else 0
        normalized_rows = []
        
        for row in v:
            normalized_row = row + [''] * (max_cols - len(row))
            normalized_rows.append(normalized_row[:max_cols])
        
        return normalized_rows

class TOCEntry(BaseModel):
    """Represents a Table of Contents entry with navigation properties"""
    
    entry_id: str = Field(..., description="Unique identifier for TOC entry")
    title: str = Field(..., description="TOC entry title")
    level: int = Field(..., ge=1, le=6, description="Hierarchical level (1-6)")
    page_number: int = Field(..., ge=1, description="Target page number")
    
    # Translation support
    original_title: str = Field(default="", description="Original title before translation")
    translated_title: str = Field(default="", description="Translated title")
    
    # Navigation properties
    anchor_id: str = Field(default="", description="Internal anchor/bookmark ID")
    parent_entry_id: Optional[str] = Field(default=None, description="Parent TOC entry ID")
    children_ids: List[str] = Field(default_factory=list, description="Child TOC entry IDs")
    
    def get_display_title(self, prefer_translation: bool = True) -> str:
        """Get appropriate title for display"""
        if prefer_translation and self.translated_title:
            return self.translated_title
        return self.title

class PageModel(BaseModel):
    """Represents a single page with all its content blocks"""
    
    page_number: int = Field(..., ge=1, description="1-based page number")
    dimensions: PageDimensions = Field(..., description="Page dimensions (width, height)")
    
    # Content blocks organized by type for efficient access
    text_blocks: List[TextBlock] = Field(default_factory=list, description="Text content blocks")
    image_blocks: List[ImageBlock] = Field(default_factory=list, description="Image content blocks") 
    table_blocks: List[TableBlock] = Field(default_factory=list, description="Table content blocks")
    
    # Page metadata
    page_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional page properties")
    rotation: int = Field(default=0, description="Page rotation in degrees")
    
    # Processing information
    extraction_time: Optional[float] = Field(default=None, description="Time taken to extract content")
    processing_strategy: str = Field(default="", description="Strategy used to process this page")
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v: PageDimensions) -> PageDimensions:
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("Page dimensions must be positive")
        return v
    
    def get_all_blocks(self) -> List[BaseBlock]:
        """Get all blocks on this page in reading order"""
        all_blocks = []
        all_blocks.extend(self.text_blocks)
        all_blocks.extend(self.image_blocks)
        all_blocks.extend(self.table_blocks)
        
        # Sort by reading order: top-to-bottom, left-to-right
        return sorted(all_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    
    def get_blocks_by_type(self, block_type: BlockType) -> List[BaseBlock]:
        """Get all blocks of specified type"""
        return [block for block in self.get_all_blocks() if block.block_type == block_type]
    
    def add_block(self, block: BaseBlock) -> None:
        """Add a block to the appropriate collection"""
        block.page_number = self.page_number
        
        if isinstance(block, TextBlock):
            self.text_blocks.append(block)
        elif isinstance(block, ImageBlock):
            self.image_blocks.append(block)
        elif isinstance(block, TableBlock):
            self.table_blocks.append(block)
        else:
            logger.warning(f"Unknown block type for addition: {type(block)}")

class DocumentModel(BaseModel):
    """Digital Twin representation of the complete document"""
    
    # Document identification
    title: str = Field(default="", description="Document title")
    filename: str = Field(..., description="Source filename")
    document_id: str = Field(default="", description="Unique document identifier")
    
    # Content structure
    pages: List[PageModel] = Field(default_factory=list, description="Document pages")
    toc_entries: List[TOCEntry] = Field(default_factory=list, description="Table of contents entries")
    
    # Document metadata
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document properties")
    creation_date: Optional[str] = Field(default=None, description="Document creation date")
    
    # Translation properties
    source_language: str = Field(default="", description="Detected/specified source language")
    target_language: str = Field(default="", description="Target translation language")
    translation_status: str = Field(default="pending", description="Translation progress status")
    
    # Processing metadata
    extraction_method: str = Field(default="pymupdf_yolo", description="Content extraction method used")
    processing_time: Optional[float] = Field(default=None, description="Total processing time")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.document_id:
            self.document_id = self.generate_document_id()
        if self.pages:
            self.total_pages = len(self.pages)
    
    def generate_document_id(self) -> str:
        """Generate unique document identifier"""
        content = f"{self.filename}:{self.title}:{len(self.pages)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_page(self, page: PageModel) -> None:
        """Add a page to the document"""
        self.pages.append(page)
        self.total_pages = len(self.pages)
    
    def get_page(self, page_number: int) -> Optional[PageModel]:
        """Get page by number (1-based)"""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def get_all_text_blocks(self) -> List[TextBlock]:
        """Get all text blocks from all pages"""
        text_blocks = []
        for page in self.pages:
            text_blocks.extend(page.text_blocks)
        return text_blocks
    
    def get_all_image_blocks(self) -> List[ImageBlock]:
        """Get all image blocks from all pages"""
        image_blocks = []
        for page in self.pages:
            image_blocks.extend(page.image_blocks)
        return image_blocks
    
    def get_translatable_blocks(self) -> List[TextBlock]:
        """Get all text blocks that should be translated"""
        translatable_roles = {
            StructuralRole.CONTENT,
            StructuralRole.ILLUSTRATION  # For captions
        }
        
        translatable_blocks = []
        for text_block in self.get_all_text_blocks():
            if text_block.structural_role in translatable_roles:
                translatable_blocks.append(text_block)
        
        return translatable_blocks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        all_text_blocks = self.get_all_text_blocks()
        all_image_blocks = self.get_all_image_blocks()
        
        return {
            'total_pages': self.total_pages,
            'total_text_blocks': len(all_text_blocks),
            'total_image_blocks': len(all_image_blocks),
            'total_tables': sum(len(page.table_blocks) for page in self.pages),
            'total_toc_entries': len(self.toc_entries),
            'total_words': sum(block.word_count for block in all_text_blocks),
            'translated_blocks': len([b for b in all_text_blocks if b.has_translation()]),
            'extraction_method': self.extraction_method,
            'translation_status': self.translation_status,
            'source_language': self.source_language,
            'target_language': self.target_language
        }
    
    def validate_structure(self) -> List[str]:
        """Validate document structure and return list of issues"""
        issues = []
        
        # Check page numbering
        expected_pages = set(range(1, self.total_pages + 1))
        actual_pages = {page.page_number for page in self.pages}
        
        if expected_pages != actual_pages:
            issues.append(f"Page numbering mismatch: expected {expected_pages}, got {actual_pages}")
        
        # Check block IDs are unique
        all_block_ids = []
        for page in self.pages:
            for block in page.get_all_blocks():
                all_block_ids.append(block.block_id)
        
        duplicate_ids = [id for id in all_block_ids if all_block_ids.count(id) > 1]
        if duplicate_ids:
            issues.append(f"Duplicate block IDs found: {set(duplicate_ids)}")
        
        # Check image paths exist
        for image_block in self.get_all_image_blocks():
            if not image_block.image_exists():
                issues.append(f"Missing image file: {image_block.image_path}")
        
        # Check TOC entry references
        all_page_numbers = {page.page_number for page in self.pages}
        for toc_entry in self.toc_entries:
            if toc_entry.page_number not in all_page_numbers:
                issues.append(f"TOC entry references non-existent page: {toc_entry.page_number}")
        
        return issues
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Export document to dictionary format for serialization"""
        return {
            'metadata': {
                'title': self.title,
                'filename': self.filename,
                'document_id': self.document_id,
                'total_pages': self.total_pages,
                'source_language': self.source_language,
                'target_language': self.target_language,
                'extraction_method': self.extraction_method,
                'statistics': self.get_statistics()
            },
            'pages': [page.model_dump() for page in self.pages],
            'toc_entries': [entry.model_dump() for entry in self.toc_entries],
            'validation_issues': self.validate_structure()
        }

# Factory functions for creating common block types
def create_text_block(
    block_id: str,
    text: str,
    bbox: BoundingBox,
    page_number: int,
    block_type: BlockType = BlockType.PARAGRAPH,
    **kwargs
) -> TextBlock:
    """Factory function for creating text blocks"""
    return TextBlock(
        block_id=block_id,
        block_type=block_type,
        bbox=bbox,
        page_number=page_number,
        original_text=text,
        **kwargs
    )

def create_image_block(
    block_id: str,
    image_path: str,
    bbox: BoundingBox,
    page_number: int,
    **kwargs
) -> ImageBlock:
    """Factory function for creating image blocks"""
    return ImageBlock(
        block_id=block_id,
        block_type=BlockType.IMAGE,
        bbox=bbox,
        page_number=page_number,
        image_path=image_path,
        **kwargs
    )

def create_table_block(
    block_id: str,
    rows: List[List[str]],
    bbox: BoundingBox,
    page_number: int,
    headers: Optional[List[str]] = None,
    **kwargs
) -> TableBlock:
    """Factory function for creating table blocks"""
    return TableBlock(
        block_id=block_id,
        block_type=BlockType.TABLE,
        bbox=bbox,
        page_number=page_number,
        rows=rows,
        headers=headers,
        **kwargs
    )

# Migration utilities for existing code
def migrate_from_models_py(page_model_data: Dict[str, Any]) -> PageModel:
    """Migrate data from the old models.py format to Digital Twin format"""
    # Implementation would convert from existing PageModel/ElementModel format
    # This bridges the gap during transition
    pass

def migrate_from_structured_document_model(structured_doc_data: Dict[str, Any]) -> DocumentModel:
    """Migrate data from structured_document_model.py format to Digital Twin format"""
    # Implementation would convert from existing StructuredDocument format
    # This bridges the gap during transition
    pass

logger.info("Digital Twin Document Model loaded - unified data schema ready") 