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
    BIBLIOGRAPHY = "bibliography"  # Add bibliography block type
    REFERENCE = "reference"  # Individual reference entries
    
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
    BIBLIOGRAPHY = "bibliography"  # Bibliography and references (preserve as-is)
    FORMATTING = "formatting"  # Spacing, breaks, styling

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
    """Represents a Table of Contents entry with navigation properties and two-way mapping support"""
    
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
    
    # Two-way mapping properties for comprehensive reconstruction
    mapped_heading_blocks: List[str] = Field(default_factory=list, description="IDs of heading blocks that match this TOC entry")
    content_fingerprint: str = Field(default="", description="Content fingerprint for matching across translations")
    original_page_in_document: int = Field(default=0, description="Actual page number found in document content")
    translated_page_in_document: int = Field(default=0, description="Actual page number in translated document")
    
    # Context for intelligent translation
    hierarchical_context: str = Field(default="", description="Parent section context for better translation")
    section_type: str = Field(default="", description="Detected section type (chapter, section, appendix, etc.)")
    content_preview: str = Field(default="", description="Preview of section content for validation")
    translation_context: str = Field(default="", description="Rich translation context for intelligent processing")
    
    # Reconstruction metadata
    heading_style_detected: Dict[str, Any] = Field(default_factory=dict, description="Detected heading formatting")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in mapping accuracy")
    processing_notes: List[str] = Field(default_factory=list, description="Processing and mapping notes")
    
    def get_display_title(self, prefer_translation: bool = True) -> str:
        """Get appropriate title for display"""
        if prefer_translation and self.translated_title:
            return self.translated_title
        return self.title
    
    def get_hierarchical_path(self) -> str:
        """Get full hierarchical path for context (filled during hierarchy building)"""
        return self.hierarchical_context
    
    def add_mapped_heading(self, heading_block_id: str, confidence: float = 1.0) -> None:
        """Add a mapped heading block with confidence tracking"""
        if heading_block_id not in self.mapped_heading_blocks:
            self.mapped_heading_blocks.append(heading_block_id)
            if confidence < self.confidence_score:
                self.confidence_score = confidence
            self.processing_notes.append(f"Mapped to heading block {heading_block_id} (confidence: {confidence:.2f})")
    
    def has_reliable_mapping(self) -> bool:
        """Check if this TOC entry has reliable content mapping"""
        return (self.mapped_heading_blocks and 
                self.confidence_score >= 0.7 and 
                self.content_fingerprint and
                self.original_page_in_document > 0)
    
    def generate_content_fingerprint(self, content_text: str) -> None:
        """Generate a content fingerprint for cross-language matching"""
        import hashlib
        import re
        
        # Normalize content for fingerprinting
        normalized = re.sub(r'[^\w\s]', '', content_text.lower()).strip()
        words = normalized.split()[:10]  # First 10 significant words
        fingerprint_text = ' '.join(words)
        
        self.content_fingerprint = hashlib.md5(fingerprint_text.encode()).hexdigest()[:16]
    
    def update_page_location(self, original_page: int, translated_page: int = None) -> None:
        """Update page location information"""
        self.original_page_in_document = original_page
        if translated_page is not None:
            self.translated_page_in_document = translated_page
        else:
            self.translated_page_in_document = original_page

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
        """Get all blocks on this page in extraction (sequential) order"""
        all_blocks = []
        all_blocks.extend(self.text_blocks)
        all_blocks.extend(self.image_blocks)
        all_blocks.extend(self.table_blocks)
        return all_blocks
    
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
        """
        Comprehensive document structure validation with enhanced quality checks.
        
        This provides thorough validation to ensure document integrity and quality.
        """
        issues = []
        
        # Basic structural validation
        issues.extend(self._validate_basic_structure())
        
        # Content quality validation
        issues.extend(self._validate_content_quality())
        
        # Relationship validation
        issues.extend(self._validate_relationships())
        
        # File system validation
        issues.extend(self._validate_file_system())
        
        # Processing integrity validation
        issues.extend(self._validate_processing_integrity())
        
        return issues
    
    def _validate_basic_structure(self) -> List[str]:
        """Validate basic document structure"""
        issues = []
        
        # Check page numbering
        expected_pages = set(range(1, self.total_pages + 1))
        actual_pages = {page.page_number for page in self.pages}
        
        if expected_pages != actual_pages:
            missing_pages = expected_pages - actual_pages
            extra_pages = actual_pages - expected_pages
            
            if missing_pages:
                issues.append(f"Missing pages: {sorted(missing_pages)}")
            if extra_pages:
                issues.append(f"Unexpected pages: {sorted(extra_pages)}")
        
        # Check block IDs are unique across entire document
        all_block_ids = []
        for page in self.pages:
            for block in page.get_all_blocks():
                all_block_ids.append(block.block_id)
        
        duplicate_ids = [id for id in all_block_ids if all_block_ids.count(id) > 1]
        if duplicate_ids:
            issues.append(f"Duplicate block IDs found: {set(duplicate_ids)}")
        
        # Check page dimensions are reasonable
        for page in self.pages:
            width, height = page.dimensions
            if width <= 0 or height <= 0:
                issues.append(f"Invalid page dimensions on page {page.page_number}: {width}x{height}")
            elif width > 10000 or height > 10000:
                issues.append(f"Unusually large page dimensions on page {page.page_number}: {width}x{height}")
        
        return issues
    
    def _validate_content_quality(self) -> List[str]:
        """Validate content quality and completeness"""
        issues = []
        
        # Check for empty pages
        for page in self.pages:
            all_blocks = page.get_all_blocks()
            if not all_blocks:
                issues.append(f"Page {page.page_number} has no content blocks")
            else:
                # Check for pages with only metadata blocks
                content_blocks = [b for b in all_blocks if b.structural_role == StructuralRole.CONTENT]
                if not content_blocks:
                    issues.append(f"Page {page.page_number} has no content blocks (only metadata)")
        
        # Check text block quality
        for text_block in self.get_all_text_blocks():
            # Check for empty text blocks
            if not text_block.original_text.strip():
                issues.append(f"Empty text block: {text_block.block_id}")
            
            # Check for extremely long text blocks (possible extraction error)
            if len(text_block.original_text) > 10000:
                issues.append(f"Unusually long text block: {text_block.block_id} ({len(text_block.original_text)} chars)")
            
            # Check bounding box validity
            x0, y0, x1, y1 = text_block.bbox
            if x0 >= x1 or y0 >= y1:
                issues.append(f"Invalid bounding box for text block {text_block.block_id}: {text_block.bbox}")
            
            # Check font size reasonableness
            if text_block.font_size > 0 and (text_block.font_size < 4 or text_block.font_size > 72):
                issues.append(f"Unusual font size for text block {text_block.block_id}: {text_block.font_size}pt")
        
        # Check image block quality
        for image_block in self.get_all_image_blocks():
            # Check bounding box validity
            x0, y0, x1, y1 = image_block.bbox
            if x0 >= x1 or y0 >= y1:
                issues.append(f"Invalid bounding box for image block {image_block.block_id}: {image_block.bbox}")
            
            # Check for very small images (possible extraction artifacts)
            width = x1 - x0
            height = y1 - y0
            if width < 10 or height < 10:
                issues.append(f"Very small image block {image_block.block_id}: {width}x{height}")
        
        # Check table block quality
        for page in self.pages:
            for table_block in page.table_blocks:
                # Check table structure
                if not table_block.rows:
                    issues.append(f"Empty table block: {table_block.block_id}")
                else:
                    # Check row consistency
                    if table_block.rows:
                        expected_cols = len(table_block.rows[0])
                        for i, row in enumerate(table_block.rows):
                            if len(row) != expected_cols:
                                issues.append(f"Inconsistent row length in table {table_block.block_id}, row {i}")
        
        return issues
    
    def _validate_relationships(self) -> List[str]:
        """Validate relationships between document elements"""
        issues = []
        
        # Check TOC entry references
        all_page_numbers = {page.page_number for page in self.pages}
        for toc_entry in self.toc_entries:
            if toc_entry.page_number not in all_page_numbers:
                issues.append(f"TOC entry '{toc_entry.title}' references non-existent page: {toc_entry.page_number}")
        
        # Check TOC hierarchy consistency
        for toc_entry in self.toc_entries:
            if toc_entry.parent_entry_id:
                parent_exists = any(t.entry_id == toc_entry.parent_entry_id for t in self.toc_entries)
                if not parent_exists:
                    issues.append(f"TOC entry '{toc_entry.title}' references non-existent parent: {toc_entry.parent_entry_id}")
        
        # Check for orphaned content
        orphaned_blocks = []
        for page in self.pages:
            for block in page.get_all_blocks():
                if (block.structural_role == StructuralRole.CONTENT and 
                    block.block_type not in [BlockType.HEADER, BlockType.FOOTER]):
                    # Check if this content block is near a heading
                    has_nearby_heading = False
                    for other_block in page.get_all_blocks():
                        if (other_block.block_type in [BlockType.HEADING, BlockType.TITLE] and
                            abs(other_block.bbox[1] - block.bbox[1]) < 100):  # Within 100 units
                            has_nearby_heading = True
                            break
                    
                    if not has_nearby_heading and len(block.get_display_text()) > 100:
                        orphaned_blocks.append(block.block_id)
        
        if orphaned_blocks:
            issues.append(f"Orphaned content blocks (no nearby headings): {orphaned_blocks[:5]}{'...' if len(orphaned_blocks) > 5 else ''}")
        
        return issues
    
    def _validate_file_system(self) -> List[str]:
        """Validate file system dependencies"""
        issues = []
        
        # Check image file existence and accessibility
        for image_block in self.get_all_image_blocks():
            if not image_block.image_exists():
                issues.append(f"Missing image file: {image_block.image_path}")
            else:
                # Check file size
                try:
                    import os
                    file_size = os.path.getsize(image_block.image_path)
                    if file_size == 0:
                        issues.append(f"Empty image file: {image_block.image_path}")
                    elif file_size < 100:
                        issues.append(f"Suspiciously small image file: {image_block.image_path} ({file_size} bytes)")
                except Exception as e:
                    issues.append(f"Cannot access image file {image_block.image_path}: {e}")
        
        return issues
    
    def _validate_processing_integrity(self) -> List[str]:
        """Validate processing integrity and completeness"""
        issues = []
        
        # Check for processing errors in metadata
        pages_with_errors = []
        for page in self.pages:
            if page.page_metadata.get('processing_failed'):
                pages_with_errors.append(page.page_number)
            
            # Check for missing extraction times (indicates incomplete processing)
            if page.extraction_time is None and not page.page_metadata.get('processing_failed'):
                issues.append(f"Page {page.page_number} missing extraction time (incomplete processing)")
        
        if pages_with_errors:
            issues.append(f"Pages with processing errors: {pages_with_errors}")
        
        # Check for blocks with processing notes indicating problems
        problematic_blocks = []
        for page in self.pages:
            for block in page.get_all_blocks():
                if any('error' in note.lower() or 'failed' in note.lower() for note in block.processing_notes):
                    problematic_blocks.append(block.block_id)
        
        if problematic_blocks:
            issues.append(f"Blocks with processing errors: {problematic_blocks[:5]}{'...' if len(problematic_blocks) > 5 else ''}")
        
        # Check document-level processing completeness
        if not self.processing_time:
            issues.append("Document missing overall processing time")
        
        if not self.extraction_method:
            issues.append("Document missing extraction method information")
        
        return issues
    
    def analyze_document_flow(self) -> Dict[str, Any]:
        """
        Analyze document flow and structure for better preservation during translation.
        
        OPTIMIZED: Provides enhanced document flow analysis for better format continuity.
        This helps maintain proper document structure during translation and reconstruction.
        """
        flow_analysis = {
            'reading_order': [],
            'section_hierarchy': {},
            'cross_references': [],
            'formatting_patterns': {},
            'content_flow_score': 0.0,
            'structural_integrity': 'good'
        }
        
        try:
            # Analyze reading order across pages
            flow_analysis['reading_order'] = self._analyze_reading_order()
            
            # Analyze section hierarchy
            flow_analysis['section_hierarchy'] = self._analyze_section_hierarchy()
            
            # Detect cross-references and links
            flow_analysis['cross_references'] = self._detect_cross_references()
            
            # Analyze formatting patterns
            flow_analysis['formatting_patterns'] = self._analyze_formatting_patterns()
            
            # Calculate content flow score
            flow_analysis['content_flow_score'] = self._calculate_content_flow_score()
            
            # Assess structural integrity
            flow_analysis['structural_integrity'] = self._assess_structural_integrity()
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Document flow analysis failed: {e}")
            flow_analysis['error'] = str(e)
            flow_analysis['structural_integrity'] = 'error'
            return flow_analysis
    
    def _analyze_reading_order(self) -> List[Dict[str, Any]]:
        """Analyze logical reading order across pages"""
        reading_order = []
        
        for page in self.pages:
            page_blocks = []
            
            # Sort blocks by position (top to bottom, left to right)
            all_blocks = page.get_all_blocks()
            sorted_blocks = sorted(all_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by y, then x
            
            for i, block in enumerate(sorted_blocks):
                page_blocks.append({
                    'block_id': block.block_id,
                    'block_type': block.block_type.value,
                    'structural_role': block.structural_role.value,
                    'position': i,
                    'bbox': block.bbox,
                    'content_preview': block.get_display_text()[:100]
                })
            
            reading_order.append({
                'page_number': page.page_number,
                'blocks': page_blocks,
                'flow_continuity': self._assess_page_flow_continuity(page_blocks)
            })
        
        return reading_order
    
    def _analyze_section_hierarchy(self) -> Dict[str, Any]:
        """Analyze document section hierarchy and structure"""
        hierarchy = {
            'sections': [],
            'max_depth': 0,
            'orphaned_content': [],
            'section_balance': 'balanced'
        }
        
        current_section = None
        current_hierarchy = []
        
        for page in self.pages:
            for block in page.text_blocks:
                if block.block_type in [BlockType.HEADING, BlockType.TITLE]:
                    # Determine hierarchy level
                    level = block.heading_level or self._infer_heading_level(block)
                    
                    section_info = {
                        'block_id': block.block_id,
                        'title': block.get_display_text(),
                        'level': level,
                        'page_number': page.page_number,
                        'content_blocks': []
                    }
                    
                    # Update hierarchy tracking
                    if level > len(current_hierarchy):
                        current_hierarchy.append(section_info)
                    else:
                        current_hierarchy = current_hierarchy[:level-1] + [section_info]
                    
                    hierarchy['sections'].append(section_info)
                    hierarchy['max_depth'] = max(hierarchy['max_depth'], level)
                    current_section = section_info
                
                elif current_section and block.structural_role == StructuralRole.CONTENT:
                    # Add content to current section
                    current_section['content_blocks'].append(block.block_id)
                
                elif current_section is None and block.structural_role == StructuralRole.CONTENT:
                    # Orphaned content (no section header)
                    hierarchy['orphaned_content'].append(block.block_id)
        
        # Assess section balance
        if hierarchy['orphaned_content']:
            hierarchy['section_balance'] = 'unbalanced'
        elif hierarchy['max_depth'] > 4:
            hierarchy['section_balance'] = 'deep'
        
        return hierarchy
    
    def _detect_cross_references(self) -> List[Dict[str, Any]]:
        """Detect cross-references and internal links"""
        cross_refs = []
        
        # Common cross-reference patterns
        ref_patterns = [
            r'[Ss]ee\s+(?:page\s+)?(\d+)',
            r'[Ff]igure\s+(\d+)',
            r'[Tt]able\s+(\d+)',
            r'[Ss]ection\s+(\d+(?:\.\d+)*)',
            r'[Cc]hapter\s+(\d+)',
            r'[Aa]ppendix\s+([A-Z])',
        ]
        
        for page in self.pages:
            for block in page.text_blocks:
                text = block.get_display_text()
                
                for pattern in ref_patterns:
                    import re
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        cross_refs.append({
                            'source_block_id': block.block_id,
                            'source_page': page.page_number,
                            'reference_text': match.group(0),
                            'reference_target': match.group(1),
                            'reference_type': pattern.split('\\')[0].lower(),
                            'position_in_text': match.start()
                        })
        
        return cross_refs
    
    def _analyze_formatting_patterns(self) -> Dict[str, Any]:
        """Analyze formatting patterns for consistency"""
        patterns = {
            'font_families': {},
            'font_sizes': {},
            'heading_styles': {},
            'paragraph_styles': {},
            'consistency_score': 0.0
        }
        
        # Collect formatting data
        for page in self.pages:
            for block in page.text_blocks:
                # Font family patterns
                if block.font_family:
                    patterns['font_families'][block.font_family] = patterns['font_families'].get(block.font_family, 0) + 1
                
                # Font size patterns
                if block.font_size:
                    size_key = f"{block.font_size:.1f}"
                    patterns['font_sizes'][size_key] = patterns['font_sizes'].get(size_key, 0) + 1
                
                # Heading style patterns
                if block.block_type in [BlockType.HEADING, BlockType.TITLE]:
                    style_key = f"{block.font_family}_{block.font_size}_{block.font_weight}"
                    patterns['heading_styles'][style_key] = patterns['heading_styles'].get(style_key, 0) + 1
                
                # Paragraph style patterns
                elif block.block_type == BlockType.PARAGRAPH:
                    style_key = f"{block.font_family}_{block.font_size}"
                    patterns['paragraph_styles'][style_key] = patterns['paragraph_styles'].get(style_key, 0) + 1
        
        # Calculate consistency score
        patterns['consistency_score'] = self._calculate_formatting_consistency(patterns)
        
        return patterns
    
    def _calculate_content_flow_score(self) -> float:
        """Calculate overall content flow quality score"""
        score = 1.0
        
        # Penalize for missing pages
        if len(self.pages) != self.total_pages:
            score *= 0.8
        
        # Penalize for orphaned content
        hierarchy = self._analyze_section_hierarchy()
        if hierarchy['orphaned_content']:
            score *= 0.9
        
        # Reward for good TOC structure
        if self.toc_entries:
            toc_coverage = len(self.toc_entries) / max(1, self.total_pages)
            if toc_coverage > 0.1:  # At least 1 TOC entry per 10 pages
                score *= 1.1
        
        # Penalize for validation issues
        issues = self.validate_structure()
        if issues:
            score *= max(0.5, 1.0 - len(issues) * 0.1)
        
        return min(1.0, score)
    
    def _assess_structural_integrity(self) -> str:
        """Assess overall structural integrity"""
        issues = self.validate_structure()
        flow_score = self._calculate_content_flow_score()
        
        if issues:
            if len(issues) > 5:
                return 'poor'
            elif len(issues) > 2:
                return 'fair'
            else:
                return 'good'
        
        if flow_score < 0.7:
            return 'fair'
        elif flow_score < 0.9:
            return 'good'
        else:
            return 'excellent'
    
    def _assess_page_flow_continuity(self, page_blocks: List[Dict[str, Any]]) -> str:
        """Assess flow continuity within a page"""
        if not page_blocks:
            return 'empty'
        
        # Check for logical block ordering
        content_blocks = [b for b in page_blocks if b['structural_role'] == 'content']
        
        if len(content_blocks) < 2:
            return 'minimal'
        
        # Check vertical spacing consistency
        y_positions = [b['bbox'][1] for b in content_blocks]
        if len(set(y_positions)) == len(y_positions):  # All blocks at different heights
            return 'good'
        else:
            return 'complex'  # Overlapping or side-by-side content
    
    def _infer_heading_level(self, block: TextBlock) -> int:
        """Infer heading level from font size and formatting"""
        if block.font_size >= 18:
            return 1
        elif block.font_size >= 16:
            return 2
        elif block.font_size >= 14:
            return 3
        elif block.is_bold:
            return 4
        else:
            return 5
    
    def _calculate_formatting_consistency(self, patterns: Dict[str, Any]) -> float:
        """Calculate formatting consistency score"""
        score = 1.0
        
        # Check font family consistency
        if patterns['font_families']:
            dominant_font = max(patterns['font_families'].values())
            total_blocks = sum(patterns['font_families'].values())
            font_consistency = dominant_font / total_blocks
            score *= font_consistency
        
        # Check font size consistency
        if patterns['font_sizes']:
            size_variety = len(patterns['font_sizes'])
            if size_variety > 8:  # Too many different sizes
                score *= 0.8
            elif size_variety < 3:  # Too few sizes (likely poor formatting)
                score *= 0.9
        
        return min(1.0, score)
    
    def optimize_document_flow(self) -> Dict[str, Any]:
        """
        Optimize document flow and structure for better translation and reconstruction.
        
        Returns optimization suggestions and applied fixes.
        """
        optimization_report = {
            'analysis': self.analyze_document_flow(),
            'optimizations_applied': [],
            'suggestions': [],
            'improved_flow_score': 0.0
        }
        
        try:
            # Apply automatic optimizations
            self._optimize_reading_order()
            optimization_report['optimizations_applied'].append('reading_order_optimization')
            
            self._optimize_section_hierarchy()
            optimization_report['optimizations_applied'].append('section_hierarchy_optimization')
            
            self._optimize_cross_references()
            optimization_report['optimizations_applied'].append('cross_reference_optimization')
            
            # Generate suggestions for manual review
            optimization_report['suggestions'] = self._generate_optimization_suggestions()
            
            # Calculate improved flow score
            optimization_report['improved_flow_score'] = self._calculate_content_flow_score()
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"Document flow optimization failed: {e}")
            optimization_report['error'] = str(e)
            return optimization_report
    
    def _optimize_reading_order(self) -> None:
        """Optimize reading order within pages"""
        for page in self.pages:
            # Sort blocks by logical reading order
            all_blocks = page.get_all_blocks()
            
            # Separate by structural role
            navigation_blocks = [b for b in all_blocks if b.structural_role == StructuralRole.NAVIGATION]
            content_blocks = [b for b in all_blocks if b.structural_role == StructuralRole.CONTENT]
            illustration_blocks = [b for b in all_blocks if b.structural_role == StructuralRole.ILLUSTRATION]
            
            # Sort each group by position
            navigation_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            content_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            illustration_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            
            # Assign z-order for proper layering
            z_order = 0
            for block_group in [navigation_blocks, content_blocks, illustration_blocks]:
                for block in block_group:
                    block.z_order = z_order
                    z_order += 1
    
    def _optimize_section_hierarchy(self) -> None:
        """Optimize section hierarchy and heading levels"""
        hierarchy = self._analyze_section_hierarchy()
        
        # Normalize heading levels
        for section in hierarchy['sections']:
            block_id = section['block_id']
            
            # Find the actual block and update heading level
            for page in self.pages:
                for block in page.text_blocks:
                    if block.block_id == block_id:
                        block.heading_level = section['level']
                        break
    
    def _optimize_cross_references(self) -> None:
        """Optimize cross-references for better linking"""
        cross_refs = self._detect_cross_references()
        
        # Add cross-reference metadata to blocks
        for ref in cross_refs:
            for page in self.pages:
                for block in page.text_blocks:
                    if block.block_id == ref['source_block_id']:
                        if 'cross_references' not in block.processing_notes:
                            block.processing_notes.append(f"Contains cross-reference: {ref['reference_text']}")
                        break
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate suggestions for manual optimization"""
        suggestions = []
        
        # Analyze current state
        flow_analysis = self.analyze_document_flow()
        
        if flow_analysis['structural_integrity'] == 'poor':
            suggestions.append("Consider manual review of document structure - multiple structural issues detected")
        
        if flow_analysis['content_flow_score'] < 0.8:
            suggestions.append("Document flow could be improved - consider reorganizing content blocks")
        
        if len(flow_analysis['cross_references']) == 0:
            suggestions.append("No cross-references detected - consider adding navigation aids")
        
        hierarchy = flow_analysis.get('section_hierarchy', {})
        if hierarchy.get('orphaned_content'):
            suggestions.append(f"Found {len(hierarchy['orphaned_content'])} orphaned content blocks - consider adding section headers")
        
        return suggestions
    
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
    try:
        # Extract basic page information
        page_number = page_model_data.get('page_number', 1)
        dimensions = page_model_data.get('dimensions', [0.0, 0.0])
        
        # Ensure dimensions is a tuple
        if isinstance(dimensions, list) and len(dimensions) == 2:
            dimensions = tuple(dimensions)
        else:
            dimensions = (0.0, 0.0)
        
        # Create Digital Twin page model
        page_model = PageModel(
            page_number=page_number,
            dimensions=dimensions
        )
        
        # Convert elements to Digital Twin blocks
        elements = page_model_data.get('elements', [])
        block_id_counter = 0
        
        for element in elements:
            block_id_counter += 1
            element_type = element.get('type', 'text')
            bbox = element.get('bbox', (0, 0, 0, 0))
            confidence = element.get('confidence', 1.0)
            
            # Generate unique block ID
            block_id = f"migrated_{page_number}_{block_id_counter}"
            
            if element_type == 'text':
                # Create text block
                content = element.get('content', '')
                if isinstance(content, str):
                    text_block = create_text_block(
                        block_id=block_id,
                        text=content,
                        bbox=bbox,
                        page_number=page_number,
                        block_type=BlockType.PARAGRAPH,
                        confidence=confidence,
                        extraction_method="migrated_from_models_py"
                    )
                    page_model.add_block(text_block)
            
            elif element_type == 'image':
                # Create image block (need to handle path properly)
                image_path = element.get('content', f'migrated_image_{block_id}.png')
                if isinstance(image_path, bytes):
                    # If content is bytes, we need to save it as a file
                    image_path = f'migrated_image_{block_id}.png'
                
                image_block = create_image_block(
                    block_id=block_id,
                    image_path=str(image_path),
                    bbox=bbox,
                    page_number=page_number,
                    confidence=confidence,
                    extraction_method="migrated_from_models_py"
                )
                page_model.add_block(image_block)
            
            elif element_type == 'table':
                # Create table block
                content = element.get('content', [])
                if isinstance(content, list):
                    table_block = create_table_block(
                        block_id=block_id,
                        rows=content,
                        bbox=bbox,
                        page_number=page_number,
                        confidence=confidence,
                        extraction_method="migrated_from_models_py"
                    )
                    page_model.add_block(table_block)
        
        logger.info(f"Successfully migrated page {page_number} with {len(elements)} elements to Digital Twin format")
        return page_model
        
    except Exception as e:
        logger.error(f"Failed to migrate page data from models.py format: {e}")
        # Return minimal page model
        return PageModel(
            page_number=page_model_data.get('page_number', 1),
            dimensions=(0.0, 0.0),
            page_metadata={'migration_error': str(e)}
        )

def migrate_from_structured_document_model(structured_doc_data: Dict[str, Any]) -> DocumentModel:
    """Migrate data from structured_document_model.py format to Digital Twin format"""
    try:
        # Extract document-level information
        title = structured_doc_data.get('title', 'Migrated Document')
        content_blocks = structured_doc_data.get('content_blocks', [])
        metadata = structured_doc_data.get('metadata', {})
        source_filepath = structured_doc_data.get('source_filepath', '')
        
        # Create Digital Twin document model
        digital_twin_doc = DocumentModel(
            title=title,
            filename=source_filepath.split('/')[-1] if source_filepath else 'migrated_document',
            document_metadata=metadata,
            extraction_method="migrated_from_structured_document_model"
        )
        
        # Group content blocks by page
        pages_dict = {}
        block_id_counter = 0
        
        for block in content_blocks:
            block_id_counter += 1
            page_num = getattr(block, 'page_num', 1)
            
            # Initialize page if not exists
            if page_num not in pages_dict:
                pages_dict[page_num] = PageModel(
                    page_number=page_num,
                    dimensions=(0.0, 0.0)  # Will be updated if we have bbox info
                )
            
            # Convert content block to Digital Twin block
            block_type_str = str(getattr(block, 'block_type', 'paragraph')).lower()
            original_text = getattr(block, 'original_text', '')
            bbox = getattr(block, 'bbox', (0, 0, 0, 0))
            block_id = f"migrated_struct_{page_num}_{block_id_counter}"
            
            # Map structured document types to Digital Twin types
            if 'heading' in block_type_str:
                dt_block_type = BlockType.HEADING
                structural_role = StructuralRole.NAVIGATION
            elif 'paragraph' in block_type_str:
                dt_block_type = BlockType.PARAGRAPH
                structural_role = StructuralRole.CONTENT
            elif 'image' in block_type_str:
                # Create image block
                image_path = getattr(block, 'image_path', f'migrated_image_{block_id}.png')
                image_block = create_image_block(
                    block_id=block_id,
                    image_path=image_path,
                    bbox=bbox,
                    page_number=page_num,
                    structural_role=StructuralRole.ILLUSTRATION,
                    extraction_method="migrated_from_structured_document_model"
                )
                pages_dict[page_num].add_block(image_block)
                continue
            elif 'table' in block_type_str:
                # Create table block
                rows = getattr(block, 'rows', [])
                headers = getattr(block, 'headers', None)
                table_block = create_table_block(
                    block_id=block_id,
                    rows=rows,
                    bbox=bbox,
                    page_number=page_num,
                    headers=headers,
                    structural_role=StructuralRole.DATA,
                    extraction_method="migrated_from_structured_document_model"
                )
                pages_dict[page_num].add_block(table_block)
                continue
            else:
                dt_block_type = BlockType.TEXT
                structural_role = StructuralRole.CONTENT
            
            # Create text block
            text_block = create_text_block(
                block_id=block_id,
                text=original_text,
                bbox=bbox,
                page_number=page_num,
                block_type=dt_block_type,
                structural_role=structural_role,
                extraction_method="migrated_from_structured_document_model"
            )
            pages_dict[page_num].add_block(text_block)
        
        # Add pages to document in order
        for page_num in sorted(pages_dict.keys()):
            digital_twin_doc.add_page(pages_dict[page_num])
        
        # Update total pages
        digital_twin_doc.total_pages = len(pages_dict)
        
        logger.info(f"Successfully migrated structured document '{title}' with {len(content_blocks)} blocks to Digital Twin format")
        return digital_twin_doc
        
    except Exception as e:
        logger.error(f"Failed to migrate structured document data: {e}")
        # Return minimal document model
        return DocumentModel(
            title="Migration Failed",
            filename="migrated_document_error",
            document_metadata={'migration_error': str(e)},
            extraction_method="migrated_from_structured_document_model"
        )

class BibliographyEntry(BaseModel):
    """
    Represents a bibliography entry that should be preserved without translation.
    
    Bibliography entries contain author names, publication details, and other
    metadata that should remain in their original form.
    """
    
    entry_id: str = Field(..., description="Unique identifier for bibliography entry")
    original_text: str = Field(..., description="Original bibliography entry text")
    entry_type: str = Field(default="generic", description="Type of bibliography entry (book, article, etc.)")
    
    # Parsed components (optional)
    authors: List[str] = Field(default_factory=list, description="Author names")
    title: str = Field(default="", description="Publication title")
    year: str = Field(default="", description="Publication year")
    publisher: str = Field(default="", description="Publisher information")
    doi: str = Field(default="", description="DOI if available")
    url: str = Field(default="", description="URL if available")
    
    # Formatting preservation
    formatting_notes: List[str] = Field(default_factory=list, description="Formatting preservation notes")
    preserve_as_is: bool = Field(default=True, description="Whether to preserve exactly as found")
    
    def should_translate(self) -> bool:
        """Bibliography entries should not be translated"""
        return False
    
    def get_display_text(self) -> str:
        """Get the text for display (always original for bibliography)"""
        return self.original_text

def is_bibliography_content(text: str) -> bool:
    """
    Detect if text content is part of a bibliography section.
    
    This function identifies bibliography content that should be preserved
    without translation.
    """
    if not text or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower().strip()
    
    # Bibliography section headers
    bibliography_headers = [
        'bibliography', 'references', 'works cited', 'sources',
        'βιβλιογραφία', 'αναφορές', 'πηγές',
        'bibliographie', 'références', 'literatur'
    ]
    
    # Check if this is a bibliography section header
    if any(header in text_lower for header in bibliography_headers):
        return True
    
    # Bibliography entry patterns
    bibliography_patterns = [
        r'^\d+\.\s+[A-Z][a-z]+,\s*[A-Z]\..*\(\d{4}\)',  # Numbered entries with author and year
        r'^[A-Z][a-z]+,\s*[A-Z]\..*\(\d{4}\)',  # Author-year format
        r'^\[[^\]]+\]\s*[A-Z][a-z]+,\s*[A-Z]\.',  # Bracketed citations
        r'doi:\s*10\.\d+',  # DOI patterns
        r'ISBN:\s*\d+',  # ISBN patterns
        r'pp\.\s*\d+-\d+',  # Page ranges
        r'Vol\.\s*\d+',  # Volume numbers
        r'No\.\s*\d+',  # Issue numbers
        r'[A-Z][a-z]+\s+Press',  # University Press
        r'Journal\s+of\s+[A-Z]',  # Journal names
        r'Proceedings\s+of\s+[A-Z]',  # Proceedings
    ]
    
    # Check for bibliography entry patterns
    import re
    for pattern in bibliography_patterns:
        if re.search(pattern, text):
            return True
    
    # Check for author name patterns (common in bibliography)
    author_patterns = [
        r'[A-Z][a-z]+,\s*[A-Z]\.',  # Lastname, F.
        r'[A-Z][a-z]+,\s*[A-Z][a-z]+',  # Lastname, Firstname
        r'[A-Z]\.\s*[A-Z][a-z]+',  # F. Lastname
    ]
    
    author_matches = sum(1 for pattern in author_patterns if re.search(pattern, text))
    
    # If multiple author patterns and publication indicators, likely bibliography
    if author_matches >= 2 and any(indicator in text_lower for indicator in ['(', ')', '.', ',']):
        return True
    
    return False

def classify_bibliography_entry_type(text: str) -> str:
    """
    Classify the type of bibliography entry.
    
    This helps with proper formatting preservation.
    """
    if not text:
        return "generic"
    
    text_lower = text.lower()
    
    # Journal article indicators
    if any(indicator in text_lower for indicator in ['journal', 'vol.', 'no.', 'pp.']):
        return "journal_article"
    
    # Book indicators
    if any(indicator in text_lower for indicator in ['press', 'publisher', 'isbn']):
        return "book"
    
    # Conference paper indicators
    if any(indicator in text_lower for indicator in ['proceedings', 'conference', 'symposium']):
        return "conference_paper"
    
    # Thesis indicators
    if any(indicator in text_lower for indicator in ['thesis', 'dissertation', 'phd', 'master']):
        return "thesis"
    
    # Web source indicators
    if any(indicator in text_lower for indicator in ['http', 'www', 'url', 'retrieved']):
        return "web_source"
    
    return "generic"

logger.info("Digital Twin Document Model loaded - unified data schema ready") 