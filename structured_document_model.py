"""
Structured Document Model for PDF Translation Pipeline Refactoring

This module implements the exact structured document model specified in the refactoring requirements.
It provides a flat list of ContentBlock objects that maintain document integrity throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Enumeration of all possible content block types in a document."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    IMAGE_PLACEHOLDER = "image_placeholder"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    EQUATION = "equation"
    CAPTION = "caption"
    METADATA = "metadata"
    PAGE_BREAK = "page_break"

@dataclass
class ContentBlock:
    """Abstract base class for all document content blocks."""
    block_type: ContentType
    original_text: str  # The raw text extracted
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_num: Optional[int] = None
    formatting: Optional[Dict[str, Any]] = None
    block_id: Optional[str] = None  # Unique identifier for the block

    def __post_init__(self):
        """Validate the content block after initialization."""
        # Generate unique block ID if not provided
        if self.block_id is None:
            import uuid
            self.block_id = str(uuid.uuid4())

        if not isinstance(self.block_type, ContentType):
            raise ValueError(f"block_type must be a ContentType enum, got {type(self.block_type)}")
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must be a tuple of 4 floats, got {self.bbox}")

@dataclass
class Heading(ContentBlock):
    """Represents a heading in the document."""
    level: int = 1
    content: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.level <= 6:
            raise ValueError(f"Heading level must be between 1 and 6, got {self.level}")
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Paragraph(ContentBlock):
    """Represents a paragraph of text."""
    content: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class ImagePlaceholder(ContentBlock):
    """Represents an image placeholder with metadata and spatial relationships."""
    image_path: str = ""
    caption: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    ocr_text: Optional[str] = None
    translation_needed: bool = False
    # Spatial layout enhancements
    caption_block_id: Optional[str] = None  # ID of associated caption block
    spatial_relationship: Optional[str] = None  # "before", "after", "alongside", "wrapped"
    reading_order_position: Optional[int] = None  # Position in spatial reading order

    def __post_init__(self):
        super().__post_init__()
        # For images, original_text might be empty or contain OCR text
        if self.ocr_text and not self.original_text:
            self.original_text = self.ocr_text

@dataclass
class Table(ContentBlock):
    """Represents a table in the document."""
    markdown_content: str = ""
    rows: Optional[List[List[str]]] = None
    headers: Optional[List[str]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.markdown_content and self.original_text:
            self.markdown_content = self.original_text.strip()

@dataclass
class CodeBlock(ContentBlock):
    """Represents a code block or preformatted text."""
    language: Optional[str] = None
    content: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class ListItem(ContentBlock):
    """Represents a list item (bulleted or numbered)."""
    content: str = ""
    list_type: str = "bullet"  # "bullet", "numbered", "lettered"
    level: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Footnote(ContentBlock):
    """Represents a footnote."""
    content: str = ""
    reference_id: Optional[str] = None
    source_block: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Equation(ContentBlock):
    """Represents a mathematical equation."""
    latex_content: Optional[str] = None
    content: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Caption(ContentBlock):
    """Represents a caption for figures, tables, etc. with formal linking support."""
    content: str = ""
    target_type: Optional[str] = None  # "figure", "table", etc.
    target_block_id: Optional[str] = None  # ID of the associated content block
    spatial_proximity: Optional[float] = None  # Distance to target block

    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Metadata(ContentBlock):
    """Represents document metadata or processing artifacts."""
    content: str = ""
    metadata_type: Optional[str] = None  # "page_number", "header", "footer", "artifact"
    
    def __post_init__(self):
        super().__post_init__()
        if not self.content and self.original_text:
            self.content = self.original_text.strip()

@dataclass
class Document:
    """The top-level container for the entire structured document."""
    title: str
    content_blocks: List[ContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_filepath: Optional[str] = None
    total_pages: Optional[int] = None
    
    def __post_init__(self):
        """Validate and initialize the document."""
        if not self.title:
            self.title = "Untitled Document"
        
        # Validate all content blocks
        for i, block in enumerate(self.content_blocks):
            if not isinstance(block, ContentBlock):
                raise ValueError(f"Content block {i} is not a ContentBlock instance: {type(block)}")
    
    def add_content_block(self, block: ContentBlock) -> None:
        """Add a content block to the document."""
        if not isinstance(block, ContentBlock):
            raise ValueError(f"Expected ContentBlock, got {type(block)}")
        self.content_blocks.append(block)
    
    def get_blocks_by_type(self, content_type: ContentType) -> List[ContentBlock]:
        """Get all content blocks of a specific type."""
        return [block for block in self.content_blocks if block.block_type == content_type]
    
    def get_blocks_by_page(self, page_num: int) -> List[ContentBlock]:
        """Get all content blocks from a specific page."""
        return [block for block in self.content_blocks if block.page_num == page_num]
    
    def get_translatable_blocks(self) -> List[ContentBlock]:
        """Get all content blocks that should be translated."""
        translatable_types = {
            ContentType.HEADING,
            ContentType.PARAGRAPH,
            ContentType.LIST_ITEM,
            ContentType.FOOTNOTE,
            ContentType.CAPTION
        }
        return [block for block in self.content_blocks if block.block_type in translatable_types]
    
    def get_non_translatable_blocks(self) -> List[ContentBlock]:
        """Get all content blocks that should NOT be translated."""
        non_translatable_types = {
            ContentType.IMAGE_PLACEHOLDER,
            ContentType.TABLE,
            ContentType.CODE_BLOCK,
            ContentType.EQUATION,
            ContentType.METADATA,
            ContentType.PAGE_BREAK
        }
        return [block for block in self.content_blocks if block.block_type in non_translatable_types]
    
    def generate_document_hash(self) -> str:
        """Generate a hash for the document content for caching purposes."""
        content_str = ""
        for block in self.content_blocks:
            content_str += f"{block.block_type.value}:{block.original_text}|"
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        stats = {
            'total_blocks': len(self.content_blocks),
            'total_pages': self.total_pages or max((block.page_num for block in self.content_blocks), default=0),
            'blocks_by_type': {},
            'translatable_blocks': len(self.get_translatable_blocks()),
            'non_translatable_blocks': len(self.get_non_translatable_blocks())
        }
        
        for block in self.content_blocks:
            block_type = block.block_type.value
            stats['blocks_by_type'][block_type] = stats['blocks_by_type'].get(block_type, 0) + 1
        
        return stats

def create_content_block_from_legacy(legacy_item: Dict[str, Any]) -> ContentBlock:
    """
    Factory function to create ContentBlock instances from legacy structured content items.
    This helps with the transition from the old format to the new structured model.
    """
    item_type = legacy_item.get('type', 'paragraph')
    text = legacy_item.get('text', '')
    page_num = legacy_item.get('page_num', 1)
    block_num = legacy_item.get('block_num')
    bbox = legacy_item.get('bbox', (0, 0, 0, 0))
    formatting = legacy_item.get('formatting', {})

    # Ensure bbox is a tuple of 4 floats
    if isinstance(bbox, list):
        bbox = tuple(bbox)
    if len(bbox) != 4:
        bbox = (0, 0, 0, 0)

    # Map legacy types to new ContentBlock types
    if item_type == 'image':
        return ImagePlaceholder(
            block_type=ContentType.IMAGE_PLACEHOLDER,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            image_path=legacy_item.get('filepath', ''),
            width=legacy_item.get('width'),
            height=legacy_item.get('height'),
            ocr_text=legacy_item.get('ocr_text'),
            translation_needed=legacy_item.get('translation_needed', False)
        )

    elif item_type in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(item_type[1])
        return Heading(
            block_type=ContentType.HEADING,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            level=level,
            content=text
        )

    elif item_type == 'list_item':
        return ListItem(
            block_type=ContentType.LIST_ITEM,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            content=text
        )

    elif item_type == 'footnote':
        return Footnote(
            block_type=ContentType.FOOTNOTE,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            content=text,
            source_block=legacy_item.get('source_block')
        )

    elif item_type == 'table':
        return Table(
            block_type=ContentType.TABLE,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            markdown_content=text
        )

    elif item_type == 'code_block':
        return CodeBlock(
            block_type=ContentType.CODE_BLOCK,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            content=text
        )

    else:  # Default to paragraph
        return Paragraph(
            block_type=ContentType.PARAGRAPH,
            original_text=text,
            page_num=page_num,
            bbox=bbox,
            block_num=block_num,
            formatting=formatting,
            content=text
        )

def convert_legacy_structured_content_to_document(
    legacy_content: List[Dict[str, Any]],
    title: str = "Converted Document",
    source_filepath: Optional[str] = None
) -> Document:
    """
    Convert legacy structured content format to the new Document model.
    """
    content_blocks = []

    for legacy_item in legacy_content:
        try:
            block = create_content_block_from_legacy(legacy_item)
            content_blocks.append(block)
        except Exception as e:
            logger.warning(f"Failed to convert legacy item to ContentBlock: {e}")
            logger.debug(f"Problematic item: {legacy_item}")
            # Create a fallback paragraph block
            fallback_block = Paragraph(
                block_type=ContentType.PARAGRAPH,
                original_text=legacy_item.get('text', ''),
                page_num=legacy_item.get('page_num', 1),
                bbox=legacy_item.get('bbox', (0, 0, 0, 0)),
                content=legacy_item.get('text', '')
            )
            content_blocks.append(fallback_block)

    # Calculate total pages
    total_pages = max((block.page_num for block in content_blocks), default=1)

    document = Document(
        title=title,
        content_blocks=content_blocks,
        source_filepath=source_filepath,
        total_pages=total_pages
    )

    logger.info(f"Converted legacy content to Document with {len(content_blocks)} blocks across {total_pages} pages")
    return document

def convert_document_to_legacy_format(document: Document) -> List[Dict[str, Any]]:
    """
    Convert a Document back to legacy structured content format for compatibility.
    """
    legacy_content = []

    for block in document.content_blocks:
        legacy_item = {
            'text': block.original_text,
            'page_num': block.page_num,
            'block_num': block.block_num,
            'bbox': list(block.bbox),
            'formatting': block.formatting or {}
        }

        # Map content block types back to legacy types
        if isinstance(block, ImagePlaceholder):
            legacy_item.update({
                'type': 'image',
                'filepath': block.image_path,
                'width': block.width,
                'height': block.height,
                'ocr_text': block.ocr_text,
                'translation_needed': block.translation_needed
            })
        elif isinstance(block, Heading):
            legacy_item['type'] = f'h{block.level}'
            legacy_item['text'] = block.content
        elif isinstance(block, ListItem):
            legacy_item['type'] = 'list_item'
            legacy_item['text'] = block.content
        elif isinstance(block, Footnote):
            legacy_item['type'] = 'footnote'
            legacy_item['text'] = block.content
            legacy_item['source_block'] = block.source_block
        elif isinstance(block, Table):
            legacy_item['type'] = 'table'
            legacy_item['text'] = block.markdown_content
        elif isinstance(block, CodeBlock):
            legacy_item['type'] = 'code_block'
            legacy_item['text'] = block.content
        else:  # Paragraph or other
            legacy_item['type'] = 'paragraph'
            if isinstance(block, Paragraph):
                legacy_item['text'] = block.content

        legacy_content.append(legacy_item)

    return legacy_content
