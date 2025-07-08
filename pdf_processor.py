"""
Structured PDF Processor - Core Implementation

This module implements the structured document model as specified in the directives.
It replaces the flat list approach with a proper hierarchical structure that preserves
document sequence and semantic integrity throughout the translation pipeline.

Key Features:
- TextBlock dataclass with sequence_id and bounding box coordinates
- Coordinate-based sorting for correct reading order
- Semantic cohesion through text block merging
- Input validation and sanitization
- Strict separation of concerns
"""

import os
import logging
import fitz  # PyMuPDF
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type enumeration for structured processing"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    CAPTION = "caption"

@dataclass
class TextBlock:
    """
    Structured text block with complete metadata for sequence preservation.
    
    This is the core data structure that replaces the flat list approach.
    Each TextBlock maintains its original position and relationship to other blocks
    through coordinates and sequence_id.
    """
    text: str
    page_num: int
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bbox: Tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    font_size: float = 12.0
    font_family: str = "Arial"
    font_weight: str = "normal"
    font_style: str = "normal"
    color: int = 0
    content_type: ContentType = ContentType.TEXT
    confidence: float = 1.0
    block_type: str = "text"
    
    def __post_init__(self):
        """Validate and initialize the text block"""
        if not self.text or not self.text.strip():
            raise ValueError("TextBlock must have non-empty text content")
        
        if len(self.bbox) != 4:
            raise ValueError("bbox must be a tuple of 4 floats (x0, y0, x1, y1)")
        
        # Ensure sequence_id is unique
        if not self.sequence_id:
            self.sequence_id = str(uuid.uuid4())
    
    def get_coordinates(self) -> Tuple[float, float, float, float]:
        """Get bounding box coordinates (x0, y0, x1, y1)"""
        return self.bbox
    
    def get_vertical_position(self) -> float:
        """Get vertical position (y0) for sorting"""
        return self.bbox[1]
    
    def get_horizontal_position(self) -> float:
        """Get horizontal position (x0) for sorting"""
        return self.bbox[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'page_num': self.page_num,
            'sequence_id': self.sequence_id,
            'bbox': self.bbox,
            'font_size': self.font_size,
            'font_family': self.font_family,
            'font_weight': self.font_weight,
            'font_style': self.font_style,
            'color': self.color,
            'content_type': self.content_type.value,
            'confidence': self.confidence,
            'block_type': self.block_type
        }

class StructuredPDFProcessor:
    """
    Structured PDF processor that implements the new document model.
    
    This class replaces the flat extraction approach with a structured,
    sequence-aware processing pipeline that preserves document integrity.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 Structured PDF Processor initialized")
    
    def extract_document_structure(self, pdf_path: str) -> List[List[TextBlock]]:
        """
        Extract document structure as list[list[TextBlock]].
        
        This is the primary data structure for the entire application.
        The outer list represents the document, each inner list represents a page.
        Each TextBlock maintains its original position and sequence information.
        """
        try:
            self.logger.info(f"📄 Extracting structured content from: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            document_structure = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_blocks = self._extract_page_blocks(page, page_num)
                # --- Extraction Order Debug Export ---
                extraction_order_debug = []
                for idx, block in enumerate(page_blocks):
                    extraction_order_debug.append({
                        'order': idx,
                        'block_type': getattr(block, 'block_type', None),
                        'page_number': page_num + 1,
                        'bbox': getattr(block, 'bbox', None),
                        'text_snippet': getattr(block, 'text', '')[:60] if hasattr(block, 'text') else None
                    })
                debug_dir = os.path.join('output', 'extraction_debug')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"structured_page_{page_num+1}_extraction_order.json")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(extraction_order_debug, f, ensure_ascii=False, indent=2)
                # --- End Extraction Order Debug Export ---
                
                # Sort blocks to establish correct reading order
                sorted_blocks = self._sort_blocks_by_reading_order(page_blocks)
                
                # Assign global sequence IDs
                self._assign_sequence_ids(sorted_blocks, page_num)
                
                # Merge related blocks into coherent paragraphs
                merged_blocks = self._merge_text_blocks(sorted_blocks)
                
                document_structure.append(merged_blocks)
                
                self.logger.info(f"   Page {page_num + 1}: {len(merged_blocks)} blocks extracted")
            
            doc.close()
            
            total_blocks = sum(len(page_blocks) for page_blocks in document_structure)
            self.logger.info(f"✅ Document structure extracted: {len(document_structure)} pages, {total_blocks} total blocks")
            
            return document_structure
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting document structure: {e}")
            raise
    
    def _extract_page_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract all text blocks from a page with complete metadata"""
        text_blocks = []
        
        try:
            # Get text dictionary with detailed positioning
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_text_blocks = self._extract_text_blocks_from_block(block, page_num)
                    text_blocks.extend(block_text_blocks)
                elif "image" in block:  # Image block
                    image_block = self._create_image_block(block, page_num)
                    if image_block:
                        text_blocks.append(image_block)
            
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"Error extracting blocks from page {page_num}: {e}")
            return []
    
    def _extract_text_blocks_from_block(self, block: Dict, page_num: int) -> List[TextBlock]:
        """Extract text blocks from a PDF block with enhanced metadata"""
        text_blocks = []
        
        try:
            for line in block.get("lines", []):
                line_text = ""
                primary_span = None
                
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        # Sanitize text before processing
                        sanitized_text = self._sanitize_text(span_text)
                        if sanitized_text:
                            line_text += sanitized_text + " "
                            if primary_span is None or len(sanitized_text) > len(primary_span.get("text", "")):
                                primary_span = span
                
                line_text = line_text.strip()
                if not line_text or not primary_span:
                    continue
                
                # Extract rich formatting metadata
                font_flags = primary_span.get("flags", 0)
                font_weight = "bold" if (font_flags & 2**4) else "normal"
                font_style = "italic" if (font_flags & 2**1) else "normal"
                
                # Determine content type based on formatting and position
                content_type = self._determine_content_type(
                    line_text, primary_span.get("size", 12.0), font_weight, font_style
                )
                
                text_block = TextBlock(
                    text=line_text,
                    page_num=page_num,
                    bbox=tuple(line["bbox"]),
                    font_size=primary_span.get("size", 12.0),
                    font_family=primary_span.get("font", ""),
                    font_weight=font_weight,
                    font_style=font_style,
                    color=primary_span.get("color", 0),
                    content_type=content_type,
                    confidence=1.0,  # PyMuPDF extraction is considered perfect
                    block_type='text'
                )
                
                text_blocks.append(text_block)
            
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"Error extracting text blocks: {e}")
            return []
    
    def _create_image_block(self, block: Dict, page_num: int) -> Optional[TextBlock]:
        """Create an image block from PDF image data"""
        try:
            bbox = block.get("bbox", [0, 0, 0, 0])
            
            return TextBlock(
                text="[Image]",
                page_num=page_num,
                bbox=tuple(bbox),
                content_type=ContentType.IMAGE,
                block_type='image'
            )
            
        except Exception as e:
            self.logger.error(f"Error creating image block: {e}")
            return None
    
    def _sort_blocks_by_reading_order(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Sort blocks to establish correct reading order.
        
        Primary sorting key: vertical position (y0)
        Secondary sorting key: horizontal position (x0)
        """
        try:
            # Sort by vertical position first, then horizontal position
            sorted_blocks = sorted(blocks, key=lambda block: (block.get_vertical_position(), block.get_horizontal_position()))
            
            self.logger.debug(f"Sorted {len(sorted_blocks)} blocks by reading order")
            return sorted_blocks
            
        except Exception as e:
            self.logger.error(f"Error sorting blocks: {e}")
            return blocks
    
    def _assign_sequence_ids(self, blocks: List[TextBlock], page_num: int) -> None:
        """Assign globally unique, sequential sequence_id to blocks"""
        try:
            for i, block in enumerate(blocks):
                # Create a globally unique sequence ID that includes page and position information
                block.sequence_id = f"page_{page_num}_block_{i:04d}_{uuid.uuid4().hex[:8]}"
            
            self.logger.debug(f"Assigned sequence IDs to {len(blocks)} blocks on page {page_num}")
            
        except Exception as e:
            self.logger.error(f"Error assigning sequence IDs: {e}")
    
    def _merge_text_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Merge consecutive blocks that form a single paragraph.
        
        This function analyzes sorted blocks and merges consecutive blocks
        that form a single paragraph based on proximity and alignment.
        """
        if not blocks:
            return []
        
        merged_blocks = []
        current_paragraph = []
        
        for i, block in enumerate(blocks):
            if block.content_type == ContentType.IMAGE:
                # Process any accumulated paragraph
                if current_paragraph:
                    merged_block = self._create_merged_block(current_paragraph)
                    merged_blocks.append(merged_block)
                    current_paragraph = []
                
                # Add image block as-is
                merged_blocks.append(block)
                continue
            
            # Check if this block should be merged with the current paragraph
            if self._should_merge_with_paragraph(block, current_paragraph):
                current_paragraph.append(block)
            else:
                # Process accumulated paragraph
                if current_paragraph:
                    merged_block = self._create_merged_block(current_paragraph)
                    merged_blocks.append(merged_block)
                
                # Start new paragraph
                current_paragraph = [block]
        
        # Process final paragraph
        if current_paragraph:
            merged_block = self._create_merged_block(current_paragraph)
            merged_blocks.append(merged_block)
        
        self.logger.debug(f"Merged {len(blocks)} blocks into {len(merged_blocks)} coherent blocks")
        return merged_blocks
    
    def _should_merge_with_paragraph(self, block: TextBlock, current_paragraph: List[TextBlock]) -> bool:
        """Determine if a block should be merged with the current paragraph"""
        if not current_paragraph:
            return True
        
        # Get the last block in the current paragraph
        last_block = current_paragraph[-1]
        
        # Check vertical distance (should be minimal for same paragraph)
        vertical_distance = abs(block.get_vertical_position() - last_block.get_vertical_position())
        max_vertical_distance = max(block.font_size, last_block.font_size) * 1.5
        
        # Check horizontal alignment (should be consistent for same paragraph)
        horizontal_alignment_diff = abs(block.get_horizontal_position() - last_block.get_horizontal_position())
        max_horizontal_diff = 50.0  # Allow some indentation
        
        # Check if this looks like a continuation of the paragraph
        should_merge = (
            vertical_distance <= max_vertical_distance and
            horizontal_alignment_diff <= max_horizontal_diff and
            block.font_size == last_block.font_size and
            block.font_family == last_block.font_family
        )
        
        return should_merge
    
    def _create_merged_block(self, blocks: List[TextBlock]) -> TextBlock:
        """Create a merged block from multiple consecutive blocks"""
        if not blocks:
            raise ValueError("Cannot create merged block from empty list")
        
        if len(blocks) == 1:
            return blocks[0]
        
        # Combine text from all blocks
        combined_text = " ".join(block.text for block in blocks)
        
        # Use metadata from the first block
        first_block = blocks[0]
        
        # Calculate combined bounding box
        min_x = min(block.bbox[0] for block in blocks)
        min_y = min(block.bbox[1] for block in blocks)
        max_x = max(block.bbox[2] for block in blocks)
        max_y = max(block.bbox[3] for block in blocks)
        combined_bbox = (min_x, min_y, max_x, max_y)
        
        # Create merged block
        merged_block = TextBlock(
            text=combined_text,
            page_num=first_block.page_num,
            sequence_id=first_block.sequence_id,  # Use first block's sequence ID
            bbox=combined_bbox,
            font_size=first_block.font_size,
            font_family=first_block.font_family,
            font_weight=first_block.font_weight,
            font_style=first_block.font_style,
            color=first_block.color,
            content_type=first_block.content_type,
            confidence=first_block.confidence,
            block_type=first_block.block_type
        )
        
        return merged_block
    
    def _determine_content_type(self, text: str, font_size: float, font_weight: str, font_style: str) -> ContentType:
        """Determine the content type based on text characteristics and formatting"""
        text_lower = text.lower().strip()
        
        # Check for headings based on font size and weight
        if font_size >= 16.0 or font_weight == "bold":
            if len(text.split()) <= 10:  # Short text is likely a heading
                return ContentType.HEADING
        
        # Check for list items
        if re.match(r'^[\d\-•\*]\s+', text):
            return ContentType.LIST_ITEM
        
        # Check for captions
        if re.match(r'^(Figure|Table|Image|Fig\.|Tab\.)', text, re.IGNORECASE):
            return ContentType.CAPTION
        
        # Check for footnotes
        if re.match(r'^\d+\.\s+', text) and font_size < 12.0:
            return ContentType.FOOTNOTE
        
        # Default to paragraph
        return ContentType.PARAGRAPH
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize extracted text to remove PDF metadata artifacts.
        
        This function uses regular expressions to scrub the extracted text
        of any PDF metadata artifacts before storing in TextBlock objects.
        """
        if not text:
            return ""
        
        # Remove PDF metadata patterns
        patterns_to_remove = [
            r'_Toc_Bookmark_\d+',  # TOC bookmark patterns with numbers
            r'_\d+_',  # Internal PDF numbering
            r'\[.*?\]',  # Square bracket metadata
            r'\{.*?\}',  # Curly bracket metadata
            r'\\[a-zA-Z]+',  # PDF escape sequences
        ]
        
        sanitized_text = text
        
        for pattern in patterns_to_remove:
            sanitized_text = re.sub(pattern, '', sanitized_text)
        
        # Remove excessive whitespace
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
        sanitized_text = sanitized_text.strip()
        
        return sanitized_text
    
    def get_document_statistics(self, document_structure: List[List[TextBlock]]) -> Dict[str, Any]:
        """Get comprehensive statistics about the document structure"""
        total_pages = len(document_structure)
        total_blocks = sum(len(page_blocks) for page_blocks in document_structure)
        
        content_type_counts = {}
        for page_blocks in document_structure:
            for block in page_blocks:
                content_type = block.content_type.value
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        return {
            'total_pages': total_pages,
            'total_blocks': total_blocks,
            'content_type_distribution': content_type_counts,
            'average_blocks_per_page': total_blocks / total_pages if total_pages > 0 else 0
        }

# Create global instance
structured_pdf_processor = StructuredPDFProcessor() 