#!/usr/bin/env python3
"""
Processing Strategies for PyMuPDF-YOLO Integration

This module implements the three processing strategies:
1. Direct Text Processing - for pure text documents (no graph overhead)
2. Minimal Graph Processing - for mixed content (area-level nodes)
3. Comprehensive Graph Processing - for visual-heavy documents (full analysis)
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pydantic import ValidationError
from pymupdf_yolo_processor import PageModel, LayoutArea
from types import SimpleNamespace
import traceback
from models import ProcessResult, ElementModel, PageModel, ContentElement, PageContent
from config_manager import Config

# Import existing services
# try:
    # from translation_service_enhanced import enhanced_translation_service
    # TRANSLATION_AVAILABLE = True
# except ImportError:
    # TRANSLATION_AVAILABLE = False
    # logger.warning("Enhanced translation service not available")

try:
    from document_model import DocumentGraph, add_yolo_detections_to_graph, add_ocr_text_regions_to_graph, build_association_matrix_from_graph, populate_document_graph, refine_document_graph, DocumentReconstructor
    DOCUMENT_MODEL_AVAILABLE = True
except ImportError:
    DOCUMENT_MODEL_AVAILABLE = False
    logger.warning("Document model not available")

# Import Digital Twin model and enhanced processor
try:
    from digital_twin_model import DocumentModel, PageModel as DigitalTwinPageModel, TextBlock as DigitalTwinTextBlock, BlockType, StructuralRole
    from pymupdf_yolo_processor import PyMuPDFYOLOProcessor
    DIGITAL_TWIN_AVAILABLE = True
except ImportError:
    DIGITAL_TWIN_AVAILABLE = False
    logger.warning("Digital Twin model or enhanced processor not available")

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing strategy execution"""
    success: bool
    strategy: str
    processing_time: float
    content: Dict[str, Any]
    statistics: Dict[str, Any]
    error: Optional[str] = None


class TableProcessor:
    """Process and translate detected tables using structured approach"""
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        self.logger.info("🔧 Table Processor initialized")
    
    def parse_table_structure(self, mapped_content_area: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse text blocks and coordinates to reconstruct table grid structure.
        
        This method analyzes the spatial distribution of text blocks to infer
        the table's row and column structure.
        """
        try:
            text_blocks = mapped_content_area.get('text_blocks', [])
            if not text_blocks:
                return {'rows': [], 'header_row': None, 'error': 'No text blocks found'}
            
            # Sort text blocks by vertical position (top to bottom)
            sorted_blocks = sorted(text_blocks, key=lambda b: b.bbox[1] if hasattr(b, 'bbox') else 0)
            
            # Group blocks into rows based on Y-coordinate proximity
            rows = []
            current_row = []
            current_y = None
            y_tolerance = 10  # Points tolerance for same row
            
            for block in sorted_blocks:
                block_y = block.bbox[1] if hasattr(block, 'bbox') else 0
                
                if current_y is None or abs(block_y - current_y) <= y_tolerance:
                    # Same row
                    current_row.append(block)
                    current_y = block_y
                else:
                    # New row
                    if current_row:
                        rows.append(current_row)
                    current_row = [block]
                    current_y = block_y
            
            # Don't forget the last row
            if current_row:
                rows.append(current_row)
            
            # Sort blocks within each row by X-coordinate (left to right)
            for row in rows:
                row.sort(key=lambda b: b.bbox[0] if hasattr(b, 'bbox') else 0)
            
            # Extract text content for each cell
            table_grid = []
            header_row = None
            
            for i, row in enumerate(rows):
                row_cells = []
                for block in row:
                    cell_text = block.text if hasattr(block, 'text') else str(block)
                    row_cells.append(cell_text.strip())
                
                if i == 0 and len(rows) > 1:
                    # First row might be header
                    header_row = row_cells
                
                table_grid.append(row_cells)
            
            self.logger.info(f"📊 Parsed table structure: {len(table_grid)} rows, {len(table_grid[0]) if table_grid else 0} columns")
            
            return {
                'rows': table_grid,
                'header_row': header_row,
                'num_rows': len(table_grid),
                'num_cols': len(table_grid[0]) if table_grid else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to parse table structure: {e}")
            return {'rows': [], 'header_row': None, 'error': str(e)}
    
    async def translate_table(self, table_structure: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Translate table by serializing to Markdown format and using robust translation.
        
        As per Directive: Uses single string to leverage existing robust translation logic,
        rather than translating cell by cell.
        """
        try:
            rows = table_structure.get('rows', [])
            if not rows:
                return {'translated_rows': [], 'error': 'No table rows to translate'}
            
            # Serialize table to Markdown format
            markdown_table = self._serialize_table_to_markdown(rows)
            
            if not markdown_table.strip():
                return {'translated_rows': [], 'error': 'Empty table content'}
            
            self.logger.info(f"📝 Serialized table to Markdown ({len(markdown_table)} chars)")
            
            # Translate the entire table as a single string
            if self.gemini_service:
                translated_markdown = await self.gemini_service.translate_text(
                    markdown_table, target_language
                )
            else:
                # Mock translation for testing
                translated_markdown = f"[TRANSLATED_TO_{target_language.upper()}] {markdown_table}"
            
            # Parse translated Markdown back to table structure
            translated_rows = self._parse_markdown_to_table(translated_markdown)
            
            self.logger.info(f"✅ Table translation completed: {len(translated_rows)} rows")
            
            return {
                'translated_rows': translated_rows,
                'original_markdown': markdown_table,
                'translated_markdown': translated_markdown
            }
            
        except Exception as e:
            self.logger.error(f"❌ Table translation failed: {e}")
            return {'translated_rows': [], 'error': str(e)}
    
    def _serialize_table_to_markdown(self, rows: List[List[str]]) -> str:
        """Convert table rows to Markdown table format"""
        if not rows:
            return ""
        
        markdown_lines = []
        
        for i, row in enumerate(rows):
            # Clean and escape cell content
            cleaned_cells = [cell.replace('|', '\\|').replace('\n', ' ').strip() for cell in row]
            
            # Create table row
            row_markdown = "| " + " | ".join(cleaned_cells) + " |"
            markdown_lines.append(row_markdown)
            
            # Add header separator after first row
            if i == 0 and len(rows) > 1:
                separator = "| " + " | ".join(["---"] * len(cleaned_cells)) + " |"
                markdown_lines.append(separator)
        
        return "\n".join(markdown_lines)
    
    def _parse_markdown_to_table(self, markdown_text: str) -> List[List[str]]:
        """Parse translated Markdown back to table structure"""
        try:
            # Extract table lines from the markdown text (handle LLM chatter)
            lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
            table_lines = []
            
            # Find lines that look like table rows
            for line in lines:
                if line.startswith('|') and line.endswith('|') and line.count('|') >= 2:
                    table_lines.append(line)
            
            if not table_lines:
                self.logger.warning("No table lines found in markdown text")
                return []
            
            table_rows = []
            
            for line in table_lines:
                # Skip header separator lines (contains only |, -, :, and spaces)
                if all(c in '|-: ' for c in line):
                    continue
                
                # Parse table row - remove leading/trailing pipes and split
                cells = [cell.strip() for cell in line[1:-1].split('|')]
                
                # Clean translation artifacts from cells
                cleaned_cells = []
                for cell in cells:
                    cleaned_cell = cell.replace('\\|', '|')  # Unescape pipes
                    
                    # Remove translation markers for cleaner output
                    if '[TRANSLATED_TO_' in cleaned_cell:
                        # Extract the actual content after the marker
                        parts = cleaned_cell.split('] ', 1)
                        if len(parts) > 1:
                            cleaned_cell = parts[1]
                        else:
                            # Fallback: remove the marker completely
                            import re
                            cleaned_cell = re.sub(r'\[TRANSLATED_TO_[^\]]*\]\s*', '', cleaned_cell)
                    
                    cleaned_cells.append(cleaned_cell.strip())
                
                # Only add non-empty rows
                if cleaned_cells and any(cell.strip() for cell in cleaned_cells):
                    table_rows.append(cleaned_cells)
            
            self.logger.info(f"📋 Parsed {len(table_rows)} rows from translated markdown")
            return table_rows
            
        except Exception as e:
            self.logger.error(f"❌ Failed to parse Markdown table: {e}")
            return []


class DirectTextProcessor:
    """Process pure text content directly without any graph overhead"""
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        self.logger.info("🔧 Direct Text Processor initialized")
    
    def _convert_mapped_content_to_page_content(self, mapped_content: Dict[str, Any], page_number: int = 1) -> PageContent:
        """Convert mapped_content dictionary to structured PageContent object"""
        content_elements = []
        image_elements = []
        
        for area_id, area_data in mapped_content.items():
            try:
                # Handle the ACTUAL data structure from the pipeline
                if isinstance(area_data, dict):
                    # Extract data using actual keys: 'type', 'content', 'bbox', 'confidence'
                    element_type = area_data.get('type', 'text')
                    text_content = area_data.get('content', '')
                    bbox = tuple(area_data.get('bbox', [0, 0, 0, 0]))
                    confidence = area_data.get('confidence', 1.0)
                    
                    self.logger.debug(f"Processing element {area_id}: type={element_type}, content_length={len(text_content)}")
                    
                    # Create ContentElement for text-related elements
                    if element_type in ['text', 'paragraph', 'title'] and text_content:
                        content_element = ContentElement(
                            id=area_id,
                            text=text_content,
                            label=element_type,  # Use 'type' as 'label'
                            bbox=bbox,
                            confidence=confidence
                        )
                        content_elements.append(content_element)
                        self.logger.debug(f"Added text element {area_id} with {len(text_content)} characters")
                    elif element_type in ['image', 'figure', 'table']:
                        # Store image/visual elements separately
                        image_elements.append({
                            'id': area_id,
                            'label': element_type,
                            'bbox': bbox,
                            'confidence': confidence
                        })
                        self.logger.debug(f"Added image element {area_id}")
                        
                # Legacy support for object-style area_data (if needed)
                elif hasattr(area_data, 'layout_info'):
                    layout_info = area_data.layout_info
                    label = layout_info.label if hasattr(layout_info, 'label') else 'text'
                    bbox = layout_info.bbox if hasattr(layout_info, 'bbox') else (0, 0, 0, 0)
                    confidence = layout_info.confidence if hasattr(layout_info, 'confidence') else 1.0
                    text_content = getattr(area_data, 'combined_text', '')
                    
                    if label in ['text', 'paragraph', 'title'] and text_content:
                        content_element = ContentElement(
                            id=area_id,
                            text=text_content,
                            label=label,
                            bbox=bbox,
                            confidence=confidence
                        )
                        content_elements.append(content_element)
                        
            except Exception as e:
                self.logger.warning(f"Failed to process area {area_id}: {e}")
                continue
        
        self.logger.info(f"Converted mapped_content: {len(content_elements)} text elements, {len(image_elements)} image elements")
        
        return PageContent(
            page_number=page_number,
            content_elements=content_elements,
            image_elements=image_elements,
            strategy="direct_text"
        )
    
    async def process_pure_text(self, mapped_content: Dict[str, Any]) -> ProcessingResult:
        """Process text content directly without graph creation"""
        start_time = time.time()
        
        try:
            # Convert mapped_content to structured PageContent
            page_content = self._convert_mapped_content_to_page_content(mapped_content)
            
            # Extract text areas for processing
            text_areas = []
            for element in page_content.content_elements:
                text_areas.append({
                    'content': element.text,
                    'bbox': element.bbox,
                    'label': element.label,
                    'confidence': element.confidence
                })
            
            # Sort by vertical position (reading order)
            text_areas.sort(key=lambda x: x['bbox'][1])
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Pure text processing completed in {processing_time:.3f}s")
            self.logger.info(f"   Content elements processed: {len(page_content.content_elements)}")
            self.logger.info(f"   Image elements found: {len(page_content.image_elements)}")
            
            return ProcessingResult(
                success=True,
                strategy='direct_text',
                processing_time=processing_time,
                content=text_areas,
                statistics={
                    'text_areas': len(text_areas),
                    'content_elements': len(page_content.content_elements),
                    'image_elements': len(page_content.image_elements),
                    'total_text_length': sum(len(item['content']) for item in text_areas),
                    'graph_nodes': 0,  # No graph created
                    'graph_edges': 0
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Direct text processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                strategy='direct_text',
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    def _apply_semantic_filtering(self, text_elements: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Apply semantic filtering as per Sub-Directive B.
        
        Separates elements by semantic labels: headers/footers are excluded from translation,
        while content elements are included.
        
        Returns:
            tuple: (elements_to_translate, excluded_elements)
        """
        elements_to_translate = []
        excluded_elements = []
        
        for element in text_elements:
            semantic_label = element.get('semantic_label') or element.get('label', '')
            
            # Exclude headers and footers from translation (Sub-Directive B)
            if semantic_label.lower() in ['header', 'footer']:
                excluded_elements.append(element)
                self.logger.info(f"Excluding {semantic_label} from translation: '{element.get('text', '')[:50]}...'")
            else:
                elements_to_translate.append(element)
        
        self.logger.info(f"Semantic filtering: {len(elements_to_translate)} elements to translate, {len(excluded_elements)} excluded")
        return elements_to_translate, excluded_elements
    
    def _sanitize_gemini_response(self, raw_response: str) -> str:
        """
        Pre-processes malformed Gemini API responses to extract clean XML content.
        
        The Gemini API often returns responses wrapped in markdown code blocks like:
        ```xml
        <seg id="0">Hello</seg>
        ``` ```xml
        <seg id="1">World</seg>
        ```
        
        This function extracts only the <seg> tags and creates valid XML.
        """
        import re
        
        # Extract all <seg> tags and their content from the raw response
        # This handles both single and multi-line seg tags
        seg_pattern = re.compile(r'<seg\s+id=["\'](\d+)["\']>(.*?)</seg>', re.DOTALL | re.IGNORECASE)
        matches = seg_pattern.findall(raw_response)
        
        if not matches:
            self.logger.warning("No <seg> tags found in Gemini response during sanitization")
            return f"<root></root>"
        
        # Reconstruct clean XML from extracted segments
        cleaned_segments = []
        for seg_id, content in matches:
            # Clean up the content (remove excess whitespace, but preserve structure)
            clean_content = content.strip()
            cleaned_segments.append(f'<seg id="{seg_id}">{clean_content}</seg>')
        
        # Wrap in root element to create valid XML
        sanitized_xml = f"<root>{''.join(cleaned_segments)}</root>"
        
        self.logger.info(f"Sanitized Gemini response: extracted {len(matches)} segments")
        self.logger.debug(f"Sanitized XML: {sanitized_xml[:200]}{'...' if len(sanitized_xml) > 200 else ''}")
        
        return sanitized_xml

    def _parse_and_reconstruct_translation(self, translated_text: str) -> dict:
        """
        Parses the XML-like response from Gemini and reconstructs the translated segments.
        Uses robust XML parser with pre-processing to handle malformed API responses.
        """
        translated_segments = {}
        import xml.etree.ElementTree as ET
        
        try:
            # CRITICAL: Pre-process the raw response to handle malformed API responses
            sanitized_xml = self._sanitize_gemini_response(translated_text)
            
            # Parse the sanitized XML using ElementTree
            root = ET.fromstring(sanitized_xml)
            
            # Find all <seg> tags and extract their content
            seg_elements = root.findall('.//seg')
            
            if not seg_elements:
                self.logger.warning(f"XML parser could not find any <seg> tags in the response.")
            
            for seg_element in seg_elements:
                seg_id = seg_element.get('id')
                seg_text = seg_element.text or ''  # Handle empty elements
                
                if seg_id:
                    # Clean up the text and store
                    translated_segments[seg_id] = seg_text.strip()
                else:
                    self.logger.warning(f"Found <seg> element without 'id' attribute")
            
            if not translated_segments:
                self.logger.warning(f"XML parser extracted 0 segments from Gemini response. Raw response: {translated_text}")
                
        except ET.ParseError as e:
            self.logger.error(f"XML parsing failed - malformed response from Gemini: {e}")
            self.logger.error(f"Problematic Gemini response: {translated_text}")
            
            # Fallback to regex for malformed XML (last resort)
            self.logger.info("Attempting fallback regex parsing...")
            try:
                import re
                seg_pattern = re.compile(r'<seg id="(\d+?)">(.*?)</seg>', re.DOTALL)
                matches = seg_pattern.findall(translated_text)
                for seg_id, seg_text in matches:
                    if seg_id:
                        translated_segments[seg_id] = seg_text.strip()
                self.logger.info(f"Fallback regex extracted {len(translated_segments)} segments")
            except Exception as fallback_error:
                self.logger.error(f"Fallback regex parsing also failed: {fallback_error}")
                
        except Exception as e:
            self.logger.error(f"Unexpected error during XML parsing of Gemini response: {e}")
            self.logger.error(f"Problematic Gemini response: {translated_text}")
            
        return translated_segments
    
    async def translate_direct_text(self, text_elements: list[dict], target_language: str) -> list[dict]:
        """
        Translates a list of text elements using a robust, tag-based
        reconstruction method to ensure perfect data integrity.
        Enhanced with semantic filtering as per Sub-Directive B.
        SIMPLIFIED: Single API call for true smart batching.
        """
        # Apply semantic filtering (Sub-Directive B)
        elements_to_translate, excluded_elements = self._apply_semantic_filtering(text_elements)
        # If no elements to translate, return original elements
        if not elements_to_translate:
            self.logger.warning("No elements to translate after semantic filtering")
            return text_elements
        
        # Create intelligent chunks with coherence preservation
        chunks = self._create_intelligent_chunks(elements_to_translate, max_chars=13000)
        self.logger.info(f"📦 Created {len(chunks)} intelligent chunks for coherent translation")
        
        # Process each chunk while maintaining order and coherence
        all_translated_blocks = []
        global_element_id = 0
        
        for chunk_idx, chunk_elements in enumerate(chunks):
            self.logger.info(f"Translating chunk {chunk_idx+1}/{len(chunks)} using tag-based reconstruction...")
            
            # Step 1: Wrap each element in a unique, numbered tag (no sanitization)
            tagged_payload_parts = []
            chunk_elements_map = {}
            
            for j, element in enumerate(chunk_elements):
                text = element.get('text', '')
                if text:
                    tagged_payload_parts.append(f'<seg id="{global_element_id}">{text}</seg>')
                    chunk_elements_map[global_element_id] = element
                    global_element_id += 1
            
            if not tagged_payload_parts:
                self.logger.warning(f"No text content to translate in chunk {chunk_idx+1}")
                continue
            
            source_text_for_api = "\n".join(tagged_payload_parts)
            
            # Step 2: API call for this chunk
            try:
                translated_blob = await self.gemini_service.translate_text(source_text_for_api, target_language)
                
                # Step 3: Use the robust parser with sanitization
                sanitized_response = self._sanitize_gemini_response(translated_blob)
                self.logger.info(f"Sanitized Gemini response: extracted {sanitized_response.count('<seg')} segments")
                translated_segments = self._parse_and_reconstruct_translation(sanitized_response)
                self.logger.info(f"Parsed {len(translated_segments)} segments from {len(chunk_elements_map)} original elements")
                
                # Step 4: Reconstruct blocks for this chunk
                for elem_id, original_element in chunk_elements_map.items():
                    if str(elem_id) in translated_segments:
                        translated_text = translated_segments[str(elem_id)]
                    else:
                        original_text = original_element.get('text', '')
                        self.logger.warning(f"Translation missing for segment ID {elem_id}. Original content: '{original_text}'. Using original text.")
                        translated_text = original_text
                    
                    all_translated_blocks.append({
                        'type': 'text',
                        'text': translated_text,
                        'label': original_element.get('label', 'paragraph'),
                        'bbox': original_element.get('bbox')
                    })
                    
            except Exception as e:
                self.logger.error(f"❌ Chunk {chunk_idx+1} translation failed: {e}")
                # Add original elements as fallback
                for element in chunk_elements:
                    all_translated_blocks.append({
                        'type': 'text',
                        'text': element.get('text', ''),
                        'label': element.get('label', 'paragraph'),
                        'bbox': element.get('bbox')
                    })
        
        # Merge translated elements with excluded elements in original order
        final_blocks = []
        translated_index = 0
        for original_element in text_elements:
            semantic_label = original_element.get('semantic_label') or original_element.get('label', '')
            if semantic_label.lower() in ['header', 'footer']:
                final_blocks.append({
                    'type': 'text',
                    'text': original_element.get('text', ''),
                    'label': original_element.get('label', 'paragraph'),
                    'bbox': original_element.get('bbox'),
                    'excluded_from_translation': True
                })
            else:
                if translated_index < len(all_translated_blocks):
                    final_blocks.append(all_translated_blocks[translated_index])
                    translated_index += 1
                else:
                    final_blocks.append({
                        'type': 'text',
                        'text': original_element.get('text', ''),
                        'label': original_element.get('label', 'paragraph'),
                        'bbox': original_element.get('bbox')
                    })
        
        self.logger.info(f"Tag-based translation completed. Created {len(final_blocks)} blocks ({len(all_translated_blocks)} translated, {len(excluded_elements)} excluded).")
        return final_blocks
    
    def _create_intelligent_chunks(self, text_elements: list[dict], max_chars: int = 13000) -> list[list[dict]]:
        """
        Create intelligent chunks that preserve text coherence while respecting character limits.
        
        Strategy:
        1. Prioritize semantic breaks (sentences ending with ., !, ?)
        2. Respect paragraph boundaries 
        3. Avoid breaking within headings/titles
        4. Keep related content together (e.g., list items)
        5. Never exceed max_chars limit
        """
        if not text_elements:
            return []
        
        chunks = []
        current_chunk = []
        current_chars = 0
        
        for i, element in enumerate(text_elements):
            element_text = element.get('text', '')
            element_chars = len(element_text)
            
            # Calculate overhead for XML tags: <seg id="X">...</seg>
            xml_overhead = len(f'<seg id="{i}"></seg>')
            total_element_size = element_chars + xml_overhead
            
            # Check if adding this element would exceed the limit
            would_exceed = (current_chars + total_element_size) > max_chars
            
            # Special handling for different content types
            element_label = element.get('label', 'paragraph').lower()
            
            # If this element alone exceeds the limit, we need to handle it specially
            if total_element_size > max_chars:
                # Close current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_chars = 0
                
                # Split large element into smaller parts while preserving sentence boundaries
                split_elements = self._split_large_element(element, max_chars)
                for split_elem in split_elements:
                    chunks.append([split_elem])
                continue
            
            # Decision logic for chunk boundaries
            should_break_chunk = False
            
            if would_exceed:
                # Check if we can make a coherent break
                if current_chunk:
                    # Look at the previous element to determine if this is a good break point
                    prev_element = current_chunk[-1] if current_chunk else None
                    should_break_chunk = self._is_good_break_point(prev_element, element)
                else:
                    # First element in chunk is too big - should not happen due to check above
                    should_break_chunk = True
            
            # Handle chunk break
            if should_break_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0
            
            # Add element to current chunk
            current_chunk.append(element)
            current_chars += total_element_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        # Log chunk statistics
        chunk_stats = []
        for i, chunk in enumerate(chunks):
            chunk_chars = sum(len(elem.get('text', '')) for elem in chunk)
            chunk_stats.append(f"Chunk {i+1}: {len(chunk)} elements, {chunk_chars} chars")
        
        self.logger.info(f"📊 Intelligent chunking statistics:")
        for stat in chunk_stats:
            self.logger.info(f"   {stat}")
        
        return chunks
    
    def _split_large_element(self, element: dict, max_chars: int) -> list[dict]:
        """Split a large element into smaller coherent parts"""
        text = element.get('text', '')
        if not text:
            return [element]
        
        # Try to split on sentence boundaries first
        sentences = self._split_on_sentence_boundaries(text)
        
        parts = []
        current_part = ""
        
        for sentence in sentences:
            # Account for XML overhead
            xml_overhead = len('<seg id="0"></seg>')
            sentence_with_overhead = len(sentence) + xml_overhead
            
            if len(current_part) + len(sentence) + xml_overhead <= max_chars:
                current_part += sentence
            else:
                # Save current part if it has content
                if current_part.strip():
                    part_element = element.copy()
                    part_element['text'] = current_part.strip()
                    parts.append(part_element)
                
                # Start new part
                if sentence_with_overhead <= max_chars:
                    current_part = sentence
                else:
                    # Even a single sentence is too long - split it arbitrarily
                    word_parts = self._split_by_words(sentence, max_chars - xml_overhead)
                    for word_part in word_parts:
                        part_element = element.copy()
                        part_element['text'] = word_part
                        parts.append(part_element)
                    current_part = ""
        
        # Add the last part
        if current_part.strip():
            part_element = element.copy()
            part_element['text'] = current_part.strip()
            parts.append(part_element)
        
        return parts if parts else [element]
    
    def _split_on_sentence_boundaries(self, text: str) -> list[str]:
        """Split text on sentence boundaries while preserving structure"""
        import re
        
        # Split on sentence endings but preserve the punctuation
        sentence_pattern = r'([.!?]+\s+)'
        parts = re.split(sentence_pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            if re.match(sentence_pattern, part):
                sentences.append(current_sentence)
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence)
        
        return sentences if sentences else [text]
    
    def _split_by_words(self, text: str, max_chars: int) -> list[str]:
        """Split text by words when sentence splitting isn't enough"""
        words = text.split()
        parts = []
        current_part = ""
        
        for word in words:
            if len(current_part) + len(word) + 1 <= max_chars:
                if current_part:
                    current_part += " " + word
                else:
                    current_part = word
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word
                
                # If a single word is too long, truncate it
                if len(word) > max_chars:
                    parts.append(word[:max_chars-3] + "...")
                    current_part = ""
        
        if current_part:
            parts.append(current_part)
        
        return parts if parts else [text]
    
    def _is_good_break_point(self, prev_element: dict, next_element: dict) -> bool:
        """Determine if this is a good place to break a chunk for coherence"""
        if not prev_element or not next_element:
            return True
        
        prev_text = prev_element.get('text', '').strip()
        next_text = next_element.get('text', '').strip()
        prev_label = prev_element.get('label', '').lower()
        next_label = next_element.get('label', '').lower()
        
        # Good break points:
        # 1. After paragraphs that end with sentence punctuation
        if prev_text.endswith(('.', '!', '?')) and prev_label == 'paragraph':
            return True
        
        # 2. Between different content types
        if prev_label != next_label:
            return True
        
        # 3. Before headings/titles
        if next_label in ['heading', 'title']:
            return True
        
        # 4. After headings/titles (but not between related headings)
        if prev_label in ['heading', 'title'] and next_label not in ['heading', 'title']:
            return True
        
        # 5. Between list items (less ideal but acceptable)
        if prev_label == 'list' and next_label != 'list':
            return True
        
        # Bad break points:
        # 1. In the middle of a list
        if prev_label == 'list' and next_label == 'list':
            return False
        
        # 2. Between closely related paragraphs (those that don't end with sentence punctuation)
        if prev_label == 'paragraph' and next_label == 'paragraph' and not prev_text.endswith(('.', '!', '?')):
            return False
        
        # Default: it's an acceptable break point
        return True

    def execute(self, page_model: PageModel, **kwargs) -> ProcessingResult:
        # ... (the start of the method remains the same)
        # ... from self.logger.info(...) to elements_to_translate = ...
        # This is where the changes begin
        final_translated_blocks = []
        untranslated_elements = list(elements_to_translate)
        retries = 2 # Set the number of retries

        for i in range(retries + 1):
            if not untranslated_elements:
                break # Exit if everything is translated

            self.logger.info(f"Translation attempt {i+1}/{retries+1}. Translating {len(untranslated_elements)} elements.")
            # Prepare the XML-like string for the current batch of untranslated elements
            xml_string = "\n".join([f'<seg id="{el.id}">{el.text}</seg>' for el in untranslated_elements])
            # Get the translation
            translated_xml = self.gemini_service.translate_text(xml_string, self.target_language)
            # Parse the response
            translated_segments = self._parse_and_reconstruct_translation(translated_xml)
            # Process the results
            successfully_translated = []
            still_untranslated = []
            for element in untranslated_elements:
                if element.id in translated_segments and translated_segments[element.id]:
                    # Success: create a new translated block
                    translated_block = TextBlock(
                        text=translated_segments[element.id],
                        bbox=element.bbox,
                        block_type=element.block_type,
                        confidence=element.confidence,
                        page_num=element.page_num,
                        id=element.id
                    )
                    final_translated_blocks.append(translated_block)
                    successfully_translated.append(element)
                else:
                    # Failure: add to the list for the next retry
                    still_untranslated.append(element)
            if still_untranslated:
                self.logger.warning(f"{len(still_untranslated)} segments failed to translate on attempt {i+1}.")
                untranslated_elements = still_untranslated # This becomes the list for the next loop
            else:
                untranslated_elements = [] # All done
        # After the loop, handle any elements that permanently failed
        for element in untranslated_elements:
            self.logger.error(f"Permanently failed to translate segment ID {element.id} after {retries+1} attempts. Using original text.")
            final_translated_blocks.append(element) # Add the original english block
        # Add back the excluded elements
        all_blocks = final_translated_blocks + excluded_elements
        # Sort all blocks by their original position
        all_blocks.sort(key=lambda b: (b.page_num, b.bbox[1], b.bbox[0]))
        return ProcessingResult(data=all_blocks, strategy_used="direct_text_with_retry")


class MinimalGraphBuilder:
    """Build graph with one node per logical area for mixed content"""
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        self.logger.info("🔧 Minimal Graph Builder initialized")
    
    def build_area_level_graph(self, mapped_content: Dict[str, Any]) -> DocumentGraph:
        """Build graph with one node per logical area"""
        if not DOCUMENT_MODEL_AVAILABLE:
            self.logger.error("❌ Document model not available")
            return None
        
        graph = DocumentGraph()
        
        for area_id, area_data in mapped_content.items():
            if area_data['label'] in ['text', 'paragraph', 'title']:
                node_id = graph.add_node(
                    bbox=area_data['bbox'],
                    class_label=area_data['label'],
                    confidence=area_data['confidence'],
                    text=area_data['combined_text'],
                    semantic_role=self._infer_semantic_role(area_data['combined_text']),
                    extra={'area_id': area_id}
                )
            elif area_data['label'] in ['figure', 'table', 'image']:
                node_id = graph.add_node(
                    bbox=area_data['bbox'],
                    class_label=area_data['label'],
                    confidence=area_data['confidence'],
                    text=None,  # Visual content
                    semantic_role='visual_content',
                    extra={'area_id': area_id, 'image_blocks': area_data['image_blocks']}
                )
        
        self.logger.info(f"🏗️ Minimal graph built with {len(graph.nodes())} nodes")
        return graph
    
    def _infer_semantic_role(self, text: str) -> str:
        """Infer semantic role from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['chapter', 'section', 'part']):
            return 'section_header'
        elif any(word in text_lower for word in ['figure', 'table', 'diagram']):
            return 'caption'
        elif text_lower.startswith(('•', '-', '1.', '2.')):
            return 'list_item'
        elif len(text) < 100 and text.endswith('.'):
            return 'sentence'
        else:
            return 'paragraph'


class ComprehensiveGraphBuilder:
    """Build comprehensive graph for visual-heavy documents"""
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        self.logger.info("🔧 Comprehensive Graph Builder initialized")
    
    def build_comprehensive_graph(self, mapped_content: Dict[str, Any], 
                                text_blocks: List[Any], 
                                image_blocks: List[Any]) -> DocumentGraph:
        """Build comprehensive graph with detailed analysis"""
        if not DOCUMENT_MODEL_AVAILABLE:
            self.logger.error("❌ Document model not available")
            return None
        
        graph = DocumentGraph()
        
        # Add YOLO detections as nodes
        yolo_detections = []
        for area_id, area_data in mapped_content.items():
            if area_data['label'] in ['text', 'paragraph', 'title']:
                yolo_detections.append({
                    'label': area_data['label'],
                    'confidence': area_data['confidence'],
                    'bounding_box': list(area_data['bbox']),
                    'class_id': area_data.get('class_id', 0)
                })
        
        # Add YOLO detections to graph
        yolo_node_ids = add_yolo_detections_to_graph(graph, yolo_detections, page_num=1)
        
        # Convert text blocks to OCR format for graph integration
        ocr_text_regions = []
        for text_block in text_blocks:
            ocr_text_regions.append({
                'text': text_block.text,
                'confidence': text_block.confidence * 100,  # Convert to percentage
                'bbox': text_block.bbox
            })
        
        # Add OCR text regions to graph
        ocr_node_ids = add_ocr_text_regions_to_graph(graph, ocr_text_regions, page_num=1)
        
        self.logger.info(f"🏗️ Comprehensive graph built with {len(graph.nodes())} nodes")
        self.logger.info(f"   YOLO nodes: {len(yolo_node_ids)}")
        self.logger.info(f"   OCR nodes: {len(ocr_node_ids)}")
        
        return graph
    
    def process_comprehensive_graph(self, graph: DocumentGraph) -> Dict[str, Any]:
        """Process comprehensive graph with full analysis pipeline"""
        if not graph:
            return {'error': 'No graph available'}
        
        try:
            # Build association matrix
            association_matrix = build_association_matrix_from_graph(graph)
            
            # Populate graph with relationships
            populator = populate_document_graph(graph, association_matrix, threshold=0.3)
            
            # Refine graph
            refiner = refine_document_graph(graph, association_matrix)
            
            # Create document reconstructor
            reconstructor = DocumentReconstructor(graph)
            
            # Generate structured output
            json_output = reconstructor.reconstruct_document(output_format='json')
            
            processing_result = {
                'graph_nodes': len(graph.nodes()),
                'graph_edges': len(graph.edges()),
                'association_matrix': association_matrix.print_matrix_summary(),
                'reading_order': [node.class_label for node in populator.get_reading_order()],
                'structured_content': json_output,
                'refinement_summary': refiner.get_refinement_summary()
            }
            
            self.logger.info(f"✅ Comprehensive graph processing completed")
            self.logger.info(f"   Graph nodes: {processing_result['graph_nodes']}")
            self.logger.info(f"   Graph edges: {processing_result['graph_edges']}")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive graph processing failed: {e}")
            return {'error': str(e)}


class ProcessingStrategyExecutor:
    """Execute the appropriate processing strategy based on content type"""
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        self.direct_text_processor = DirectTextProcessor(gemini_service)
        self.table_processor = TableProcessor(gemini_service)  # Add table processor
        self.minimal_graph_builder = MinimalGraphBuilder(gemini_service)
        self.comprehensive_graph_builder = ComprehensiveGraphBuilder(gemini_service)
        
        # Initialize Digital Twin processor if available
        if DIGITAL_TWIN_AVAILABLE:
            self.digital_twin_processor = PyMuPDFYOLOProcessor()
            self.logger.info("✅ Digital Twin processor initialized")
        else:
            self.digital_twin_processor = None
            self.logger.warning("⚠️ Digital Twin processor not available")
        
        self.performance_stats = {
            'pure_text_fast': {'total_time': 0.0, 'count': 0},
            'coordinate_based_extraction': {'total_time': 0.0, 'count': 0},
            'direct_text': {'total_time': 0.0, 'count': 0},
            'minimal_graph': {'total_time': 0.0, 'count': 0},
            'comprehensive_graph': {'total_time': 0.0, 'count': 0},
            'digital_twin': {'total_time': 0.0, 'count': 0}
        }
        
        self.logger.info("🔧 Processing Strategy Executor initialized")
    
    async def cleanup(self):
        """Cleanup all processing components and services"""
        try:
            self.logger.debug("🧹 Cleaning up Processing Strategy Executor...")
            
            # Cleanup Gemini service if available
            if self.gemini_service and hasattr(self.gemini_service, 'cleanup'):
                await self.gemini_service.cleanup()
            
            # Clear references to prevent memory leaks
            self.digital_twin_processor = None
            self.direct_text_processor = None
            self.table_processor = None
            self.minimal_graph_builder = None
            self.comprehensive_graph_builder = None
            
            self.logger.debug("✅ Processing Strategy Executor cleanup completed")
            
        except Exception as e:
            self.logger.debug(f"⚠️ Cleanup warning (non-critical): {e}")
    
    async def execute_strategy(self, processing_result: Dict[str, Any], 
                             target_language: str = 'en') -> ProcessingResult:
        """Execute the appropriate processing strategy with enhanced error handling and validation"""
        self.logger.info(f"[DEBUG] Entering execute_strategy for page with keys: {list(processing_result.keys())}")
        
        # Enhanced validation with detailed error messages
        if not isinstance(processing_result, dict):
            raise ValueError(f"processing_result must be a dictionary, got {type(processing_result)}")
        
        strategy = processing_result.get('strategy', None)
        mapped_content = processing_result.get('mapped_content', None)
        
        if strategy is None:
            raise ValueError("Missing 'strategy' in processing_result")
        if mapped_content is None:
            raise ValueError("Missing 'mapped_content' in processing_result")
        if not hasattr(strategy, 'strategy'):
            raise ValueError(f"Invalid strategy object: {strategy}")
        
        start_time = time.time()
        strategy_name = strategy.strategy
        
        try:
            self.logger.info(f"🎯 Executing {strategy_name} strategy")
            
            # Route to appropriate processing method with explicit strategy validation
            if strategy_name == 'pure_text_fast':
                result = await self._process_pure_text_fast(processing_result, target_language)
            elif strategy_name == 'coordinate_based_extraction':
                result = await self._process_coordinate_based_extraction(processing_result, target_language)
            elif strategy_name == 'direct_text':
                result = await self._process_direct_text(mapped_content, target_language)
            elif strategy_name == 'minimal_graph':
                result = await self._process_minimal_graph(mapped_content, target_language)
            elif strategy_name == 'comprehensive_graph':
                result = await self._process_comprehensive_graph(processing_result, target_language)
            elif strategy_name == 'digital_twin':
                result = await self._process_digital_twin(processing_result, target_language)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Validate result
            if result is None:
                raise RuntimeError(f"Strategy {strategy_name} returned None")
            
            if not isinstance(result, ProcessingResult):
                raise RuntimeError(f"Strategy {strategy_name} returned invalid result type: {type(result)}")
            
            self.logger.info(f"✅ {strategy_name} strategy completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Strategy {strategy_name} execution failed: {e}", exc_info=True)
            
            return ProcessingResult(
                success=False,
                strategy=strategy_name,
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    def _ensure_dict_of_areas(self, mapped_content):
        # If it's already a dict of areas, return as is
        if isinstance(mapped_content, dict):
            return mapped_content
        # If it's a list of dicts (elements), convert to dict by index
        if isinstance(mapped_content, list) and all(isinstance(el, dict) for el in mapped_content):
            return {i: el for i, el in enumerate(mapped_content)}
        # If it's something else, return empty dict
        return {}

    async def _process_pure_text_fast(self, processing_result: Dict[str, Any], 
                                    target_language: str) -> ProcessingResult:
        """
        Orchestrates the pure text processing and translation, ensuring the final
        result is correctly structured and returned in a ProcessingResult object.

        This function serves as the master controller for the 'pure_text_fast' strategy.
        It converts raw data, invokes the semantic translation process, and critically,
        packages the final structured content into the ProcessingResult for the main
        pipeline to consume.
        """
        self.logger.info("🎯 Executing pure_text_fast strategy")
        start_time = time.time()
        
        # Step 1: Convert the initial raw dictionary from the upstream mapping process
        # into a clean list of text element objects.
        mapped_content = processing_result.get('mapped_content', {})
        page_content = self.direct_text_processor._convert_mapped_content_to_page_content(mapped_content)
        text_elements = []
        
        # Extract text elements with proper format for translation
        for element in page_content.content_elements:
            text_elements.append({
                'text': element.text,
                'bbox': element.bbox,
                'label': element.label,
                'confidence': element.confidence
            })
        
        if not text_elements:
            self.logger.warning("No text elements found to process for this page.")
            return ProcessingResult(success=True, content=[], processing_time=0)

        # Step 2: Invoke the high-quality translation method.
        # This method now performs semantic batching and returns a list of structured
        # dictionaries, where each dictionary is a translated paragraph.
        translated_blocks = await self.direct_text_processor.translate_direct_text(
            text_elements, target_language
        )
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ Pure text processing completed in {processing_time:.3f}s")
        self.logger.info(f"   Content elements processed: {len(text_elements)}")
        
        # Step 3: THE CRITICAL FIX - Package the final data correctly.
        # Return a ProcessingResult object where the 'content' field is explicitly
        # assigned the list of translated blocks. This is the data structure that
        # will be passed back to the main process from the worker.
        return ProcessingResult(
            success=True,
            strategy='pure_text_fast',
            processing_time=processing_time,
            content=translated_blocks,  # The translated data is now guaranteed to be here.
            statistics={
                'text_elements_processed': len(text_elements),
                'translated_blocks_created': len(translated_blocks) if isinstance(translated_blocks, list) else 0
            }
        )
    
    async def _process_coordinate_based_extraction(self, processing_result: Dict[str, Any], 
                                                 target_language: str) -> ProcessingResult:
        """Process mixed content using coordinate-based extraction with intelligent batching"""
        self.logger.info(f"[DEBUG] _process_coordinate_based_extraction received type: {type(processing_result)}")
        if not processing_result:
            self.logger.error("[ERROR] processing_result is empty or None in _process_coordinate_based_extraction")
            return ProcessingResult(
                success=False,
                strategy='coordinate_based_extraction',
                processing_time=0.0,
                content={},
                statistics={},
                error='processing_result is empty or None'
            )
        processing_result = self._ensure_dict_of_areas(processing_result)
        if not processing_result:
            self.logger.error(f"[ERROR] processing_result could not be converted to dict in _process_coordinate_based_extraction: {processing_result}")
            return ProcessingResult(
                success=False,
                strategy='coordinate_based_extraction',
                processing_time=0.0,
                content={},
                statistics={},
                error='processing_result could not be converted to dict'
            )
        start_time = time.time()
        try:
            mapped_content = processing_result.get('mapped_content', {})
            text_areas = []
            table_areas = []
            non_text_areas = []
            
            for area_id, area_data in mapped_content.items():
                # Check if this is a table area
                if hasattr(area_data, 'layout_info') and area_data.layout_info.label == 'table':
                    table_areas.append((area_id, area_data))
                elif hasattr(area_data, 'combined_text') and area_data.combined_text:
                    text_areas.append(area_data)
                else:
                    non_text_areas.append(area_data)
            text_areas.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
            translated_texts = []
            if self.gemini_service:
                # CRITICAL: Use the robust translate_direct_text method instead of fragile splitting
                # Convert text areas to the format expected by translate_direct_text
                text_elements = []
                for area in text_areas:
                    text_elements.append({
                        'text': area.combined_text,
                        'bbox': area.layout_info.bbox if hasattr(area, 'layout_info') else [0, 0, 0, 0],
                        'label': area.layout_info.label if hasattr(area, 'layout_info') else 'text'
                    })
                
                # Use the DirectTextProcessor's robust translation method
                direct_processor = DirectTextProcessor(self.gemini_service)
                translated_blocks = await direct_processor.translate_direct_text(text_elements, target_language)
                
                # Extract translated texts in the same order
                translated_texts = [block['text'] for block in translated_blocks]
                
                self.logger.info(f"✅ Used robust tag-based translation: {len(translated_texts)} texts translated")
            else:
                self.logger.warning("⚠️ Gemini service not available for coordinate-based extraction.")
                translated_texts = [area.combined_text for area in text_areas]
            
            # Process table areas using TableProcessor (Sub-task 2.3)
            processed_tables = []
            for area_id, table_area in table_areas:
                self.logger.info(f"📊 Processing table area: {area_id}")
                
                # Parse table structure from text blocks and coordinates
                table_structure = self.table_processor.parse_table_structure({
                    'text_blocks': table_area.text_blocks if hasattr(table_area, 'text_blocks') else [],
                    'layout_info': table_area.layout_info if hasattr(table_area, 'layout_info') else None
                })
                
                if table_structure.get('error'):
                    self.logger.warning(f"❌ Table parsing failed for {area_id}: {table_structure['error']}")
                    # Add as fallback text content
                    processed_tables.append({
                        'type': 'table',
                        'area_id': area_id,
                        'content': table_area.combined_text if hasattr(table_area, 'combined_text') else 'Table content unavailable',
                        'error': table_structure['error'],
                        'layout_info': table_area.layout_info if hasattr(table_area, 'layout_info') else {}
                    })
                    continue
                
                # Translate the table structure
                translation_result = await self.table_processor.translate_table(table_structure, target_language)
                
                if translation_result.get('error'):
                    self.logger.warning(f"❌ Table translation failed for {area_id}: {translation_result['error']}")
                    # Use original table structure
                    translated_rows = table_structure.get('rows', [])
                else:
                    translated_rows = translation_result.get('translated_rows', [])
                
                # Create TableModel-compatible structure
                processed_table = {
                    'type': 'table',
                    'area_id': area_id,
                    'content': translated_rows,  # List[List[str]] as required by TableModel
                    'header_row': table_structure.get('header_row'),
                    'caption': None,  # Could be enhanced to detect captions
                    'num_rows': len(translated_rows),
                    'num_cols': len(translated_rows[0]) if translated_rows else 0,
                    'layout_info': table_area.layout_info if hasattr(table_area, 'layout_info') else {},
                    'original_markdown': translation_result.get('original_markdown', ''),
                    'translated_markdown': translation_result.get('translated_markdown', '')
                }
                
                processed_tables.append(processed_table)
                self.logger.info(f"✅ Table {area_id} processed: {processed_table['num_rows']}x{processed_table['num_cols']}")
            
            if processed_tables:
                self.logger.info(f"📊 Processed {len(processed_tables)} tables successfully")
            final_content = []
            for i, area in enumerate(text_areas):
                final_content.append({
                    'type': 'text',
                    'original_text': area.combined_text,
                    'translated_text': translated_texts[i] if i < len(translated_texts) else area.combined_text,
                    'layout_info': area.layout_info if hasattr(area, 'layout_info') else {'bbox': getattr(area, 'bbox', [0, 0, 0, 0])}
                })
            # Add processed tables to final content
            for table in processed_tables:
                final_content.append(table)
            
            for area in non_text_areas:
                final_content.append({
                    'type': 'visual_element',
                    'layout_info': area.layout_info if hasattr(area, 'layout_info') else {'bbox': getattr(area, 'bbox', [0, 0, 0, 0])},
                    'image_blocks': getattr(area, 'image_blocks', [])
                })
            # Sort by layout position (y-coordinate first, then x-coordinate)
            def get_sort_key(item):
                layout_info = item.get('layout_info', {})
                if hasattr(layout_info, 'bbox'):
                    bbox = layout_info.bbox
                elif isinstance(layout_info, dict):
                    bbox = layout_info.get('bbox', [0, 0, 0, 0])
                else:
                    bbox = [0, 0, 0, 0]
                return (bbox[1], bbox[0])  # y, x
            
            final_content.sort(key=get_sort_key)
            processing_time = time.time() - start_time
            self.performance_stats['coordinate_based_extraction']['total_time'] += processing_time
            self.performance_stats['coordinate_based_extraction']['count'] += 1
            self.logger.info(f"🎯 Coordinate-based extraction completed in {processing_time:.3f}s")
            self.logger.info(f"   Total areas processed: {len(mapped_content)}")
            self.logger.info(f"   Text areas: {len(text_areas)}")
            self.logger.info(f"   Table areas: {len(table_areas)}")
            self.logger.info(f"   Non-text areas: {len(non_text_areas)}")
            
            return ProcessingResult(
                success=True,
                strategy='coordinate_based_extraction',
                processing_time=processing_time,
                content={'final_content': final_content},
                statistics={
                    'text_areas': len(text_areas),
                    'table_areas': len(table_areas),
                    'non_text_areas': len(non_text_areas),
                    'total_areas': len(mapped_content)
                }
            )
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Coordinate-based extraction failed: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                strategy='coordinate_based_extraction',
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    # REMOVED: _create_text_batches - now uses DirectTextProcessor's robust batching
    
    async def _process_direct_text(self, mapped_content: Dict[str, Any], 
                                 target_language: str) -> ProcessingResult:
        """DEPRECATED: Process text directly (now part of _process_pure_text_fast)"""
        self.logger.warning("⚠️ _process_direct_text is deprecated and should not be called.")
        # This method is now effectively replaced by _process_pure_text_fast
        # and its direct call to the translator. 
        # For simplicity, we can just route to the new implementation.
        return await self._process_pure_text_fast({'mapped_content': mapped_content}, target_language)
    
    async def _process_minimal_graph(self, mapped_content: Dict[str, Any], 
                                   target_language: str) -> ProcessingResult:
        """Process with minimal graph (area-level nodes)"""
        start_time = time.time()
        
        # Convert MappedContent objects to dict format if needed
        processed_mapped_content = {}
        for area_id, area_data in mapped_content.items():
            if hasattr(area_data, 'layout_info'):
                # Convert MappedContent object to dict
                processed_mapped_content[area_id] = {
                    'layout_info': {
                        'label': area_data.layout_info.label,
                        'bbox': area_data.layout_info.bbox,
                        'confidence': area_data.layout_info.confidence
                    },
                    'combined_text': area_data.combined_text,
                    'text_blocks': area_data.text_blocks,
                    'image_blocks': area_data.image_blocks
                }
            else:
                # Already in dict format
                processed_mapped_content[area_id] = area_data
        
        # Build minimal graph
        graph = self.minimal_graph_builder.build_area_level_graph(processed_mapped_content)
        
        if not graph:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                strategy='minimal_graph',
                processing_time=processing_time,
                content={},
                statistics={},
                error='Failed to build minimal graph'
            )
        
        # Process graph (simplified version)
        graph_result = {
            'graph_nodes': len(graph.nodes()),
            'graph_edges': len(graph.edges()),
            'content_areas': len(processed_mapped_content),
            'graph_type': 'minimal'
        }
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ Minimal graph processing completed in {processing_time:.3f}s")
        
        return ProcessingResult(
            success=True,
            strategy='minimal_graph',
            processing_time=processing_time,
            content=graph_result,
            statistics={
                'graph_nodes': graph_result['graph_nodes'],
                'graph_edges': graph_result['graph_edges'],
                'content_areas': graph_result['content_areas'],
                'graph_overhead': processing_time * 0.3  # Estimated graph overhead
            }
        )
    
    async def _process_comprehensive_graph(self, processing_result: Dict[str, Any], 
                                         target_language: str) -> ProcessingResult:
        """Process with comprehensive graph analysis"""
        start_time = time.time()
        
        # Extract additional data needed for comprehensive processing
        text_blocks = processing_result.get('text_blocks', [])
        image_blocks = processing_result.get('image_blocks', [])
        mapped_content = processing_result['mapped_content']
        
        # Build comprehensive graph
        graph = self.comprehensive_graph_builder.build_comprehensive_graph(
            mapped_content, text_blocks, image_blocks
        )
        
        if not graph:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                strategy='comprehensive_graph',
                processing_time=processing_time,
                content={},
                statistics={},
                error='Failed to build comprehensive graph'
            )
        
        # Process comprehensive graph
        graph_result = self.comprehensive_graph_builder.process_comprehensive_graph(graph)
        
        processing_time = time.time() - start_time
        
        if 'error' in graph_result:
            return ProcessingResult(
                success=False,
                strategy='comprehensive_graph',
                processing_time=processing_time,
                content={},
                statistics={},
                error=graph_result['error']
            )
        
        self.logger.info(f"✅ Comprehensive graph processing completed in {processing_time:.3f}s")
        
        return ProcessingResult(
            success=True,
            strategy='comprehensive_graph',
            processing_time=processing_time,
            content=graph_result,
            statistics={
                'graph_nodes': graph_result['graph_nodes'],
                'graph_edges': graph_result['graph_edges'],
                'graph_overhead': processing_time * 0.7,  # Higher overhead for comprehensive processing
                'association_matrix_built': True,
                'graph_refined': True
            }
        )
    
    def get_strategy_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between strategies"""
        return {
            'direct_text': {
                'speed': 'maximum',
                'memory_usage': 'minimal',
                'accuracy': 'high_for_text',
                'graph_overhead': 0.0,
                'best_for': 'Pure text documents'
            },
            'minimal_graph': {
                'speed': 'high',
                'memory_usage': 'low',
                'accuracy': 'balanced',
                'graph_overhead': 'low',
                'best_for': 'Mixed content documents'
            },
            'comprehensive_graph': {
                'speed': 'moderate',
                'memory_usage': 'high',
                'accuracy': 'maximum',
                'graph_overhead': 'high',
                'best_for': 'Visual-heavy documents'
            }
        }

    async def execute_strategy_digital_twin(self, pdf_path: str, output_dir: str, 
                                          target_language: str = 'en') -> ProcessingResult:
        """
        Execute complete Digital Twin document processing strategy.
        
        This is the master method for Digital Twin processing that:
        1. Processes entire PDF using enhanced PyMuPDF-YOLO processor
        2. Creates complete Digital Twin representation with images and TOC
        3. Applies translation to all translatable blocks
        4. Returns processed document ready for reconstruction
        """
        if not DIGITAL_TWIN_AVAILABLE or not self.digital_twin_processor:
            raise RuntimeError("Digital Twin processing not available")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting Digital Twin strategy execution for: {pdf_path}")
            
            # Step 1: Process entire document with Digital Twin processor
            self.logger.info("📖 Processing document with Digital Twin model...")
            digital_twin_doc = await self.digital_twin_processor.process_document_digital_twin(
                pdf_path, output_dir
            )
            
            if not digital_twin_doc or digital_twin_doc.total_pages == 0:
                raise RuntimeError("Digital Twin document processing failed or returned empty document")
            
            # Step 2: Apply translation to all translatable blocks
            if self.gemini_service and target_language != 'en':
                self.logger.info(f"🌐 Translating Digital Twin document to {target_language}...")
                translated_doc = await self._translate_digital_twin_document(digital_twin_doc, target_language)
            else:
                self.logger.info("⏭️ Skipping translation (no service or target is English)")
                translated_doc = digital_twin_doc
            
            # Step 3: Package results
            processing_time = time.time() - start_time
            self.performance_stats['digital_twin']['total_time'] += processing_time
            self.performance_stats['digital_twin']['count'] += 1
            
            # Create comprehensive statistics
            doc_stats = translated_doc.get_statistics()
            
            self.logger.info(f"✅ Digital Twin strategy completed in {processing_time:.3f}s")
            self.logger.info(f"   📊 Final Statistics:")
            for key, value in doc_stats.items():
                self.logger.info(f"      - {key}: {value}")
            
            return ProcessingResult(
                success=True,
                strategy='digital_twin',
                processing_time=processing_time,
                content={
                    'digital_twin_document': translated_doc,
                    'document_statistics': doc_stats,
                    'processing_metadata': {
                        'pdf_path': pdf_path,
                        'output_dir': output_dir,
                        'target_language': target_language,
                        'translation_applied': target_language != 'en' and self.gemini_service is not None
                    }
                },
                statistics={
                    'total_pages': doc_stats['total_pages'],
                    'total_text_blocks': doc_stats['total_text_blocks'],
                    'total_image_blocks': doc_stats['total_image_blocks'],
                    'total_tables': doc_stats['total_tables'],
                    'total_toc_entries': doc_stats['total_toc_entries'],
                    'translated_blocks': doc_stats['translated_blocks'],
                    'processing_time': processing_time,
                    'extraction_method': doc_stats['extraction_method']
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Digital Twin strategy execution failed: {e}", exc_info=True)
            
            return ProcessingResult(
                success=False,
                strategy='digital_twin',
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    async def _translate_digital_twin_document(self, digital_twin_doc: DocumentModel, 
                                             target_language: str) -> DocumentModel:
        """
        Apply translation to all translatable blocks in a Digital Twin document.
        
        This method integrates the existing robust translation logic with
        the new Digital Twin model structure.
        """
        try:
            # Get all translatable text blocks
            translatable_blocks = digital_twin_doc.get_translatable_blocks()
            
            if not translatable_blocks:
                self.logger.warning("No translatable blocks found in Digital Twin document")
                return digital_twin_doc
            
            self.logger.info(f"📝 Translating {len(translatable_blocks)} text blocks...")
            
            # Convert Digital Twin blocks to format expected by direct text processor
            text_elements = []
            for block in translatable_blocks:
                text_elements.append({
                    'text': block.original_text,
                    'bbox': block.bbox,
                    'label': block.block_type.value,
                    'confidence': block.confidence or 1.0,
                    'block_id': block.block_id,
                    'structural_role': block.structural_role.value
                })
            
            # Apply robust translation using existing DirectTextProcessor
            translated_blocks = await self.direct_text_processor.translate_direct_text(
                text_elements, target_language
            )
            
            # Map translations back to Digital Twin blocks
            translated_count = 0
            for i, translated_block in enumerate(translated_blocks):
                if i < len(translatable_blocks):
                    original_block = translatable_blocks[i]
                    original_block.translated_text = translated_block.get('text', original_block.original_text)
                    
                    # Add translation metadata
                    original_block.processing_notes.append(
                        f"Translated to {target_language} using robust tag-based method"
                    )
                    
                    translated_count += 1
            
            # Translate TOC entries
            if digital_twin_doc.toc_entries:
                await self._translate_toc_entries(digital_twin_doc.toc_entries, target_language)
            
            # Update document metadata
            digital_twin_doc.target_language = target_language
            digital_twin_doc.translation_status = 'completed'
            
            self.logger.info(f"✅ Translation completed: {translated_count} blocks translated")
            
            return digital_twin_doc
            
        except Exception as e:
            self.logger.error(f"❌ Digital Twin document translation failed: {e}")
            # Return original document with error noted
            digital_twin_doc.translation_status = 'failed'
            digital_twin_doc.document_metadata['translation_error'] = str(e)
            return digital_twin_doc
    
    async def _translate_toc_entries(self, toc_entries: List, target_language: str) -> None:
        """Translate TOC entries while preserving structure"""
        try:
            if not toc_entries:
                return
            
            # Extract titles for translation
            titles_to_translate = [entry.title for entry in toc_entries]
            
            # Create batch translation request
            combined_text = '\n'.join([f'<seg id="{i}">{title}</seg>' for i, title in enumerate(titles_to_translate)])
            
            # Translate using Gemini service
            translated_response = await self.gemini_service.translate_text(combined_text, target_language)
            
            # Parse translations
            translated_segments = self.direct_text_processor._parse_and_reconstruct_translation(translated_response)
            
            # Map back to TOC entries
            for i, entry in enumerate(toc_entries):
                if str(i) in translated_segments:
                    entry.translated_title = translated_segments[str(i)]
                else:
                    entry.translated_title = entry.title  # Fallback to original
            
            self.logger.info(f"✅ Translated {len(toc_entries)} TOC entries")
            
        except Exception as e:
            self.logger.error(f"❌ TOC translation failed: {e}")
            # Ensure all entries have fallback translations
            for entry in toc_entries:
                if not entry.translated_title:
                    entry.translated_title = entry.title
    
    async def _process_digital_twin(self, processing_result: Dict[str, Any], 
                                  target_language: str) -> ProcessingResult:
        """
        Process Digital Twin data when it's provided as part of the standard strategy pipeline.
        
        This method bridges the gap between the old pipeline format and the new Digital Twin approach.
        """
        try:
            # Extract Digital Twin document if available
            digital_twin_doc = processing_result.get('digital_twin_document')
            
            if not digital_twin_doc:
                raise ValueError("No Digital Twin document found in processing result")
            
            # Apply translation if needed
            if self.gemini_service and target_language != 'en':
                translated_doc = await self._translate_digital_twin_document(digital_twin_doc, target_language)
            else:
                translated_doc = digital_twin_doc
            
            # Return in standard ProcessingResult format
            doc_stats = translated_doc.get_statistics()
            
            return ProcessingResult(
                success=True,
                strategy='digital_twin',
                processing_time=processing_result.get('processing_time', 0.0),
                content={'digital_twin_document': translated_doc},
                statistics=doc_stats
            )
            
        except Exception as e:
            self.logger.error(f"❌ Digital Twin processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                strategy='digital_twin',
                processing_time=0.0,
                content={},
                statistics={},
                error=str(e)
            )

def _dict_to_layout_area(d):
    if isinstance(d, LayoutArea):
        return d
    return LayoutArea(
        label=d['label'],
        bbox=tuple(d['bbox']),
        confidence=d.get('confidence', 1.0),
        area_id=d.get('area_id', ''),
        class_id=d.get('class_id', 0)
    )

def _dict_to_mapped_content(d):
    # Accepts either a dict or a SimpleNamespace/MappedContent
    if hasattr(d, 'layout_info'):
        return d
    layout_info = d.get('layout_info')
    if isinstance(layout_info, dict):
        layout_info = _dict_to_layout_area(layout_info)
    return SimpleNamespace(
        layout_info=layout_info,
        combined_text=d.get('combined_text', ''),
        text_blocks=d.get('text_blocks', []),
        image_blocks=d.get('image_blocks', []),
    )

# REMOVED: process_page_worker function has been eliminated as part of the architectural refactoring.
# This function was the source of the "rogue worker" issue that caused mojibake, hyphenation failures,
# and data loss. The system now uses the architecturally sound PyMuPDFContentExtractor and
# ProcessingStrategyExecutor as the single source of truth for content extraction and translation. 