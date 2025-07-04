#!/usr/bin/env python3
"""
PyMuPDF-YOLO Content Mapping Foundation

This module implements the core PyMuPDF-YOLO integration for high-performance
document processing with intelligent content mapping and processing routing.

Key Features:
- PyMuPDF content extraction with precise coordinates
- YOLO layout analysis with 0.15 confidence threshold
- Content-to-layout mapping with overlap detection
- Intelligent processing router based on content type
- Direct text processing for pure text documents
- Minimal graph processing for mixed content
- Comprehensive graph processing for visual-heavy documents
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Import existing services
try:
    from yolov8_service import YOLOv8Service
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 service not available")

try:
    from document_model import DocumentGraph, add_yolo_detections_to_graph, add_ocr_text_regions_to_graph
    DOCUMENT_MODEL_AVAILABLE = True
except ImportError:
    DOCUMENT_MODEL_AVAILABLE = False
    logger.warning("Document model not available")

# --- Import the Centralized Models ---
# All data structures are now imported from the single source of truth.
from models import PageModel, ElementModel, BoundingBox, ElementType

# Import the new Digital Twin model for enhanced functionality
from digital_twin_model import (
    DocumentModel, PageModel as DigitalTwinPageModel, 
    TextBlock as DigitalTwinTextBlock, ImageBlock as DigitalTwinImageBlock, 
    TableBlock as DigitalTwinTableBlock, TOCEntry, BlockType, StructuralRole,
    create_text_block, create_image_block, create_table_block
)

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type classification for processing strategy selection"""
    PURE_TEXT = "pure_text"
    MIXED_CONTENT = "mixed_content"
    VISUAL_HEAVY = "visual_heavy"

@dataclass
class TextBlock:
    """Represents a text block extracted by PyMuPDF"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_family: str
    confidence: float = 1.0
    block_type: str = 'text'

@dataclass
class ImageBlock:
    """Represents an image block extracted by PyMuPDF"""
    image_index: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_type: str = 'image'

@dataclass
class LayoutArea:
    """Represents a layout area detected by YOLO"""
    label: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float
    area_id: str
    class_id: int

@dataclass
class MappedContent:
    """Represents content mapped to a layout area"""
    layout_info: LayoutArea
    text_blocks: List[TextBlock]
    image_blocks: List[ImageBlock]
    combined_text: str
    text_density: float = 0.0
    visual_density: float = 0.0

@dataclass
class ProcessingStrategy:
    """Processing strategy configuration"""
    strategy: str
    description: str
    skip_graph: bool
    optimization_level: str
    confidence_threshold: float = 0.15

class PyMuPDFContentExtractor:
    """Extract high-quality text blocks with precise coordinates using PyMuPDF"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 PyMuPDF Content Extractor initialized")
    
    def extract_text_blocks(self, page: fitz.Page) -> List[TextBlock]:
        """Extract all text blocks with coordinates using PyMuPDF"""
        text_blocks = []
        
        try:
            # Get text dictionary with detailed positioning
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_text = self._extract_text_from_block(block)
                    
                    if block_text.strip():
                        # Get dominant font information
                        font_size = self._get_dominant_font_size(block)
                        font_family = self._get_dominant_font(block)
                        
                        text_block = TextBlock(
                            text=block_text.strip(),
                            bbox=tuple(block.get("bbox", [0, 0, 0, 0])),
                            font_size=font_size,
                            font_family=font_family,
                            confidence=1.0,  # PyMuPDF extraction is considered perfect
                            block_type='text'
                        )
                        
                        text_blocks.append(text_block)
            
            # CRITICAL: Apply hyphenation reconstruction at page level on ALL text blocks
            text_blocks = self._apply_page_level_hyphenation_reconstruction(text_blocks)
            
            self.logger.info(f"📄 Extracted {len(text_blocks)} text blocks from page (with hyphenation reconstruction)")
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting text blocks: {e}")
            return []
    
    def _apply_page_level_hyphenation_reconstruction(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Apply the directive's hyphenation reconstruction across ALL text blocks for the entire page"""
        if not text_blocks:
            return []
        
        # Convert TextBlocks to the format expected by directive's function
        blocks_for_reconstruction = [{'text': tb.text} for tb in text_blocks]
        
        # Apply the directive's exact hyphenation reconstruction
        reconstructed_blocks = self._reconstruct_hyphenated_text(blocks_for_reconstruction)
        
        # Rebuild TextBlock objects with reconstructed text
        result_blocks = []
        for i, reconstructed_block in enumerate(reconstructed_blocks):
            if i < len(text_blocks):
                # Use original block metadata but with reconstructed text
                original_block = text_blocks[i]
                result_blocks.append(TextBlock(
                    text=reconstructed_block['text'],
                    bbox=original_block.bbox,
                    font_size=original_block.font_size,
                    font_family=original_block.font_family,
                    confidence=original_block.confidence,
                    block_type=original_block.block_type
                ))
        
        self.logger.info(f"🔧 Applied page-level hyphenation reconstruction: {len(text_blocks)} → {len(result_blocks)} blocks")
        return result_blocks
    
    def extract_images(self, page: fitz.Page) -> List[ImageBlock]:
        """Extract native images with coordinates (patched for full image list)"""
        image_blocks = []
        try:
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    bbox = None
                    # Try to get image bbox using get_image_rects
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = (img_rects[0].x0, img_rects[0].y0, img_rects[0].x1, img_rects[0].y1)
                    else:
                        # Fallback: use (0,0,0,0) if bbox not found
                        bbox = (0, 0, 0, 0)
                    image_block = ImageBlock(
                        image_index=img_index,
                        bbox=tuple(bbox),
                        block_type='image'
                    )
                    image_blocks.append(image_block)
                except Exception as e:
                    self.logger.warning(f"Could not extract bbox for image {img_index}: {e}")
                    continue
            self.logger.info(f"🖼️ Extracted {len(image_blocks)} image blocks from page (patched)")
            return image_blocks
        except Exception as e:
            self.logger.error(f"❌ Error extracting images (patched): {e}")
            return []
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract raw text from a PyMuPDF block (hyphenation will be handled at page level)"""
        try:
            # Extract all lines as simple text - no hyphenation processing here
            lines = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                if line_text.strip():
                    lines.append(line_text.strip())
            
            # Return raw text - hyphenation reconstruction happens at page level
            return "\n".join(lines).strip()
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from block: {e}")
            return ""
    
    def _reconstruct_hyphenated_text(self, blocks: list) -> list:
        """
        Intelligently reconstructs paragraphs from raw text blocks, correcting
        for words that are hyphenated across line breaks.
        """
        if not blocks:
            return []

        reconstructed_texts = []
        # Start with the text from the first block.
        current_text = blocks[0].get('text', '')

        # Iterate up to the second-to-last block to allow look-ahead.
        for i in range(len(blocks) - 1):
            cleaned_text = current_text.strip()
            # Check if the current, cleaned text ends with a hyphen.
            if cleaned_text.endswith('-'):
                # Look ahead to the next block's text.
                next_block_text = blocks[i+1].get('text', '')
                # Merge: remove the hyphen and append the next block's text.
                current_text = cleaned_text[:-1] + next_block_text
            else:
                # No hyphen found. Finalize the current text block.
                # Replace internal newlines with spaces and strip whitespace.
                reconstructed_texts.append(current_text.replace('\n', ' ').strip())
                # Start the next block.
                current_text = blocks[i+1].get('text', '')

        # Append the final text block after the loop finishes.
        reconstructed_texts.append(current_text.replace('\n', ' ').strip())

        # Return a list of dictionaries, ensuring no empty text elements are included.
        return [{'text': text} for text in reconstructed_texts if text]
    
    def _get_dominant_font_size(self, block: Dict) -> float:
        """Get the dominant font size in a block"""
        font_sizes = []
        
        try:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span.get("size", 12.0))
            
            if font_sizes:
                return max(set(font_sizes), key=font_sizes.count)
            return 12.0
            
        except Exception:
            return 12.0
    
    def _get_dominant_font(self, block: Dict) -> str:
        """Get the dominant font family in a block"""
        font_families = []
        
        try:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_families.append(span.get("font", "Arial"))
            
            if font_families:
                return max(set(font_families), key=font_families.count)
            return "Arial"
            
        except Exception:
            return "Arial"

class YOLOLayoutAnalyzer:
    """Detect logical areas with ultra-low confidence threshold (0.15)"""
    
    def __init__(self):
        self.config = {
            'confidence_threshold': 0.15,  # Ultra-low as requested
            'iou_threshold': 0.4,
            'max_detections': 300,  # Increased for comprehensive coverage
            'supported_classes': [
                'text', 'title', 'paragraph', 'list', 'table', 
                'figure', 'caption', 'quote', 'footnote', 'equation'
            ]
        }
        
        self.yolo_service = None
        if YOLO_AVAILABLE:
            self.yolo_service = YOLOv8Service()
            # Override confidence threshold
            if hasattr(self.yolo_service, 'conf_thres'):
                self.yolo_service.conf_thres = 0.15
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔧 YOLO Layout Analyzer initialized with {self.config['confidence_threshold']} confidence threshold")
    
    def analyze_layout(self, page_image: Image.Image) -> List[LayoutArea]:
        """Analyze page layout using YOLO with 0.15 confidence"""
        if not self.yolo_service:
            self.logger.warning("⚠️ YOLO service not available")
            return []
        
        try:
            detections = self.yolo_service.detect(page_image)
            
            layout_areas = []
            for detection in detections:
                layout_area = LayoutArea(
                    label=detection['label'],
                    bbox=tuple(detection['bounding_box']),
                    confidence=detection['confidence'],
                    area_id=f"{detection['label']}_{len(layout_areas)}",
                    class_id=detection.get('class_id', 0)
                )
                
                layout_areas.append(layout_area)
            
            self.logger.info(f"🎯 Detected {len(layout_areas)} layout areas with YOLO")
            return layout_areas
            
        except Exception as e:
            self.logger.error(f"❌ YOLO layout analysis failed: {e}")
            return []

class ContentLayoutMapper:
    """Map PyMuPDF content blocks to YOLO-detected logical areas"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 Content Layout Mapper initialized")
    
    def map_content_to_layout(self, text_blocks: List[TextBlock], 
                            image_blocks: List[ImageBlock], 
                            layout_areas: List[LayoutArea]) -> Dict[str, MappedContent]:
        """Map PyMuPDF content to YOLO layout areas"""
        mapped_content = {}
        
        for area in layout_areas:
            area_id = area.area_id
            mapped_content[area_id] = MappedContent(
                layout_info=area,
                text_blocks=[],
                image_blocks=[],
                combined_text='',
                text_density=0.0,
                visual_density=0.0
            )
            
            # Map text blocks to this area
            for text_block in text_blocks:
                if self._bbox_overlaps(text_block.bbox, area.bbox):
                    mapped_content[area_id].text_blocks.append(text_block)
                    mapped_content[area_id].combined_text += text_block.text + ' '
            
            # Map image blocks to this area
            for image_block in image_blocks:
                if self._bbox_overlaps(image_block.bbox, area.bbox):
                    mapped_content[area_id].image_blocks.append(image_block)
            
            # Clean up combined text
            mapped_content[area_id].combined_text = mapped_content[area_id].combined_text.strip()
            
            # Calculate densities
            mapped_content[area_id].text_density = self._calculate_text_density(
                mapped_content[area_id].combined_text, area.bbox
            )
            mapped_content[area_id].visual_density = self._calculate_visual_density(
                mapped_content[area_id].image_blocks, area.bbox
            )
        
        self.logger.info(f"🗺️ Mapped content to {len(mapped_content)} layout areas")
        return mapped_content
    
    def _bbox_overlaps(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float], 
                      threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Use intersection over union (IoU) or intersection over smaller area
        smaller_area = min(area1, area2)
        overlap_ratio = intersection_area / smaller_area
        
        return overlap_ratio >= threshold
    
    def _calculate_text_density(self, text: str, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate text density in a bounding box"""
        if not text.strip():
            return 0.0
        
        # Calculate area
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area <= 0:
            return 0.0
        
        # Text density = character count / area
        return len(text) / area
    
    def _calculate_visual_density(self, image_blocks: List[ImageBlock], 
                                bbox: Tuple[float, float, float, float]) -> float:
        """Calculate visual density in a bounding box"""
        if not image_blocks:
            return 0.0
        
        # Calculate total image area
        total_image_area = 0.0
        for img_block in image_blocks:
            x1, y1, x2, y2 = img_block.bbox
            total_image_area += (x2 - x1) * (y2 - y1)
        
        # Calculate bounding box area
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area <= 0:
            return 0.0
        
        # Visual density = image area / bounding box area
        return total_image_area / bbox_area

class ContentTypeClassifier:
    """Determine optimal processing strategy based on mapped content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 Content Type Classifier initialized")
    
    def classify_mapped_content(self, mapped_content: Dict[str, MappedContent]) -> ContentType:
        """Classify content type based on YOLO-PyMuPDF mapping"""
        if not mapped_content:
            return ContentType.PURE_TEXT
        
        total_areas = len(mapped_content)
        text_areas = sum(1 for area in mapped_content.values() 
                        if area.layout_info.label in ['text', 'paragraph', 'title'])
        visual_areas = sum(1 for area in mapped_content.values() 
                          if area.layout_info.label in ['figure', 'table', 'image'])
        
        # Calculate text density
        total_text_length = sum(len(area.combined_text) 
                               for area in mapped_content.values())
        
        # Calculate average text and visual densities
        avg_text_density = np.mean([area.text_density for area in mapped_content.values()])
        avg_visual_density = np.mean([area.visual_density for area in mapped_content.values()])
        
        # Classification logic
        if text_areas >= total_areas * 0.8 and total_text_length > 500:
            self.logger.info(f"📝 Classified as PURE_TEXT (text_areas: {text_areas}/{total_areas}, text_length: {total_text_length})")
            return ContentType.PURE_TEXT
        elif visual_areas >= total_areas * 0.5 or avg_visual_density > 0.3:
            self.logger.info(f"🖼️ Classified as VISUAL_HEAVY (visual_areas: {visual_areas}/{total_areas}, visual_density: {avg_visual_density:.3f})")
            return ContentType.VISUAL_HEAVY
        else:
            self.logger.info(f"🔄 Classified as MIXED_CONTENT (text_areas: {text_areas}/{total_areas}, visual_areas: {visual_areas}/{total_areas})")
            return ContentType.MIXED_CONTENT
    
    def get_processing_strategy(self, content_type: ContentType, 
                              mapped_content: Dict[str, MappedContent]) -> ProcessingStrategy:
        """Intelligent processing strategy based on content type - implements user's strategic vision"""
        
        if content_type == ContentType.PURE_TEXT:
            # Pure text: Fast PyMuPDF-only processing (no YOLO overhead)
            self.logger.info("📝 Pure text detected: Using fast PyMuPDF-only processing")
            return ProcessingStrategy(
                strategy='pure_text_fast',
                description='Pure text: PyMuPDF-only extraction with format preservation',
                skip_graph=True,
                optimization_level='maximum',
                confidence_threshold=0.15
            )
        else:
            # Mixed/Visual content: Coordinate-based PyMuPDF+YOLO processing
            self.logger.info(f"🎯 {content_type.value} detected: Using coordinate-based PyMuPDF+YOLO processing")
            return ProcessingStrategy(
                strategy='coordinate_based_extraction',
                description='Mixed content: YOLO detection + PyMuPDF coordinate-based extraction',
                skip_graph=True,  # Still no graph logic, but coordinate-based processing
                optimization_level='balanced',
                confidence_threshold=0.15
            )

class PyMuPDFYOLOProcessor:
    """Main PyMuPDF-YOLO integration processor"""
    
    def __init__(self):
        self.content_extractor = PyMuPDFContentExtractor()
        self.layout_analyzer = YOLOLayoutAnalyzer()
        self.content_mapper = ContentLayoutMapper()
        self.classifier = ContentTypeClassifier()
        
        # Configuration for layout analysis refinement (Directive III)
        self.yolo_pruning_threshold = 0.2  # Confidence threshold for pruning
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 PyMuPDF-YOLO Processor initialized")
    
    def _prune_and_merge_layout_areas(self, layout_areas: List[LayoutArea]) -> List[LayoutArea]:
        """
        Prune and merge layout areas to reduce noise and improve accuracy.
        
        Directive III Implementation:
        1. Pruning: Remove areas with confidence below threshold
        2. Merging: Merge contained areas unless they are captions within figures/tables
        """
        if not layout_areas:
            return []
        
        # Step 1: Prune low-confidence detections
        pruned_areas = []
        for area in layout_areas:
            if area.confidence >= self.yolo_pruning_threshold:
                pruned_areas.append(area)
            else:
                self.logger.debug(f"Pruned low-confidence area: {area.label} (conf: {area.confidence:.3f})")
        
        self.logger.info(f"🔧 Pruned {len(layout_areas) - len(pruned_areas)} low-confidence areas")
        
        # Step 2: Merge contained areas of same type (with caption exception)
        merged_areas = []
        areas_to_skip = set()
        
        for i, area in enumerate(pruned_areas):
            if i in areas_to_skip:
                continue
                
            # Check if this area is contained within another area
            is_contained = False
            containing_area = None
            
            for j, other_area in enumerate(pruned_areas):
                if i != j and j not in areas_to_skip:
                    if self._is_fully_contained(area.bbox, other_area.bbox):
                        # Special exception: don't merge captions within figures/tables
                        if area.label == 'caption' and other_area.label in ['figure', 'table']:
                            is_contained = False
                            break
                        # Only merge if same type
                        elif area.label == other_area.label:
                            is_contained = True
                            containing_area = other_area
                            break
            
            if is_contained and containing_area:
                # Skip this area (it will be merged into the containing area)
                areas_to_skip.add(i)
                self.logger.debug(f"Merged contained area: {area.label} into larger {containing_area.label}")
            else:
                # Keep this area
                merged_areas.append(area)
        
        self.logger.info(f"🔧 Merged {len(pruned_areas) - len(merged_areas)} contained areas")
        
        return merged_areas
    
    def _is_fully_contained(self, inner_bbox: Tuple[float, float, float, float], 
                           outer_bbox: Tuple[float, float, float, float]) -> bool:
        """Check if inner bounding box is fully contained within outer bounding box"""
        x1_inner, y1_inner, x2_inner, y2_inner = inner_bbox
        x1_outer, y1_outer, x2_outer, y2_outer = outer_bbox
        
        return (x1_inner >= x1_outer and y1_inner >= y1_outer and 
                x2_inner <= x2_outer and y2_inner <= y2_outer)

    def _quick_content_scan(self, page: fitz.Page) -> bool:
        """Quick scan to determine if page is pure text (avoids YOLO overhead)"""
        try:
            # 1. Check for images
            images = page.get_images()
            if images:
                return False
            
            # 2. Check for complex layouts using text analysis
            text_dict = page.get_text("dict")
            if not text_dict.get("blocks"):
                return True  # Empty page is "pure text"
            
            # 3. Analyze text block distribution and spacing
            text_blocks = []
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    text_blocks.append(block)
            
            if len(text_blocks) < 3:
                return True  # Very simple layout
            
            # 4. Check for regular text flow (not complex layouts)
            y_positions = []
            for block in text_blocks:
                if "bbox" in block:
                    y_positions.append(block["bbox"][1])  # Top Y coordinate
            
            if len(y_positions) < 2:
                return True
            
            # Calculate spacing variance (low variance = regular text flow)
            y_positions.sort()
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            avg_spacing = sum(spacings) / len(spacings) if spacings else 0
            
            # If spacing is very regular, likely pure text
            irregular_spacings = sum(1 for s in spacings if abs(s - avg_spacing) > avg_spacing * 0.5)
            irregularity_ratio = irregular_spacings / len(spacings) if spacings else 0
            
            return irregularity_ratio < 0.3  # Less than 30% irregular spacing
            
        except Exception as e:
            self.logger.warning(f"Error in quick content scan: {e}, defaulting to mixed content")
            return False
    
    async def process_page(self, pdf_path: str, page_num: int) -> PageModel:
        """
        Processes a single page and returns a validated PageModel object.

        This method is now the single point of responsibility for creating the
        "single version of truth" for a page's structure.
        """
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            page_width, page_height = page.rect.width, page.rect.height

            elements: List[ElementModel] = []

            # Quick content scan to determine processing path
            if self._quick_content_scan(page):
                # Process as pure text
                text_blocks = self.content_extractor.extract_text_blocks(page)
                
                # Convert text blocks to ElementModel objects
                for text_block in text_blocks:
                    elements.append(ElementModel(
                        type='text',
                        bbox=text_block.bbox,
                        content=text_block.text,
                        formatting={
                            'font_size': text_block.font_size,
                            'font_family': text_block.font_family,
                            'confidence': text_block.confidence,
                            'block_type': text_block.block_type
                        },
                        confidence=text_block.confidence
                    ))
                
                self.logger.info(f"⚡ Page {page_num + 1}: Fast text processing completed in {time.time() - start_time:.3f}s")
            else:
                # Process as mixed content with YOLO
                text_blocks = self.content_extractor.extract_text_blocks(page)
                image_blocks = self.content_extractor.extract_images(page)
                
                # Render page for YOLO
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                page_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                
                # Analyze layout with YOLO
                layout_areas = self.layout_analyzer.analyze_layout(page_image)
                
                # Apply pruning and merging (Directive III)
                layout_areas = self._prune_and_merge_layout_areas(layout_areas)
                
                # Convert text blocks to ElementModel objects
                for text_block in text_blocks:
                    elements.append(ElementModel(
                        type='text',
                        bbox=text_block.bbox,
                        content=text_block.text,
                        formatting={
                            'font_size': text_block.font_size,
                            'font_family': text_block.font_family,
                            'confidence': text_block.confidence,
                            'block_type': text_block.block_type
                        },
                        confidence=text_block.confidence
                    ))
                
                # Convert image blocks to ElementModel objects
                for image_block in image_blocks:
                    elements.append(ElementModel(
                        type='image',
                        bbox=image_block.bbox,
                        content=f"Image {image_block.image_index}",  # Placeholder content
                        formatting={
                            'block_type': image_block.block_type,
                            'image_index': image_block.image_index
                        },
                        confidence=None
                    ))
                
                # Convert layout areas to ElementModel objects
                for layout_area in layout_areas:
                    # Map YOLO labels to ElementType values
                    element_type = layout_area.label if layout_area.label in ['text', 'image', 'table', 'figure', 'title', 'list', 'caption', 'quote', 'footnote', 'equation', 'marginalia', 'bibliography', 'header', 'footer'] else 'text'
                    
                    elements.append(ElementModel(
                        type=element_type,
                        bbox=layout_area.bbox,
                        content=f"Content for {element_type}",  # Placeholder content
                        formatting={
                            'area_id': layout_area.area_id,
                            'class_id': layout_area.class_id
                        },
                        confidence=layout_area.confidence
                    ))
                
                self.logger.info(f"🎯 Page {page_num + 1}: Mixed content processing completed in {time.time() - start_time:.3f}s")

            # Create and return the final PageModel
            page_model = PageModel(
                page_number=page_num + 1,  # 1-based page numbering
                dimensions=[page_width, page_height],  # List of two floats as expected
                elements=elements
            )
            
            doc.close()
            return page_model
            
        except Exception as e:
            self.logger.error(f"❌ Error processing page {page_num + 1}: {e}", exc_info=True)
            # Return a minimal PageModel with error information
            return PageModel(
                page_number=page_num + 1,
                dimensions=[0.0, 0.0],
                elements=[]
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'yolo_available': YOLO_AVAILABLE,
            'document_model_available': DOCUMENT_MODEL_AVAILABLE,
            'confidence_threshold': 0.15,
            'max_detections': 300,
            'supported_classes': [
                'text', 'title', 'paragraph', 'list', 'table', 
                'figure', 'caption', 'quote', 'footnote', 'equation'
            ]
        }
    
    async def process_page_digital_twin(self, pdf_path: str, page_num: int, 
                                      output_dir: str) -> DigitalTwinPageModel:
        """
        Enhanced page processing that creates a Digital Twin representation
        with proper image extraction, saving, and structured content preservation.
        
        This implements the user's "Digital Twin" vision by:
        1. Extracting all content with precise coordinates
        2. Saving images to designated output directory
        3. Creating structured blocks with proper linking
        4. Preserving spatial relationships and metadata
        """
        start_time = time.time()
        
        try:
            # Open document and get page
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            page_width, page_height = page.rect.width, page.rect.height
            
            # Create output directory for images if it doesn't exist
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Initialize Digital Twin page model
            digital_twin_page = DigitalTwinPageModel(
                page_number=page_num + 1,  # 1-based numbering
                dimensions=(page_width, page_height),
                page_metadata={
                    'source_file': pdf_path,
                    'extraction_method': 'pymupdf_yolo_digital_twin',
                    'processing_timestamp': time.time()
                },
                processing_strategy='digital_twin'
            )
            
            # Extract text blocks with enhanced metadata
            raw_text_blocks = self.content_extractor.extract_text_blocks(page)
            
            # Convert to Digital Twin text blocks
            text_block_id = 0
            for text_block in raw_text_blocks:
                text_block_id += 1
                
                # Determine block type based on content analysis
                block_type = self._classify_text_block_type(text_block)
                structural_role = self._determine_structural_role(text_block, block_type)
                
                # Create Digital Twin text block
                dt_text_block = create_text_block(
                    block_id=f"text_{page_num + 1}_{text_block_id}",
                    text=text_block.text,
                    bbox=text_block.bbox,
                    page_number=page_num + 1,
                    block_type=block_type,
                    structural_role=structural_role,
                    font_family=text_block.font_family,
                    font_size=text_block.font_size,
                    confidence=text_block.confidence,
                    extraction_method='pymupdf'
                )
                
                digital_twin_page.add_block(dt_text_block)
            
            # Extract and save images with proper linking
            raw_image_blocks = self.content_extractor.extract_images(page)
            
            image_block_id = 0
            for image_block in raw_image_blocks:
                image_block_id += 1
                
                try:
                    # Extract and save the actual image
                    image_path = self._extract_and_save_image(
                        page, image_block, images_dir, page_num, image_block_id
                    )
                    
                    if image_path:
                        # Create Digital Twin image block with proper file linking
                        dt_image_block = create_image_block(
                            block_id=f"image_{page_num + 1}_{image_block_id}",
                            image_path=image_path,
                            bbox=image_block.bbox,
                            page_number=page_num + 1,
                            structural_role=StructuralRole.ILLUSTRATION,
                            extraction_method='pymupdf',
                            image_format='png',
                            processing_notes=[f"Extracted from page {page_num + 1}"]
                        )
                        
                        digital_twin_page.add_block(dt_image_block)
                        
                        self.logger.info(f"📸 Saved image: {image_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {image_block_id} from page {page_num + 1}: {e}")
                    continue
            
            # Apply YOLO analysis for mixed content if needed
            if not self._quick_content_scan(page):
                self.logger.info(f"🎯 Applying YOLO analysis for mixed content on page {page_num + 1}")
                
                # Render page for YOLO analysis
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                page_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                
                # Analyze layout with YOLO
                layout_areas = self.layout_analyzer.analyze_layout(page_image)
                layout_areas = self._prune_and_merge_layout_areas(layout_areas)
                
                # Enhance blocks with YOLO-detected structure
                self._enhance_blocks_with_yolo_structure(digital_twin_page, layout_areas)
            
            # Record processing time
            processing_time = time.time() - start_time
            digital_twin_page.extraction_time = processing_time
            
            self.logger.info(f"✅ Digital Twin processing completed for page {page_num + 1} in {processing_time:.3f}s")
            self.logger.info(f"   Text blocks: {len(digital_twin_page.text_blocks)}")
            self.logger.info(f"   Image blocks: {len(digital_twin_page.image_blocks)}")
            
            doc.close()
            return digital_twin_page
            
        except Exception as e:
            self.logger.error(f"❌ Digital Twin processing failed for page {page_num + 1}: {e}", exc_info=True)
            
            # Return minimal page model with error information
            error_page = DigitalTwinPageModel(
                page_number=page_num + 1,
                dimensions=(0.0, 0.0),
                page_metadata={'error': str(e), 'processing_failed': True}
            )
            
            if 'doc' in locals():
                doc.close()
            
            return error_page
    
    def _extract_and_save_image(self, page: fitz.Page, image_block: ImageBlock, 
                               images_dir: str, page_num: int, image_id: int) -> Optional[str]:
        """
        Extract and save image to filesystem as required by Digital Twin model.
        
        This implements the user's requirement for proper image file linking
        between the extractor and generator.
        """
        try:
            # Get image list and find the specific image
            images = page.get_images(full=True)
            if image_id - 1 >= len(images):
                self.logger.warning(f"Image index {image_id} out of range for page {page_num + 1}")
                return None
            
            img_info = images[image_id - 1]
            xref = img_info[0]
            
            # Extract image data
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Generate safe filename
            filename = f"page_{page_num + 1}_image_{image_id}.{image_ext}"
            image_path = os.path.join(images_dir, filename)
            
            # Save image to file
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Return relative path for portability
            return os.path.relpath(image_path)
            
        except Exception as e:
            self.logger.error(f"Failed to extract and save image: {e}")
            return None
    
    def _classify_text_block_type(self, text_block: TextBlock) -> BlockType:
        """
        Classify text block type based on content and formatting analysis.
        
        This implements intelligent block type detection to support
        proper document structure reconstruction.
        """
        text = text_block.text.strip()
        font_size = text_block.font_size
        
        # Title detection (large font, short text, capitalized)
        if font_size > 16 and len(text) < 100 and text.isupper():
            return BlockType.TITLE
        
        # Heading detection (larger font, sentence case, ends with colon or period)
        elif font_size > 14 and len(text) < 200 and (text.endswith(':') or text.endswith('.')):
            return BlockType.HEADING
        
        # List item detection (starts with bullet or number)
        elif text.startswith(('•', '▪', '◦', '-')) or (len(text.split('.', 1)) == 2 and text.split('.', 1)[0].isdigit()):
            return BlockType.LIST_ITEM
        
        # Caption detection (starts with "Figure", "Table", "Chart")
        elif any(text.lower().startswith(word) for word in ['figure', 'table', 'chart', 'diagram']):
            return BlockType.CAPTION
        
        # Footnote detection (small font, starts with number or symbol)
        elif font_size < 10 and (text.startswith(('*', '†', '‡')) or (text[0].isdigit() and text[1] in '.):')):
            return BlockType.FOOTNOTE
        
        # Quote detection (indented or starts with quote marks)
        elif text.startswith(('"', '"', '«', '„')) or 'quote' in text.lower():
            return BlockType.QUOTE
        
        # Default to paragraph
        else:
            return BlockType.PARAGRAPH
    
    def _determine_structural_role(self, text_block: TextBlock, block_type: BlockType) -> StructuralRole:
        """Determine the structural role of a text block in the document"""
        
        if block_type in [BlockType.HEADING, BlockType.TITLE]:
            return StructuralRole.NAVIGATION
        elif block_type == BlockType.CAPTION:
            return StructuralRole.ILLUSTRATION
        elif block_type == BlockType.FOOTNOTE:
            return StructuralRole.ANNOTATION
        elif block_type in [BlockType.HEADER, BlockType.FOOTER]:
            return StructuralRole.METADATA
        else:
            return StructuralRole.CONTENT
    
    def _enhance_blocks_with_yolo_structure(self, page_model: DigitalTwinPageModel, 
                                          layout_areas: List[LayoutArea]) -> None:
        """
        Enhance existing text blocks with YOLO-detected structural information.
        
        This refines block classification using YOLO's layout analysis while
        preserving the high-quality PyMuPDF text extraction.
        """
        for layout_area in layout_areas:
            # Find overlapping text blocks
            overlapping_blocks = []
            for text_block in page_model.text_blocks:
                if self._bbox_overlaps(text_block.bbox, layout_area.bbox):
                    overlapping_blocks.append(text_block)
            
            # Update block types based on YOLO detection
            if overlapping_blocks and layout_area.label in ['title', 'heading', 'table', 'figure']:
                for block in overlapping_blocks:
                    # Refine block type if YOLO provides better classification
                    if layout_area.label == 'title':
                        block.block_type = BlockType.TITLE
                    elif layout_area.label == 'heading':
                        block.block_type = BlockType.HEADING
                    
                    # Add YOLO confidence as processing note
                    block.processing_notes.append(
                        f"YOLO classification: {layout_area.label} (conf: {layout_area.confidence:.3f})"
                    )
        
        self.logger.info(f"Enhanced {len(page_model.text_blocks)} text blocks with YOLO structure")
    
    def _bbox_overlaps(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float], 
                      threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap with a given threshold"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)
        
        if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
            return False  # No intersection
        
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        
        # Calculate area of smaller box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        smaller_area = min(area1, area2)
        
        # Check if intersection is significant
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
        return overlap_ratio >= threshold
    
    async def process_document_digital_twin(self, pdf_path: str, output_dir: str) -> DocumentModel:
        """
        Process entire PDF document and create complete Digital Twin representation.
        
        This is the master method that implements the user's complete vision:
        1. Extract TOC using PyMuPDF's native get_toc() method
        2. Process all pages with image extraction and structured content
        3. Create a complete DocumentModel with proper linking
        4. Preserve document structure and relationships
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting Digital Twin document processing: {pdf_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Extract document metadata
            document_metadata = doc.metadata
            document_title = document_metadata.get('title', '') or os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Create Digital Twin document model
            digital_twin_doc = DocumentModel(
                title=document_title,
                filename=os.path.basename(pdf_path),
                total_pages=total_pages,
                document_metadata=document_metadata,
                source_language='auto-detect',  # Will be updated by translation service
                extraction_method='pymupdf_yolo_digital_twin'
            )
            
            # Extract Table of Contents using PyMuPDF's native method
            self.logger.info("📖 Extracting Table of Contents...")
            toc_entries = self._extract_toc_digital_twin(doc)
            
            if toc_entries:
                digital_twin_doc.toc_entries.extend(toc_entries)
                self.logger.info(f"✅ Extracted {len(toc_entries)} TOC entries")
            else:
                self.logger.info("📝 No TOC found in document")
            
            # Process all pages
            self.logger.info(f"📄 Processing {total_pages} pages...")
            
            for page_num in range(total_pages):
                try:
                    # Process page using Digital Twin method
                    digital_twin_page = await self.process_page_digital_twin(
                        pdf_path, page_num, output_dir
                    )
                    
                    # Add page to document
                    digital_twin_doc.add_page(digital_twin_page)
                    
                    self.logger.info(f"✅ Processed page {page_num + 1}/{total_pages}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process page {page_num + 1}: {e}")
                    # Create error page to maintain document integrity
                    error_page = DigitalTwinPageModel(
                        page_number=page_num + 1,
                        dimensions=(0.0, 0.0),
                        page_metadata={'error': str(e), 'processing_failed': True}
                    )
                    digital_twin_doc.add_page(error_page)
            
            # Finalize document processing
            processing_time = time.time() - start_time
            digital_twin_doc.processing_time = processing_time
            
            # Validate document structure
            validation_issues = digital_twin_doc.validate_structure()
            if validation_issues:
                self.logger.warning(f"⚠️ Document validation issues found: {len(validation_issues)}")
                for issue in validation_issues:
                    self.logger.warning(f"   - {issue}")
            
            # Log final statistics
            stats = digital_twin_doc.get_statistics()
            self.logger.info(f"🎉 Digital Twin document processing completed in {processing_time:.3f}s")
            self.logger.info(f"   📊 Document Statistics:")
            self.logger.info(f"      - Total pages: {stats['total_pages']}")
            self.logger.info(f"      - Text blocks: {stats['total_text_blocks']}")
            self.logger.info(f"      - Image blocks: {stats['total_image_blocks']}")
            self.logger.info(f"      - Table blocks: {stats['total_tables']}")
            self.logger.info(f"      - TOC entries: {stats['total_toc_entries']}")
            self.logger.info(f"      - Total words: {stats['total_words']}")
            
            doc.close()
            return digital_twin_doc
            
        except Exception as e:
            self.logger.error(f"❌ Digital Twin document processing failed: {e}", exc_info=True)
            
            # Create minimal document with error information
            error_doc = DocumentModel(
                title="Processing Failed",
                filename=os.path.basename(pdf_path),
                total_pages=0,
                document_metadata={'error': str(e), 'processing_failed': True}
            )
            
            if 'doc' in locals():
                doc.close()
            
            return error_doc
    
    def _extract_toc_digital_twin(self, doc: fitz.Document) -> List[TOCEntry]:
        """
        Extract Table of Contents using PyMuPDF's native get_toc() method.
        
        This implements the user's requirement for structured TOC handling
        rather than treating TOC as plain text.
        """
        try:
            # Get TOC from PyMuPDF
            raw_toc = doc.get_toc()
            
            if not raw_toc:
                return []
            
            toc_entries = []
            entry_id = 0
            
            for toc_item in raw_toc:
                entry_id += 1
                
                # Parse TOC item: [level, title, page_number, dest_dict]
                level = toc_item[0]
                title = toc_item[1].strip()
                page_number = toc_item[2]
                
                # Create TOC entry with Digital Twin model
                toc_entry = TOCEntry(
                    entry_id=f"toc_{entry_id}",
                    title=title,
                    original_title=title,  # Will be used for translation mapping
                    level=level,
                    page_number=page_number,
                    anchor_id=f"toc_anchor_{entry_id}"
                )
                
                toc_entries.append(toc_entry)
                
                self.logger.debug(f"TOC Entry: Level {level}, Page {page_number}, Title: {title[:50]}...")
            
            # Build hierarchical relationships
            self._build_toc_hierarchy(toc_entries)
            
            return toc_entries
            
        except Exception as e:
            self.logger.error(f"Failed to extract TOC: {e}")
            return []
    
    def _build_toc_hierarchy(self, toc_entries: List[TOCEntry]) -> None:
        """
        Build parent-child relationships in TOC entries based on hierarchical levels.
        
        This creates proper navigation structure for the Digital Twin model.
        """
        if not toc_entries:
            return
        
        # Stack to track parent entries at each level
        parent_stack = []
        
        for entry in toc_entries:
            # Remove parents that are at the same or deeper level
            while parent_stack and parent_stack[-1].level >= entry.level:
                parent_stack.pop()
            
            # Set parent relationship if there's a parent in the stack
            if parent_stack:
                parent_entry = parent_stack[-1]
                entry.parent_entry_id = parent_entry.entry_id
                parent_entry.children_ids.append(entry.entry_id)
            
            # Add current entry to stack
            parent_stack.append(entry)
        
        self.logger.info(f"Built hierarchical TOC structure with {len(toc_entries)} entries") 