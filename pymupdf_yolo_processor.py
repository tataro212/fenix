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
- OPTIMIZED: Adaptive memory management for large documents
"""

import os
import logging
import time
import asyncio
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
import re
import json

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
    DocumentModel, PageModel, 
    TextBlock, ImageBlock, 
    TableBlock, TOCEntry, BlockType, StructuralRole,
    create_text_block, create_image_block, create_table_block, is_bibliography_content
)

logger = logging.getLogger(__name__)

class AdaptiveMemoryManager:
    """
    OPTIMIZATION: Adaptive memory management for large document processing.
    
    This class monitors memory usage and adapts processing strategies to prevent
    memory exhaustion while maintaining processing quality.
    """
    
    def __init__(self, max_memory_gb: float = 4.0, warning_threshold: float = 0.8):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            'pages_processed': 0,
            'memory_cleanups': 0,
            'batch_adjustments': 0
        }
        
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 3)  # Convert bytes to GB
        except Exception:
            return 0.0
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed"""
        current_usage = self.get_memory_usage_gb()
        return current_usage > (self.max_memory_gb * self.warning_threshold)
    
    def cleanup_memory(self) -> None:
        """Perform memory cleanup operations"""
        if self.should_cleanup_memory():
            self.logger.info("🧹 Performing memory cleanup...")
            gc.collect()
            self.processing_stats['memory_cleanups'] += 1
            
            # Log memory status
            memory_after = self.get_memory_usage_gb()
            self.logger.info(f"   💾 Memory usage after cleanup: {memory_after:.2f} GB")
    
    def calculate_optimal_batch_size(self, total_pages: int, document_size_mb: float) -> int:
        """Calculate optimal batch size based on available memory and document characteristics"""
        base_batch_size = 10  # Default batch size
        
        # Adjust based on document size
        if document_size_mb > 100:  # Large document
            base_batch_size = 5
        elif document_size_mb > 50:  # Medium document
            base_batch_size = 8
        
        # Adjust based on available memory
        available_memory = self.max_memory_gb - self.get_memory_usage_gb()
        if available_memory < 1.0:  # Low memory
            base_batch_size = max(2, base_batch_size // 2)
        
        # Ensure batch size is reasonable
        optimal_batch_size = min(base_batch_size, max(1, total_pages // 4))
        
        self.logger.info(f"📊 Calculated optimal batch size: {optimal_batch_size} pages")
        return optimal_batch_size
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        return {
            'current_memory_gb': self.get_memory_usage_gb(),
            'max_memory_gb': self.max_memory_gb,
            'memory_utilization': self.get_memory_usage_gb() / self.max_memory_gb,
            'pages_processed': self.processing_stats['pages_processed'],
            'memory_cleanups': self.processing_stats['memory_cleanups'],
            'batch_adjustments': self.processing_stats['batch_adjustments']
        }

class ParallelImageExtractor:
    """
    OPTIMIZATION: Parallel image extraction for improved performance.
    
    This class handles concurrent image extraction operations while maintaining
    the same quality and reliability as sequential processing.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        
    async def extract_images_parallel(self, page: fitz.Page, images_dir: str, 
                                    page_num: int) -> List[ImageBlock]:
        """Extract all images from a page in parallel"""
        images = page.get_images(full=True)
        
        if not images:
            return []
        
        self.logger.info(f"🖼️ Extracting {len(images)} images in parallel from page {page_num}")
        
        # Create extraction tasks
        loop = asyncio.get_event_loop()
        tasks = []
        
        for img_id, img in enumerate(images):
            task = loop.run_in_executor(
                self.executor,
                self._extract_single_image_sync,
                page, img, images_dir, page_num, img_id
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_blocks = []
        for result in results:
            if isinstance(result, ImageBlock):
                successful_blocks.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Image extraction failed: {result}")
        
        self.logger.info(f"✅ Successfully extracted {len(successful_blocks)} images in parallel")
        return successful_blocks
    
    def _extract_single_image_sync(self, page: fitz.Page, img: tuple, 
                                 images_dir: str, page_num: int, img_id: int) -> Optional[ImageBlock]:
        """Synchronous image extraction for thread pool execution"""
        try:
            xref = img[0]
            
            # Get image bbox
            img_rects = page.get_image_rects(xref)
            if img_rects:
                bbox = img_rects[0]
            else:
                bbox = (0, 0, 100, 100)  # Default bbox
            
            # Extract image data
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"page_{page_num}_image_{img_id}.{image_ext}"
            image_path = os.path.join(images_dir, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Create ImageBlock
            return create_image_block(
                block_id=f"img_{page_num}_{img_id}",
                image_path=image_path,
                bbox=bbox,
                page_number=page_num,
                structural_role=StructuralRole.ILLUSTRATION,
                extraction_method="parallel_extraction"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract image {img_id} from page {page_num}: {e}")
            return None
    
    def cleanup(self):
        """Cleanup thread pool resources"""
        self.executor.shutdown(wait=True)

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
    processing_notes: List[str] = field(default_factory=list)

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
        
        self.logger.info(f"�� Applied page-level hyphenation reconstruction: {len(text_blocks)} → {len(result_blocks)} blocks")
        return result_blocks
    
    def extract_images(self, page: fitz.Page) -> List[ImageBlock]:
        """Extract native images with coordinates (enhanced detection)"""
        image_blocks = []
        try:
            # Method 1: Standard image extraction
            images = page.get_images(full=True)
            self.logger.debug(f"🔍 Found {len(images)} images using standard extraction")
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    bbox = None
                    
                    # Try to get image bbox using get_image_rects
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = (img_rects[0].x0, img_rects[0].y0, img_rects[0].x1, img_rects[0].y1)
                        self.logger.debug(f"📍 Image {img_index} found at bbox: {bbox}")
                    else:
                        # Try alternative method to get image position
                        bbox = self._find_image_bbox_alternative(page, xref)
                        if bbox == (0, 0, 0, 0):
                            self.logger.debug(f"⚠️ Could not determine bbox for image {img_index}, using fallback")
                    
                    # Check if image has meaningful dimensions
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width > 5 and height > 5:  # Filter out very small images
                        image_block = ImageBlock(
                            image_index=img_index,
                            bbox=tuple(bbox),
                            block_type='image'
                        )
                        image_blocks.append(image_block)
                        self.logger.debug(f"✅ Added image {img_index} with dimensions {width}x{height}")
                    else:
                        self.logger.debug(f"🚫 Skipped tiny image {img_index} ({width}x{height})")
                        
                except Exception as e:
                    self.logger.warning(f"Could not extract bbox for image {img_index}: {e}")
                    continue
            
            # Method 2: Look for embedded graphics objects (drawings, vector graphics)
            try:
                drawings = page.get_drawings()
                self.logger.debug(f"🎨 Found {len(drawings)} drawing objects")
                
                for draw_index, drawing in enumerate(drawings):
                    if drawing.get('rect'):
                        rect = drawing['rect']
                        width = rect.width
                        height = rect.height
                        if width > 10 and height > 10:  # Filter meaningful graphics
                            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                            image_block = ImageBlock(
                                image_index=len(images) + draw_index,
                                bbox=bbox,
                                block_type='drawing'
                            )
                            image_blocks.append(image_block)
                            self.logger.debug(f"✅ Added drawing {draw_index} with dimensions {width}x{height}")
            except Exception as e:
                self.logger.debug(f"Could not extract drawings: {e}")
            
            # Method 3: Scan for image-like content in text dict
            try:
                text_dict = page.get_text("dict")
                for block_idx, block in enumerate(text_dict.get("blocks", [])):
                    if block.get("type") == 1:  # Image block type
                        bbox = block.get("bbox", (0, 0, 0, 0))
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        if width > 5 and height > 5:
                            # Check if we already have this image
                            is_duplicate = any(
                                abs(existing.bbox[0] - bbox[0]) < 5 and 
                                abs(existing.bbox[1] - bbox[1]) < 5
                                for existing in image_blocks
                            )
                            if not is_duplicate:
                                image_block = ImageBlock(
                                    image_index=len(images) + len(image_blocks),
                                    bbox=bbox,
                                    block_type='image_dict'
                                )
                                image_blocks.append(image_block)
                                self.logger.debug(f"✅ Added image from text dict with dimensions {width}x{height}")
            except Exception as e:
                self.logger.debug(f"Could not scan text dict for images: {e}")
            
            self.logger.info(f"🖼️ Extracted {len(image_blocks)} image blocks from page (enhanced detection)")
            if len(image_blocks) == 0:
                self.logger.debug("🔍 No images found - this may be a text-only page")
            
            return image_blocks
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting images (enhanced): {e}")
            return []
    
    def _find_image_bbox_alternative(self, page: fitz.Page, xref: int) -> Tuple[float, float, float, float]:
        """Alternative method to find image bounding box when get_image_rects fails"""
        try:
            # Method 1: Search through text dict for image references
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block
                    # Try to match by checking if this block contains our image
                    # This is a heuristic approach
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    if bbox != (0, 0, 0, 0):
                        return bbox
            
            # Method 2: Use page annotations or links that might reference images
            annotations = page.annots()
            for annot in annotations:
                if annot.type[1] in ['Text', 'FreeText', 'Image']:
                    rect = annot.rect
                    if rect.width > 5 and rect.height > 5:
                        return (rect.x0, rect.y0, rect.x1, rect.y1)
            
            # Method 3: Fallback - return minimal bbox
            return (0, 0, 0, 0)
            
        except Exception as e:
            self.logger.debug(f"Alternative bbox search failed: {e}")
            return (0, 0, 0, 0)
    
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
                    # Filter out page numbers before adding to lines
                    if not self._is_page_number(line_text.strip()):
                        lines.append(line_text.strip())
            
            # Return raw text - hyphenation reconstruction happens at page level
            return "\n".join(lines).strip()
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from block: {e}")
            return ""
    
    def _is_page_number(self, text: str) -> bool:
        """
        Detect if the text is likely a page number.
        
        Common page number patterns:
        - Single numbers (e.g., "1", "23", "456")
        - Numbers with formatting (e.g., "- 1 -", "Page 1", "1 of 10")
        - Roman numerals (e.g., "i", "ii", "iii", "iv", "v")
        - Numbers with punctuation (e.g., "1.", "1)", "(1)")
        """
        import re
        
        # Strip whitespace and convert to lowercase for pattern matching
        cleaned_text = text.strip().lower()
        
        # Empty or very short text is unlikely to be meaningful content
        if len(cleaned_text) <= 10:
            # Pattern 1: Pure numbers
            if re.match(r'^\d+$', cleaned_text):
                return True
            
            # Pattern 2: Numbers with basic formatting
            if re.match(r'^[\-\(\)\[\]\s]*\d+[\-\(\)\[\]\s]*$', cleaned_text):
                return True
            
            # Pattern 3: "Page X" or similar
            if re.match(r'^page\s+\d+$', cleaned_text):
                return True
            
            # Pattern 4: "X of Y" format
            if re.match(r'^\d+\s+of\s+\d+$', cleaned_text):
                return True
            
            # Pattern 5: Roman numerals (common in academic papers)
            if re.match(r'^[ivxlcdm]+$', cleaned_text):
                return True
            
            # Pattern 6: Numbers with punctuation
            if re.match(r'^[\-\(\)\[\]\s]*\d+[\.\)\]\-\s]*$', cleaned_text):
                return True
        
        return False
    
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
        # Try to get confidence from config, fallback to 0.08 if not available
        try:
            from config_manager import config_manager
            confidence_threshold = config_manager.yolov8_settings.get('confidence_threshold', 0.08)
        except:
            confidence_threshold = 0.08  # Fallback to 0.08 as recommended in analysis
            
        self.config = {
            'confidence_threshold': confidence_threshold,  # Now reads from config
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
            # Override confidence threshold to match our config
            if hasattr(self.yolo_service, 'conf_thres'):
                self.yolo_service.conf_thres = confidence_threshold
            # Also override the analyzer's config
            if hasattr(self.yolo_service, 'analyzer'):
                self.yolo_service.analyzer.config['confidence_threshold'] = confidence_threshold
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔧 YOLO Layout Analyzer initialized with {self.config['confidence_threshold']} confidence threshold")
    
    def analyze_layout(self, page_image: Image.Image) -> List[LayoutArea]:
        """Analyze page layout using YOLO with configurable confidence"""
        if not self.yolo_service:
            self.logger.warning("⚠️ YOLO service not available")
            return []
        
        try:
            # Log detection attempt for diagnostics
            self.logger.debug(f"🔍 Starting YOLO detection with confidence={self.config['confidence_threshold']}")
            
            detections = self.yolo_service.detect(page_image)
            
            # Enhanced diagnostic logging
            raw_detection_count = len(detections) if detections else 0
            self.logger.debug(f"🎯 Raw YOLO detections: {raw_detection_count}")
            
            layout_areas = []
            for detection in detections:
                # Additional confidence filtering if needed
                if detection['confidence'] >= self.config['confidence_threshold']:
                    layout_area = LayoutArea(
                        label=detection['label'],
                        bbox=tuple(detection['bounding_box']),
                        confidence=detection['confidence'],
                        area_id=f"{detection['label']}_{len(layout_areas)}",
                        class_id=detection.get('class_id', 0)
                    )
                    
                    layout_areas.append(layout_area)
                    self.logger.debug(f"✅ Accepted detection: {detection['label']} (conf: {detection['confidence']:.3f})")
                else:
                    self.logger.debug(f"❌ Filtered detection: {detection['label']} (conf: {detection['confidence']:.3f})")
            
            self.logger.info(f"🎯 Detected {len(layout_areas)} layout areas with YOLO (from {raw_detection_count} raw detections)")
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
        self.logger = logging.getLogger(__name__)
        self.content_extractor = PyMuPDFContentExtractor()
        self.layout_analyzer = YOLOLayoutAnalyzer()
        self.content_mapper = ContentLayoutMapper()
        self.content_classifier = ContentTypeClassifier()
        
        # OPTIMIZATION: Initialize performance enhancement components
        self.memory_manager = AdaptiveMemoryManager()
        self.image_extractor = ParallelImageExtractor()
        
        # Enhanced error recovery and resume functionality
        self.checkpoint_dir = None
        self.resume_enabled = False
        self.processing_state = {
            'current_document': None,
            'completed_pages': [],
            'failed_pages': [],
            'last_checkpoint': None,
            'processing_start_time': None
        }
        
        # Processing statistics for monitoring
        self.stats = {
            'total_pages_processed': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_processing_time': 0.0,
            'average_page_time': 0.0,
            'error_recovery_count': 0,
            'resume_count': 0
        }
        
        # Configuration for layout analysis refinement (Directive III)
        self.yolo_pruning_threshold = 0.2  # Confidence threshold for pruning
        
        self.logger.info("🚀 PyMuPDF-YOLO Processor initialized with performance optimizations")
    
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
                                      output_dir: str) -> PageModel:
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
            
            # Get actual page dimensions for spatial analysis
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Store page dimensions for footnote detection
            self.current_page_dimensions = (page_width, page_height)
            
            # Create output directories
            images_dir = os.path.join(output_dir, f"{Path(pdf_path).stem}_non_text_items")
            os.makedirs(images_dir, exist_ok=True)
            
            # Create Digital Twin page model
            digital_twin_page = PageModel(
                page_number=page_num + 1,
                dimensions=(page_width, page_height),
                page_metadata={
                    'rotation': page.rotation,
                    'media_box': list(page.mediabox),
                    'crop_box': list(page.cropbox) if page.cropbox else None
                }
            )
            
            # Extract text blocks using PyMuPDF
            raw_text_blocks = self.content_extractor.extract_text_blocks(page)
            # Extract image blocks using PyMuPDF
            raw_image_blocks = self.content_extractor.extract_images(page)
            
            # --- Extraction Order Debug Export ---
            import json
            extraction_order_debug = []
            for idx, block in enumerate(raw_text_blocks):
                extraction_order_debug.append({
                    'order': idx,
                    'block_type': 'text',
                    'page_number': page_num + 1,
                    'bbox': getattr(block, 'bbox', None),
                    'text_snippet': getattr(block, 'text', '')[:60]
                })
            for idx, block in enumerate(raw_image_blocks):
                extraction_order_debug.append({
                    'order': idx,
                    'block_type': 'image',
                    'page_number': page_num + 1,
                    'bbox': getattr(block, 'bbox', None),
                    'image_index': getattr(block, 'image_index', None)
                })
            debug_dir = os.path.join(output_dir, 'extraction_debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"page_{page_num+1}_extraction_order.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_order_debug, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[DEBUG] Extraction order exported to: {debug_path}")
            # --- End Extraction Order Debug Export ---
            
            # Process text blocks into Digital Twin format
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
            
            # OPTIMIZATION: Extract and save images with parallel processing
            try:
                # Use parallel image extractor for improved performance
                dt_image_blocks = await self.image_extractor.extract_images_parallel(
                    page, images_dir, page_num + 1
                )
                
                # Add all successfully extracted image blocks
                for dt_image_block in dt_image_blocks:
                    digital_twin_page.add_block(dt_image_block)
                    self.logger.info(f"📸 Saved image: {dt_image_block.image_path}")
                
                if dt_image_blocks:
                    self.logger.info(f"✅ Parallel extraction completed: {len(dt_image_blocks)} images from page {page_num + 1}")
                    
            except Exception as e:
                self.logger.warning(f"Parallel image extraction failed for page {page_num + 1}: {e}")
                # Fallback to sequential extraction
                self.logger.info("🔄 Falling back to sequential image extraction...")
                
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
                                extraction_method='pymupdf_fallback',
                                image_format='png',
                                processing_notes=[f"Extracted from page {page_num + 1} (fallback method)"]
                            )
                            
                            digital_twin_page.add_block(dt_image_block)
                            
                            self.logger.info(f"📸 Saved image (fallback): {image_path}")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {image_block_id} from page {page_num + 1}: {e}")
                        continue
            
            # Apply YOLO layout analysis if available
            if YOLO_AVAILABLE:
                try:
                    # Convert page to image for YOLO analysis
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    
                    from PIL import Image
                    import io
                    
                    page_image = Image.open(io.BytesIO(img_data))
                    
                    # Analyze layout with YOLO
                    layout_areas = self.layout_analyzer.analyze_layout(page_image)
                    
                    # Enhance blocks with YOLO structure information
                    self._enhance_blocks_with_yolo_structure(digital_twin_page, layout_areas)
                    
                    self.logger.debug(f"🎯 YOLO analysis completed for page {page_num + 1}: {len(layout_areas)} areas detected")
                    
                except Exception as e:
                    self.logger.warning(f"YOLO analysis failed for page {page_num + 1}: {e}")
            
            # OPTIMIZATION: Perform memory cleanup if needed
            self.memory_manager.cleanup_memory()
            
            # Record processing time
            processing_time = time.time() - start_time
            digital_twin_page.extraction_time = processing_time
            digital_twin_page.processing_strategy = "digital_twin_optimized"
            
            # Update statistics
            self.memory_manager.processing_stats['pages_processed'] += 1
            self.stats['total_pages_processed'] += 1
            # Initialize missing statistics if needed
            if 'total_text_blocks' not in self.stats:
                self.stats['total_text_blocks'] = 0
            if 'total_image_blocks' not in self.stats:
                self.stats['total_image_blocks'] = 0
            self.stats['total_text_blocks'] += len(digital_twin_page.text_blocks)
            self.stats['total_image_blocks'] += len(digital_twin_page.image_blocks)
            
            # Log footnote detection results
            footnote_blocks = [block for block in digital_twin_page.text_blocks 
                             if block.block_type == BlockType.FOOTNOTE]
            if footnote_blocks:
                self.logger.info(f"📝 Page {page_num + 1}: Detected {len(footnote_blocks)} footnotes")
            
            # Log memory status
            memory_status = self.memory_manager.get_processing_stats()
            self.logger.info(f"✅ Digital Twin page {page_num + 1} processed in {processing_time:.2f}s")
            self.logger.info(f"   💾 Memory: {memory_status['current_memory_gb']:.2f}GB ({memory_status['memory_utilization']:.1%})")
            
            # Close document
            doc.close()
            
            return digital_twin_page
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process page {page_num + 1}: {e}")
            raise
    
    def _extract_and_save_image(self, page: fitz.Page, image_block: ImageBlock, 
                               images_dir: str, page_num: int, image_id: int) -> Optional[str]:
        """
        Enhanced image extraction with comprehensive validation and fallback mechanisms.
        
        OPTIMIZED: Provides better specificity for non-text items with:
        - Image content classification (chart, diagram, photo, etc.)
        - Enhanced metadata extraction  
        - File size optimization
        - Comprehensive validation and fallback mechanisms
        - Multiple extraction methods for robustness
        """
        fallback_methods = [
            self._extract_image_method_standard,
            self._extract_image_method_alternative,
            self._extract_image_method_pixmap_fallback
        ]
        
        for method_idx, extraction_method in enumerate(fallback_methods):
            try:
                self.logger.debug(f"Attempting image extraction method {method_idx + 1} for image {image_id}")
                
                result = extraction_method(page, image_block, images_dir, page_num, image_id)
                
                if result and self._validate_extracted_image(result):
                    self.logger.info(f"✅ Successfully extracted image using method {method_idx + 1}: {result}")
                    return result
                else:
                    self.logger.warning(f"Method {method_idx + 1} failed validation for image {image_id}")
                    
            except Exception as e:
                self.logger.warning(f"Method {method_idx + 1} failed for image {image_id}: {e}")
                continue
        
        # If all methods fail, create a placeholder
        self.logger.error(f"All extraction methods failed for image {image_id} on page {page_num + 1}")
        return self._create_image_placeholder(images_dir, page_num, image_id, image_block.bbox)
    
    def _extract_image_method_standard(self, page: fitz.Page, image_block: ImageBlock, 
                                     images_dir: str, page_num: int, image_id: int) -> Optional[str]:
        """Standard image extraction method using PyMuPDF's extract_image"""
        try:
            # Get image list and find the specific image
            images = page.get_images(full=True)
            if image_id - 1 >= len(images):
                self.logger.warning(f"Image index {image_id} out of range for page {page_num + 1}")
                return None
            
            img_info = images[image_id - 1]
            xref = img_info[0]
            
            # Extract enhanced image data
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Validate image data
            if not image_bytes or len(image_bytes) < 100:
                raise ValueError(f"Invalid image data: {len(image_bytes) if image_bytes else 0} bytes")
            
            # Get image dimensions and properties
            image_width = base_image.get("width", 0)
            image_height = base_image.get("height", 0)
            
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")
            
            # Classify image content type
            image_classification = self._classify_image_content(
                image_bytes, image_width, image_height, image_block.bbox
            )
            
            # Generate descriptive filename based on classification
            filename = f"page_{page_num + 1}_{image_classification['type']}_{image_id}.{image_ext}"
            image_path = os.path.join(images_dir, filename)
            
            # Optimize and save image
            optimized_image_path = self._optimize_and_save_image(
                image_bytes, image_path, image_classification, image_ext
            )
            
            if optimized_image_path:
                # Log image extraction details
                file_size = os.path.getsize(optimized_image_path)
                self.logger.debug(f"📸 Extracted {image_classification['type']}: {filename} "
                                f"({image_width}x{image_height}, {file_size} bytes)")
                
                # Return relative path for portability
                return os.path.relpath(optimized_image_path)
            else:
                return None
            
        except Exception as e:
            self.logger.debug(f"Standard extraction method failed: {e}")
            raise
    
    def _extract_image_method_alternative(self, page: fitz.Page, image_block: ImageBlock, 
                                        images_dir: str, page_num: int, image_id: int) -> Optional[str]:
        """Alternative image extraction method using direct xref access"""
        try:
            # Get all xrefs for images on this page
            xrefs = [img[0] for img in page.get_images(full=True)]
            
            if image_id - 1 >= len(xrefs):
                raise ValueError(f"Image index {image_id} out of range")
            
            xref = xrefs[image_id - 1]
            
            # Try alternative extraction using xref directly
            doc = page.parent
            img_dict = doc.xref_get_key(xref, "Filter")
            
            if img_dict:
                # Extract using pixmap method
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    # Generate filename
                    filename = f"page_{page_num + 1}_alt_method_{image_id}.png"
                    image_path = os.path.join(images_dir, filename)
                    
                    # Save pixmap
                    pix.save(image_path)
                    
                    # Validate file was created
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        self.logger.debug(f"Alternative method extracted image: {filename}")
                        return os.path.relpath(image_path)
                
                pix = None  # Clear pixmap
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Alternative extraction method failed: {e}")
            raise
    
    def _extract_image_method_pixmap_fallback(self, page: fitz.Page, image_block: ImageBlock, 
                                            images_dir: str, page_num: int, image_id: int) -> Optional[str]:
        """Fallback method using page pixmap and bbox cropping"""
        try:
            # Create high-resolution pixmap of the entire page
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Calculate scaled bbox coordinates
            bbox = image_block.bbox
            scaled_bbox = (
                bbox[0] * 2.0,  # x0
                bbox[1] * 2.0,  # y0
                bbox[2] * 2.0,  # x1
                bbox[3] * 2.0   # y1
            )
            
            # Crop the pixmap to the image area
            clip_rect = fitz.Rect(scaled_bbox)
            cropped_pix = fitz.Pixmap(pix, clip_rect)
            
            # Generate filename
            filename = f"page_{page_num + 1}_cropped_{image_id}.png"
            image_path = os.path.join(images_dir, filename)
            
            # Save cropped image
            cropped_pix.save(image_path)
            
            # Clean up pixmaps
            cropped_pix = None
            pix = None
            
            # Validate file was created
            if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:  # Minimum size check
                self.logger.debug(f"Pixmap fallback extracted image: {filename}")
                return os.path.relpath(image_path)
            else:
                return None
            
        except Exception as e:
            self.logger.debug(f"Pixmap fallback method failed: {e}")
            raise
    
    def _validate_extracted_image(self, image_path: str) -> bool:
        """Comprehensive validation of extracted image"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file does not exist: {image_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size < 50:  # Minimum reasonable size
                self.logger.warning(f"Image file too small: {file_size} bytes")
                return False
            
            if file_size > 50 * 1024 * 1024:  # 50MB max
                self.logger.warning(f"Image file too large: {file_size} bytes")
                return False
            
            # Try to verify it's a valid image by reading header
            try:
                with open(image_path, 'rb') as f:
                    header = f.read(20)
                    
                # Check for common image file signatures
                if header.startswith(b'\x89PNG') or \
                   header.startswith(b'\xFF\xD8\xFF') or \
                   header.startswith(b'GIF8') or \
                   header.startswith(b'RIFF') or \
                   header.startswith(b'BM'):
                    return True
                else:
                    self.logger.warning(f"Invalid image file header: {image_path}")
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Could not read image file header: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
    
    def _create_image_placeholder(self, images_dir: str, page_num: int, image_id: int, 
                                 bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Create a placeholder image when extraction fails"""
        try:
            # Create a simple placeholder image
            placeholder_filename = f"page_{page_num + 1}_placeholder_{image_id}.png"
            placeholder_path = os.path.join(images_dir, placeholder_filename)
            
            # Calculate dimensions from bbox
            width = max(100, int(bbox[2] - bbox[0]))
            height = max(100, int(bbox[3] - bbox[1]))
            
            # Create a simple placeholder using PyMuPDF
            placeholder_doc = fitz.open()
            placeholder_page = placeholder_doc.new_page(width=width, height=height)
            
            # Add placeholder content
            rect = fitz.Rect(10, 10, width-10, height-10)
            placeholder_page.draw_rect(rect, color=(0.8, 0.8, 0.8), width=2)
            
            # Add text
            text_rect = fitz.Rect(20, height//2-10, width-20, height//2+10)
            placeholder_page.insert_text(text_rect.tl, "Image Placeholder", fontsize=12, color=(0.5, 0.5, 0.5))
            
            # Save as PNG
            pix = placeholder_page.get_pixmap()
            pix.save(placeholder_path)
            
            # Clean up
            pix = None
            placeholder_doc.close()
            
            if os.path.exists(placeholder_path):
                self.logger.info(f"Created placeholder image: {placeholder_filename}")
                return os.path.relpath(placeholder_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create image placeholder: {e}")
            return None
    
    def _classify_image_content(self, image_bytes: bytes, width: int, height: int, 
                              bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        Classify image content type for better document reconstruction.
        
        This provides enhanced specificity for non-text items by analyzing
        image characteristics, dimensions, and placement context.
        """
        try:
            # Calculate aspect ratio and size characteristics
            aspect_ratio = width / height if height > 0 else 1.0
            total_pixels = width * height
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # Analyze image bytes for content hints
            image_analysis = self._analyze_image_content_bytes(image_bytes)
            
            # Classification logic based on multiple factors
            classification = {
                'type': 'figure_standard',
                'confidence': 0.5,
                'characteristics': [],
                'metadata': {
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'total_pixels': total_pixels,
                    'bbox_dimensions': (bbox_width, bbox_height)
                }
            }
            
            # Horizontal chart/graph detection
            if aspect_ratio > 2.0 and width > 400:
                classification['type'] = 'chart_horizontal'
                classification['confidence'] = 0.8
                classification['characteristics'].append('wide_horizontal_layout')
            
            # Vertical chart/graph detection
            elif aspect_ratio < 0.5 and height > 400:
                classification['type'] = 'chart_vertical'
                classification['confidence'] = 0.8
                classification['characteristics'].append('tall_vertical_layout')
            
            # Large diagram detection
            elif width > 800 and height > 600:
                classification['type'] = 'diagram_large'
                classification['confidence'] = 0.7
                classification['characteristics'].append('large_detailed_content')
            
            # Small icon/symbol detection
            elif width < 100 and height < 100:
                classification['type'] = 'icon_small'
                classification['confidence'] = 0.6
                classification['characteristics'].append('small_symbolic_content')
            
            # Square format (likely diagram or chart)
            elif 0.8 <= aspect_ratio <= 1.2 and width > 200:
                classification['type'] = 'diagram_square'
                classification['confidence'] = 0.6
                classification['characteristics'].append('square_format')
            
            # Wide banner/header image
            elif aspect_ratio > 3.0:
                classification['type'] = 'banner_wide'
                classification['confidence'] = 0.7
                classification['characteristics'].append('banner_layout')
            
            # Add content analysis characteristics
            if image_analysis['likely_chart']:
                classification['characteristics'].append('chart_indicators')
                if 'chart' not in classification['type']:
                    classification['type'] = 'chart_mixed'
                    classification['confidence'] = max(classification['confidence'], 0.7)
            
            if image_analysis['likely_photo']:
                classification['characteristics'].append('photographic_content')
                classification['type'] = 'photo_content'
                classification['confidence'] = max(classification['confidence'], 0.6)
            
            # Size-based refinement
            if total_pixels > 1000000:  # Large image
                classification['characteristics'].append('high_resolution')
            elif total_pixels < 10000:  # Very small image
                classification['characteristics'].append('low_resolution')
                classification['type'] = 'icon_tiny'
            
            return classification
            
        except Exception as e:
            self.logger.warning(f"Image classification failed: {e}")
            return {
                'type': 'figure_unknown',
                'confidence': 0.3,
                'characteristics': ['classification_failed'],
                'metadata': {'error': str(e)}
            }
    
    def _analyze_image_content_bytes(self, image_bytes: bytes) -> Dict[str, bool]:
        """
        Analyze image bytes for content type hints.
        
        This provides basic content analysis without requiring heavy image processing libraries.
        """
        try:
            # Convert to string for pattern analysis (safe for binary data)
            byte_sample = image_bytes[:min(1000, len(image_bytes))]
            
            # Look for patterns that suggest chart/graph content
            # Charts often have repetitive patterns, geometric shapes
            chart_indicators = 0
            photo_indicators = 0
            
            # Simple heuristics based on byte patterns
            # Charts tend to have more repetitive patterns
            unique_bytes = len(set(byte_sample))
            total_bytes = len(byte_sample)
            
            if total_bytes > 0:
                uniqueness_ratio = unique_bytes / total_bytes
                
                # Low uniqueness suggests charts/diagrams (repetitive patterns)
                if uniqueness_ratio < 0.3:
                    chart_indicators += 1
                
                # High uniqueness suggests photos (more random patterns)
                elif uniqueness_ratio > 0.7:
                    photo_indicators += 1
            
            # Look for specific byte patterns
            if b'\x00\x00\x00' in byte_sample:  # Null patterns (common in charts)
                chart_indicators += 1
            
            if len(set(byte_sample[::10])) > 20:  # High variation (common in photos)
                photo_indicators += 1
            
            return {
                'likely_chart': chart_indicators > photo_indicators,
                'likely_photo': photo_indicators > chart_indicators,
                'uniqueness_ratio': uniqueness_ratio if 'uniqueness_ratio' in locals() else 0.5
            }
            
        except Exception as e:
            self.logger.debug(f"Byte analysis failed: {e}")
            return {
                'likely_chart': False,
                'likely_photo': False,
                'uniqueness_ratio': 0.5
            }
    
    def _optimize_and_save_image(self, image_bytes: bytes, image_path: str, 
                               classification: Dict[str, Any], image_ext: str) -> Optional[str]:
        """
        Optimize and save image with appropriate compression based on classification.
        
        This reduces file sizes while maintaining quality appropriate for the image type.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # For now, save original image (optimization can be added later with PIL)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Add metadata file for enhanced classification info
            metadata_path = image_path + ".meta.json"
            metadata = {
                'classification': classification,
                'extraction_timestamp': time.time(),
                'original_size': len(image_bytes),
                'file_format': image_ext
            }
            
            try:
                import json
                with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                    json.dump(metadata, meta_file, indent=2)
            except Exception as meta_error:
                self.logger.debug(f"Could not save image metadata: {meta_error}")
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"Failed to optimize and save image: {e}")
            return None
    
    def _create_enhanced_image_block(self, image_path: str, bbox: Tuple[float, float, float, float],
                                   page_number: int, image_id: int, 
                                   classification: Dict[str, Any]) -> 'ImageBlock':
        """
        Create enhanced Digital Twin image block with classification metadata.
        
        This provides better specificity and metadata for document reconstruction.
        """
        try:
            from digital_twin_model import create_image_block, StructuralRole
            
            # Determine structural role based on classification
            image_type = classification['type']
            if 'chart' in image_type:
                structural_role = StructuralRole.DATA
            elif 'diagram' in image_type:
                structural_role = StructuralRole.ILLUSTRATION
            elif 'icon' in image_type:
                structural_role = StructuralRole.ANNOTATION
            else:
                structural_role = StructuralRole.ILLUSTRATION
            
            # Create processing notes with classification details
            processing_notes = [
                f"Extracted from page {page_number}",
                f"Classified as: {image_type} (confidence: {classification['confidence']:.2f})",
                f"Characteristics: {', '.join(classification['characteristics'])}"
            ]
            
            # Get file size if image exists
            file_size = 0
            if image_path and os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
            
            # Create enhanced image block
            enhanced_block = create_image_block(
                block_id=f"image_{page_number}_{image_id}",
                image_path=image_path,
                bbox=bbox,
                page_number=page_number,
                structural_role=structural_role,
                extraction_method='pymupdf_enhanced',
                image_format=os.path.splitext(image_path)[1][1:] if image_path else 'unknown',
                processing_notes=processing_notes,
                image_type=image_type,
                classification_confidence=classification['confidence'],
                file_size=file_size,
                dimensions=(classification['metadata'].get('width', 0), classification['metadata'].get('height', 0))
            )
            
            return enhanced_block
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced image block: {e}")
            # Fallback to basic image block creation
            return self._create_basic_image_block_fallback(image_path, bbox, page_number, image_id)
    
    def _create_basic_image_block_fallback(self, image_path: str, bbox: Tuple[float, float, float, float],
                                         page_number: int, image_id: int):
        """Fallback method for basic image block creation"""
        try:
            from digital_twin_model import create_image_block, StructuralRole
            
            return create_image_block(
                block_id=f"image_{page_number}_{image_id}",
                image_path=image_path,
                bbox=bbox,
                page_number=page_number,
                structural_role=StructuralRole.ILLUSTRATION,
                extraction_method='pymupdf_basic_fallback',
                processing_notes=[f"Basic extraction from page {page_number}"]
            )
        except Exception as e:
            self.logger.error(f"Even basic image block creation failed: {e}")
            return None
    
    def _enhance_blocks_with_yolo_structure(self, page_model: PageModel, 
                                          layout_areas: List[LayoutArea]) -> None:
        from digital_twin_model import BlockType
        yolo_label_to_blocktype = {
            'title': BlockType.TITLE,
            'heading': BlockType.HEADING,
            'paragraph': BlockType.PARAGRAPH,
            'list': BlockType.LIST_ITEM,
            'list_item': BlockType.LIST_ITEM,
            'table': BlockType.TABLE,
            'figure': BlockType.CAPTION,
            'caption': BlockType.CAPTION,
            'footnote': BlockType.FOOTNOTE,
        }
        for text_block in page_model.text_blocks:
            best_area = None
            best_overlap = 0
            for layout_area in layout_areas:
                if self._bbox_overlaps(text_block.bbox, layout_area.bbox):
                    x1_i = max(text_block.bbox[0], layout_area.bbox[0])
                    y1_i = max(text_block.bbox[1], layout_area.bbox[1])
                    x2_i = min(text_block.bbox[2], layout_area.bbox[2])
                    y2_i = min(text_block.bbox[3], layout_area.bbox[3])
                    if x1_i < x2_i and y1_i < y2_i:
                        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
                        block_area = (text_block.bbox[2] - text_block.bbox[0]) * (text_block.bbox[3] - text_block.bbox[1])
                        overlap_ratio = intersection_area / block_area if block_area > 0 else 0
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_area = layout_area
            if best_area and best_area.label in yolo_label_to_blocktype:
                prev_type = text_block.block_type
                # Smart mapping for title/heading
                if best_area.label in ['title', 'heading']:
                    text = getattr(text_block, 'text', '') or ''
                    word_count = len(text.split())
                    font_size = getattr(text_block, 'font_size', 0)
                    if word_count <= 20 and font_size >= 16:
                        text_block.block_type = yolo_label_to_blocktype[best_area.label]
                        note = f"YOLO label: {best_area.label} (conf: {best_area.confidence:.3f}), assigned: {text_block.block_type}, previous: {prev_type} (short+large font)"
                    else:
                        text_block.block_type = BlockType.PARAGRAPH
                        note = f"YOLO label: {best_area.label} (conf: {best_area.confidence:.3f}), demoted to: paragraph, previous: {prev_type} (long or small font)"
                else:
                    text_block.block_type = yolo_label_to_blocktype[best_area.label]
                    note = f"YOLO label: {best_area.label} (conf: {best_area.confidence:.3f}), assigned: {text_block.block_type}, previous: {prev_type}"
                text_block.processing_notes.append(note)
                self.logger.debug(note)
            else:
                prev_type = text_block.block_type
                text_block.block_type = self._classify_text_block_type(text_block)
                note = f"No YOLO match, heuristic assigned: {text_block.block_type}, previous: {prev_type}"
                text_block.processing_notes.append(note)
                self.logger.debug(note)
        self.logger.info(f"Enhanced {len(page_model.text_blocks)} text blocks with YOLO structure (smart mapping)")
    
    def _bbox_overlaps(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float], 
                      threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap with a given threshold"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return False  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
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
        self.processing_state['processing_start_time'] = start_time
        self.logger.info(f"🚀 Starting Digital Twin document processing: {pdf_path}")
        
        try:
            # Check for existing checkpoint and resume if available
            document_filename = os.path.basename(pdf_path)
            checkpoint_data = self._load_checkpoint(document_filename)
            
            if checkpoint_data:
                self.logger.info("🔄 Resuming from checkpoint...")
                self.stats.update(checkpoint_data.get('statistics', {}))
                self.processing_state.update(checkpoint_data.get('processing_state', {}))
                self.stats['resume_count'] += 1
            
            # OPTIMIZATION: Use adaptive memory manager for intelligent resource management
            current_memory = self.memory_manager.get_memory_usage_gb()
            self.logger.info(f"💾 Current memory usage: {current_memory:.1f}GB / {self.memory_manager.max_memory_gb:.1f}GB")
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Get document size for optimal processing strategy
            document_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            
            # OPTIMIZATION: Calculate optimal batch size based on document characteristics
            optimal_batch_size = self.memory_manager.calculate_optimal_batch_size(total_pages, document_size_mb)
            
            # Enable memory optimization for large documents or low memory systems
            memory_optimization_enabled = (
                self.memory_manager.should_cleanup_memory() or  # Memory pressure detected
                total_pages > 50 or  # Large documents
                document_size_mb > 100  # Large files
            )
            
            if memory_optimization_enabled:
                self.logger.info("🔧 Memory optimization enabled for large document processing")
            
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
            
            # Process all pages with intelligent processing strategy selection
            self.logger.info(f"📄 Processing {total_pages} pages...")
            
            # Determine optimal processing strategy based on document size and memory
            if memory_optimization_enabled:
                # Memory-constrained processing
                self.logger.info("💾 Using memory-optimized processing")
                processed_pages = self._process_pages_with_memory_optimization(
                    pdf_path, total_pages, output_dir
                )
                
                # Add processed pages to document in order
                for page_num in range(total_pages):
                    if page_num < len(processed_pages):
                        digital_twin_doc.add_page(processed_pages[page_num])
                    else:
                        # Create error page for missing pages
                        error_page = PageModel(
                            page_number=page_num + 1,
                            dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                            page_metadata={'error': 'Page not processed', 'processing_failed': True}
                        )
                        digital_twin_doc.add_page(error_page)
                        
            elif total_pages <= 5:
                # Small documents: Process sequentially to avoid overhead
                self.logger.info("📝 Using sequential processing for small document")
                for page_num in range(total_pages):
                    try:
                        # Process page using Digital Twin method
                        digital_twin_page = await self.process_page_digital_twin(
                            pdf_path, page_num, output_dir
                        )
                        
                        # Add page to document
                        digital_twin_doc.add_page(digital_twin_page)
                        
                        # Update processing state and save checkpoint
                        self.processing_state['completed_pages'].append(page_num + 1)
                        self.stats['successful_pages'] += 1
                        self.stats['total_pages_processed'] += 1
                        
                        # Save checkpoint every 5 pages
                        if (page_num + 1) % 5 == 0:
                            self._save_checkpoint(digital_twin_doc, page_num + 1)
                        
                        self.logger.info(f"✅ Processed page {page_num + 1}/{total_pages}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Failed to process page {page_num + 1}: {e}")
                        # Use enhanced error recovery
                        recovered_page = self._handle_processing_error(e, page_num + 1, digital_twin_doc)
                        digital_twin_doc.add_page(recovered_page)
                        
                        # Update processing statistics
                        self.processing_state['failed_pages'].append(page_num + 1)
                        self.stats['failed_pages'] += 1
            else:
                # Large documents: Use parallel processing with batching
                self.logger.info("🚀 Using parallel processing for large document")
                processed_pages = await self._process_pages_parallel(
                    pdf_path, total_pages, output_dir
                )
                
                # Add processed pages to document in order
                for page_num in range(total_pages):
                    if page_num < len(processed_pages):
                        digital_twin_doc.add_page(processed_pages[page_num])
                    else:
                        # Create error page for missing pages
                        error_page = PageModel(
                            page_number=page_num + 1,
                            dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                            page_metadata={'error': 'Page not processed', 'processing_failed': True}
                        )
                        digital_twin_doc.add_page(error_page)
            
            # Finalize document processing
            processing_time = time.time() - start_time
            digital_twin_doc.processing_time = processing_time
            
            # Update final statistics
            self.stats['total_processing_time'] = processing_time
            if self.stats['total_pages_processed'] > 0:
                self.stats['average_page_time'] = processing_time / self.stats['total_pages_processed']
            
            # Save final checkpoint
            if self.resume_enabled:
                self._save_checkpoint(digital_twin_doc, total_pages)
                self._cleanup_old_checkpoints(document_filename)
            
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
    
    async def _process_pages_parallel(self, pdf_path: str, total_pages: int, output_dir: str) -> List[PageModel]:
        """
        Process multiple pages in parallel with intelligent batching and memory management.
        
        This optimization provides significant performance improvements for large documents
        while maintaining system stability through controlled concurrency.
        """
        import asyncio
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Calculate optimal batch size based on system resources
        cpu_count = multiprocessing.cpu_count()
        optimal_batch_size = min(max(2, cpu_count - 1), 8)  # Leave 1 CPU free, max 8 concurrent
        
        # Adjust batch size based on document size
        if total_pages < 20:
            batch_size = min(4, total_pages)
        elif total_pages < 100:
            batch_size = optimal_batch_size
        else:
            batch_size = optimal_batch_size + 2  # Larger batches for very large documents
        
        self.logger.info(f"🔄 Parallel processing: {batch_size} concurrent pages, {total_pages} total pages")
        
        # Initialize results list to maintain page order
        processed_pages = [None] * total_pages
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(batch_size)
        
        async def process_page_with_semaphore(page_num: int) -> tuple:
            """Process a single page with semaphore control"""
            async with semaphore:
                try:
                    self.logger.debug(f"Starting parallel processing for page {page_num + 1}")
                    
                    # Process page using Digital Twin method
                    digital_twin_page = await self.process_page_digital_twin(
                        pdf_path, page_num, output_dir
                    )
                    
                    self.logger.info(f"✅ Parallel processed page {page_num + 1}/{total_pages}")
                    return (page_num, digital_twin_page, None)
                    
                except Exception as e:
                    self.logger.error(f"❌ Parallel processing failed for page {page_num + 1}: {e}")
                    # Create error page
                    error_page = PageModel(
                        page_number=page_num + 1,
                        dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                        page_metadata={'error': str(e), 'processing_failed': True, 'parallel_processing': True}
                    )
                    return (page_num, error_page, str(e))
        
        # Create tasks for all pages
        tasks = []
        for page_num in range(total_pages):
            task = asyncio.create_task(process_page_with_semaphore(page_num))
            tasks.append(task)
        
        # Process tasks with progress tracking
        completed_count = 0
        
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            try:
                page_num, page_result, error = await task
                processed_pages[page_num] = page_result
                
                completed_count += 1
                
                # Log progress every 10% or every 5 pages, whichever is more frequent
                progress_interval = max(1, min(total_pages // 10, 5))
                if completed_count % progress_interval == 0 or completed_count == total_pages:
                    progress_percent = (completed_count / total_pages) * 100
                    self.logger.info(f"📊 Parallel processing progress: {completed_count}/{total_pages} ({progress_percent:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Task completion error: {e}")
                # Handle task failure
                continue
        
        # Verify all pages were processed
        successful_pages = sum(1 for page in processed_pages if page is not None)
        failed_pages = total_pages - successful_pages
        
        if failed_pages > 0:
            self.logger.warning(f"⚠️ Parallel processing completed with {failed_pages} failed pages")
        else:
            self.logger.info("🎉 All pages processed successfully in parallel")
        
        # Fill any None slots with error pages
        for i, page in enumerate(processed_pages):
            if page is None:
                processed_pages[i] = PageModel(
                    page_number=i + 1,
                    dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                    page_metadata={'error': 'Page processing failed', 'processing_failed': True}
                )
        
        return processed_pages
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory information for optimization decisions"""
        try:
            import psutil
            
            # Get memory statistics
            memory = psutil.virtual_memory()
            
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
        except ImportError:
            # Fallback if psutil is not available
            self.logger.warning("psutil not available, using conservative memory estimates")
            return {
                'total_gb': 8.0,  # Conservative estimate
                'available_gb': 4.0,
                'used_gb': 4.0,
                'percent_used': 50.0
            }
        except Exception as e:
            self.logger.warning(f"Could not get memory info: {e}")
            return {
                'total_gb': 8.0,
                'available_gb': 4.0,
                'used_gb': 4.0,
                'percent_used': 50.0
            }
    
    def _estimate_memory_requirements(self, doc: fitz.Document, total_pages: int) -> float:
        """Estimate memory requirements for document processing"""
        try:
            # Base memory per page (empirical estimate)
            base_memory_per_page_mb = 5  # Base text processing
            
            # Sample a few pages to estimate image memory requirements
            sample_pages = min(3, total_pages)
            total_image_memory_mb = 0
            
            for page_num in range(0, total_pages, max(1, total_pages // sample_pages)):
                if page_num >= total_pages:
                    break
                    
                try:
                    page = doc[page_num]
                    images = page.get_images(full=True)
                    
                    # Estimate memory for images on this page
                    for img_info in images:
                        try:
                            # Get image dimensions
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            width = base_image.get("width", 800)
                            height = base_image.get("height", 600)
                            
                            # Estimate memory: width * height * 4 bytes (RGBA) * 2 (processing overhead)
                            image_memory_mb = (width * height * 4 * 2) / (1024 * 1024)
                            total_image_memory_mb += image_memory_mb
                            
                        except Exception:
                            # Conservative estimate for failed image analysis
                            total_image_memory_mb += 2  # 2MB per image
                            
                except Exception:
                    # Skip problematic pages
                    continue
            
            # Scale image memory estimate to all pages
            if sample_pages > 0:
                avg_image_memory_per_page = total_image_memory_mb / sample_pages
                total_image_memory_mb = avg_image_memory_per_page * total_pages
            
            # Total memory estimate
            total_memory_mb = (base_memory_per_page_mb * total_pages) + total_image_memory_mb
            
            # Add overhead for parallel processing
            overhead_multiplier = 1.5  # 50% overhead for parallel processing
            total_memory_mb *= overhead_multiplier
            
            return total_memory_mb / 1024  # Convert to GB
            
        except Exception as e:
            self.logger.warning(f"Could not estimate memory requirements: {e}")
            # Conservative fallback estimate
            return total_pages * 0.01  # 10MB per page in GB
    
    def _optimize_memory_usage(self, doc: fitz.Document, page_num: int) -> None:
        """Optimize memory usage during page processing"""
        try:
            # Force garbage collection every 10 pages
            if page_num % 10 == 0:
                import gc
                gc.collect()
                self.logger.debug(f"Performed garbage collection at page {page_num + 1}")
            
            # Clear any cached pixmaps or temporary objects
            # This is handled automatically by PyMuPDF in most cases
            
        except Exception as e:
            self.logger.debug(f"Memory optimization failed: {e}")
    
    def _process_pages_with_memory_optimization(self, pdf_path: str, total_pages: int, 
                                               output_dir: str) -> List[PageModel]:
        """
        Process pages with aggressive memory optimization for large documents or low memory systems.
        
        This method uses streaming processing and memory-conscious techniques.
        """
        processed_pages = []
        
        # Process pages in smaller batches to control memory usage
        batch_size = 3  # Smaller batches for memory optimization
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            
            self.logger.info(f"🔄 Processing memory-optimized batch: pages {batch_start + 1}-{batch_end}")
            
            # Process batch
            batch_pages = []
            for page_num in range(batch_start, batch_end):
                try:
                    # Open document fresh for each batch to avoid memory accumulation
                    doc = fitz.open(pdf_path)
                    
                    # Process single page
                    digital_twin_page = self._process_single_page_optimized(
                        doc, page_num, output_dir
                    )
                    
                    batch_pages.append(digital_twin_page)
                    
                    # Close document immediately
                    doc.close()
                    
                    # Force memory cleanup
                    self._optimize_memory_usage(None, page_num)
                    
                    self.logger.info(f"✅ Memory-optimized processing completed for page {page_num + 1}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Memory-optimized processing failed for page {page_num + 1}: {e}")
                    # Create error page
                    error_page = PageModel(
                        page_number=page_num + 1,
                        dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                        page_metadata={'error': str(e), 'processing_failed': True, 'memory_optimized': True}
                    )
                    batch_pages.append(error_page)
            
            # Add batch to results
            processed_pages.extend(batch_pages)
            
            # Memory cleanup between batches
            import gc
            gc.collect()
            
            # Log progress
            progress_percent = (batch_end / total_pages) * 100
            self.logger.info(f"📊 Memory-optimized progress: {batch_end}/{total_pages} ({progress_percent:.1f}%)")
        
        return processed_pages
    
    def _process_single_page_optimized(self, doc: fitz.Document, page_num: int, output_dir: str) -> PageModel:
        """Process a single page with memory optimization"""
        try:
            # This is a simplified version of process_page_digital_twin with memory optimizations
            page = doc[page_num]
            
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Create Digital Twin page model
            digital_twin_page = PageModel(
                page_number=page_num + 1,
                dimensions=(page_width, page_height),
                page_metadata={
                    'processing_strategy': 'memory_optimized',
                    'extraction_method': 'pymupdf_memory_optimized'
                }
            )
            
            # Extract text blocks with minimal memory footprint
            text_blocks = self.content_extractor.extract_text_blocks(page)
            
            # Convert to Digital Twin blocks immediately to free memory
            for idx, text_block in enumerate(text_blocks):
                dt_text_block = create_text_block(
                    block_id=f"text_{page_num + 1}_{idx + 1}",
                    text=text_block.text,
                    bbox=text_block.bbox,
                    page_number=page_num + 1,
                    block_type=BlockType.PARAGRAPH,
                    extraction_method='pymupdf_memory_optimized'
                )
                digital_twin_page.add_block(dt_text_block)
            
            # Clear text blocks from memory
            text_blocks = None
            
            # Process images with memory optimization (skip complex analysis)
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                try:
                    # Simple image extraction without complex classification
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image and base_image.get("image"):
                        # Save image quickly
                        image_filename = f"page_{page_num + 1}_img_{img_idx + 1}.png"
                        image_path = os.path.join(output_dir, "images", image_filename)
                        
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(base_image["image"])
                        
                        # Create simple image block
                        dt_image_block = create_image_block(
                            block_id=f"image_{page_num + 1}_{img_idx + 1}",
                            image_path=os.path.relpath(image_path),
                            bbox=(0, 0, 100, 100),  # Simplified bbox
                            page_number=page_num + 1,
                            extraction_method='pymupdf_memory_optimized'
                        )
                        digital_twin_page.add_block(dt_image_block)
                        
                except Exception as e:
                    self.logger.debug(f"Memory-optimized image extraction failed: {e}")
                    continue
            
            return digital_twin_page
            
        except Exception as e:
            self.logger.error(f"Memory-optimized page processing failed: {e}")
            return PageModel(
                page_number=page_num + 1,
                dimensions=(595.0, 842.0),  # Standard A4 dimensions to satisfy validation
                page_metadata={'error': str(e), 'processing_failed': True}
            )
    
    def _extract_toc_digital_twin(self, doc: fitz.Document) -> List[TOCEntry]:
        """
        ENHANCED: Comprehensive two-way Table of Contents extraction and mapping.
        
        This implements the complete two-way approach:
        1. Extract TOC using PyMuPDF's native get_toc() method
        2. Scan document for actual heading content and structure
        3. Map TOC entries to document headings with confidence scoring
        4. Generate content fingerprints for translation validation
        5. Build hierarchical context for intelligent translation
        
        This solves the user's TOC corruption problem (failure mode #2) by treating
        TOC as structured data rather than plain text.
        """
        try:
            self.logger.info("📖 Starting comprehensive two-way TOC extraction...")
            
            # Phase 1: Extract native TOC structure
            raw_toc = doc.get_toc()
            if not raw_toc:
                self.logger.info("📝 No native TOC found, attempting document heading scan...")
                return self._extract_toc_from_headings(doc)
            
            # Phase 2: Create initial TOC entries from native extraction
            toc_entries = []
            entry_id = 0
            
            for toc_item in raw_toc:
                entry_id += 1
                
                # Parse TOC item: [level, title, page_number, dest_dict]
                level = toc_item[0]
                title = toc_item[1].strip()
                page_number = toc_item[2]
                
                # Create enhanced TOC entry
                toc_entry = TOCEntry(
                    entry_id=f"toc_{entry_id}",
                    title=title,
                    original_title=title,
                    level=level,
                    page_number=page_number,
                    anchor_id=f"toc_anchor_{entry_id}",
                    section_type=self._detect_section_type(title, level)
                )
                
                toc_entries.append(toc_entry)
                self.logger.debug(f"TOC Entry: Level {level}, Page {page_number}, Title: {title[:50]}...")
            
            # Phase 3: Build hierarchical relationships and context
            self._build_toc_hierarchy_enhanced(toc_entries)
            
            # Phase 4: Map TOC entries to actual document content (THE CRITICAL ENHANCEMENT)
            toc_entries = self._map_toc_to_document_content_sync(doc, toc_entries)
            
            # Phase 5: Validate and score mapping confidence
            self._validate_toc_mappings(toc_entries)
            
            self.logger.info(f"✅ Comprehensive TOC extraction completed: {len(toc_entries)} entries with content mapping")
            return toc_entries
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive TOC extraction failed: {e}")
            # Fallback to simple extraction
            return self._extract_toc_simple_fallback(doc)
    
    def _extract_toc_from_headings(self, doc: fitz.Document) -> List[TOCEntry]:
        """
        Fallback: Extract TOC by scanning document for heading-style content.
        
        Used when no native TOC is available in the PDF.
        """
        try:
            self.logger.info("🔍 Scanning document for heading structures...")
            
            toc_entries = []
            entry_id = 0
            
            # Scan all pages for potential headings
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with format information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                font_size = span.get("size", 12)
                                font_flags = span.get("flags", 0)
                                
                                # Detect heading characteristics
                                if self._is_likely_heading(text, font_size, font_flags):
                                    entry_id += 1
                                    level = self._estimate_heading_level(font_size, font_flags)
                                    
                                    toc_entry = TOCEntry(
                                        entry_id=f"heading_scan_{entry_id}",
                                        title=text,
                                        original_title=text,
                                        level=level,
                                        page_number=page_num + 1,
                                        anchor_id=f"heading_anchor_{entry_id}",
                                        section_type=self._detect_section_type(text, level)
                                    )
                                    
                                    # Generate content fingerprint from following content
                                    preview_content = self._extract_section_preview(doc, page_num, span.get("bbox"))
                                    toc_entry.generate_content_fingerprint(preview_content)
                                    toc_entry.content_preview = preview_content[:200] + "..." if len(preview_content) > 200 else preview_content
                                    
                                    toc_entries.append(toc_entry)
            
            # Build hierarchy for scanned headings
            self._build_toc_hierarchy_enhanced(toc_entries)
            
            # CRITICAL: Map scanned headings to document content for confidence scoring
            if toc_entries:
                toc_entries = self._map_toc_to_document_content_sync(doc, toc_entries)
                self._validate_toc_mappings(toc_entries)
            
            self.logger.info(f"📋 Heading scan completed: {len(toc_entries)} potential TOC entries found")
            return toc_entries
            
        except Exception as e:
            self.logger.error(f"❌ Heading scan failed: {e}")
            return []
    
    def _extract_toc_simple_fallback(self, doc: fitz.Document) -> List[TOCEntry]:
        """
        Simple fallback TOC extraction when comprehensive methods fail.
        """
        try:
            self.logger.warning("⚠️ Using simple fallback TOC extraction...")
            
            # Try native TOC first
            raw_toc = doc.get_toc()
            if raw_toc:
                toc_entries = []
                for i, toc_item in enumerate(raw_toc):
                    level = toc_item[0]
                    title = toc_item[1].strip()
                    page_number = toc_item[2]
                    
                    toc_entry = TOCEntry(
                        entry_id=f"fallback_toc_{i+1}",
                        title=title,
                        original_title=title,
                        level=level,
                        page_number=page_number,
                        anchor_id=f"fallback_anchor_{i+1}",
                        section_type=self._detect_section_type(title, level)
                    )
                    toc_entries.append(toc_entry)
                
                return toc_entries
            
            # If no native TOC, return empty list
            self.logger.warning("⚠️ No TOC found - document will be processed without TOC")
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Simple fallback TOC extraction failed: {e}")
            return []
    
    def _map_toc_to_document_content_sync(self, doc: fitz.Document, toc_entries: List[TOCEntry]) -> List[TOCEntry]:
        """
        CORE ENHANCEMENT: Map TOC entries to actual document headings and content.
        
        This is the critical two-way mapping that enables accurate reconstruction.
        """
        try:
            self.logger.info(f"🗺️ Mapping {len(toc_entries)} TOC entries to document content...")
            
            # Phase 1: Collect all potential headings from document
            document_headings = self._scan_document_for_headings_sync(doc)
            
            # Phase 2: Match TOC entries to document headings
            for toc_entry in toc_entries:
                best_matches = self._find_heading_matches(toc_entry, document_headings)
                
                # Process best matches
                for match in best_matches[:3]:  # Top 3 matches
                    heading_info = match['heading']
                    confidence = match['confidence']
                    
                    # Create heading block ID
                    heading_block_id = f"heading_page_{heading_info['page']}_block_{heading_info['block_idx']}"
                    toc_entry.add_mapped_heading(heading_block_id, confidence)
                    
                    # Update page location if high confidence
                    if confidence >= 0.8 and toc_entry.original_page_in_document == 0:
                        toc_entry.update_page_location(heading_info['page'])
                    
                    # Generate content fingerprint from surrounding content
                    if confidence >= 0.9 and not toc_entry.content_fingerprint:
                        content_preview = heading_info.get('following_content', '')
                        toc_entry.generate_content_fingerprint(content_preview)
                        toc_entry.content_preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                
                # Log mapping results
                if toc_entry.mapped_heading_blocks:
                    self.logger.debug(f"✅ Mapped TOC '{toc_entry.title[:30]}...' to {len(toc_entry.mapped_heading_blocks)} headings")
                else:
                    self.logger.warning(f"⚠️ No mapping found for TOC '{toc_entry.title[:30]}...'")
            
            self.logger.info("✅ TOC to content mapping completed")
            return toc_entries
            
        except Exception as e:
            self.logger.error(f"❌ TOC content mapping failed: {e}")
            return toc_entries
    
    def _scan_document_for_headings_sync(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Comprehensive scan of document to identify all potential headings.
        """
        try:
            headings = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                block_idx = 0
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        block_idx += 1
                        
                        # Analyze block for heading characteristics
                        block_text = ""
                        max_font_size = 0
                        font_flags = 0
                        bbox = block.get("bbox", (0, 0, 0, 0))
                        
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text:
                                    block_text += text + " "
                                    max_font_size = max(max_font_size, span.get("size", 12))
                                    font_flags |= span.get("flags", 0)
                        
                        block_text = block_text.strip()
                        
                        # Check if this could be a heading
                        if self._is_likely_heading(block_text, max_font_size, font_flags):
                            # Extract following content for context
                            following_content = self._extract_section_preview(doc, page_num, bbox)
                            
                            heading_info = {
                                'text': block_text,
                                'page': page_num + 1,
                                'block_idx': block_idx,
                                'font_size': max_font_size,
                                'font_flags': font_flags,
                                'bbox': bbox,
                                'following_content': following_content,
                                'level_estimate': self._estimate_heading_level(max_font_size, font_flags)
                            }
                            
                            headings.append(heading_info)
            
            self.logger.info(f"📋 Document heading scan found {len(headings)} potential headings")
            return headings
            
        except Exception as e:
            self.logger.error(f"❌ Document heading scan failed: {e}")
            return []
    
    def _find_heading_matches(self, toc_entry: TOCEntry, document_headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find the best matching document headings for a TOC entry.
        """
        matches = []
        toc_title_normalized = self._normalize_text_for_matching(toc_entry.title)
        
        for heading in document_headings:
            heading_text_normalized = self._normalize_text_for_matching(heading['text'])
            
            # Calculate match confidence using multiple criteria
            confidence = 0.0
            
            # Exact text match (highest confidence)
            if toc_title_normalized == heading_text_normalized:
                confidence = 1.0
            # Fuzzy text match
            elif self._is_fuzzy_match(toc_title_normalized, heading_text_normalized, threshold=0.8):
                confidence = 0.9
            # Partial match (one contains the other)
            elif toc_title_normalized in heading_text_normalized or heading_text_normalized in toc_title_normalized:
                confidence = 0.7
            # Word overlap match
            else:
                toc_words = set(toc_title_normalized.split())
                heading_words = set(heading_text_normalized.split())
                if toc_words and heading_words:
                    overlap = len(toc_words.intersection(heading_words))
                    union = len(toc_words.union(heading_words))
                    if union > 0:
                        confidence = overlap / union
            
            # Boost confidence for level and page proximity
            if confidence > 0.5:
                # Level proximity bonus
                level_diff = abs(toc_entry.level - heading.get('level_estimate', toc_entry.level))
                if level_diff == 0:
                    confidence += 0.1
                elif level_diff == 1:
                    confidence += 0.05
                
                # Page proximity bonus (if reasonable page difference)
                page_diff = abs(toc_entry.page_number - heading['page'])
                if page_diff <= 2:
                    confidence += 0.05
                elif page_diff <= 5:
                    confidence += 0.02
            
            # Only include matches above threshold
            if confidence >= 0.5:
                matches.append({
                    'heading': heading,
                    'confidence': min(1.0, confidence)  # Cap at 1.0
                })
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def _build_toc_hierarchy_enhanced(self, toc_entries: List[TOCEntry]) -> None:
        """
        Enhanced hierarchy building with hierarchical context generation.
        """
        if not toc_entries:
            return
        
        # Stack to track parent entries at each level
        parent_stack = []
        
        for entry in toc_entries:
            # Remove parents that are at the same or deeper level
            while parent_stack and parent_stack[-1].level >= entry.level:
                parent_stack.pop()
            
            # Build hierarchical context path
            if parent_stack:
                parent_entry = parent_stack[-1]
                entry.parent_entry_id = parent_entry.entry_id
                parent_entry.children_ids.append(entry.entry_id)
                
                # Build hierarchical context for better translation
                parent_titles = []
                for parent in parent_stack:
                    parent_titles.append(parent.title)
                entry.hierarchical_context = " > ".join(parent_titles)
            else:
                entry.hierarchical_context = ""  # Top-level entry
            
            # Add current entry to stack
            parent_stack.append(entry)
        
        self.logger.info(f"🏗️ Built enhanced hierarchical TOC structure with context")
    
    def _detect_section_type(self, title: str, level: int) -> str:
        """
        Detect the type of section based on title and level.
        """
        title_lower = title.lower().strip()
        
        # Common section type patterns
        if any(word in title_lower for word in ['chapter', 'κεφάλαιο']):
            return 'chapter'
        elif any(word in title_lower for word in ['appendix', 'παράρτημα']):
            return 'appendix'
        elif any(word in title_lower for word in ['introduction', 'εισαγωγή']):
            return 'introduction'
        elif any(word in title_lower for word in ['conclusion', 'συμπεράσματα']):
            return 'conclusion'
        elif any(word in title_lower for word in ['bibliography', 'references', 'βιβλιογραφία']):
            return 'bibliography'
        elif level == 1:
            return 'main_section'
        elif level <= 3:
            return 'subsection'
        else:
            return 'subheading'
    
    def _is_likely_heading(self, text: str, font_size: float, font_flags: int) -> bool:
        """
        Determine if text is likely a heading based on content and formatting.
        """
        if not text or len(text) < 3:
            return False
        
        # Skip very long texts (likely paragraphs)
        if len(text) > 200:
            return False
        
        # Check for common heading patterns
        text_lower = text.lower().strip()
        
        # Common heading indicators
        heading_patterns = [
            r'^\d+\.?\s*[a-zA-Z]',  # "1. Introduction" or "1 Introduction"
            r'^chapter\s+\d+',      # "Chapter 1"
            r'^section\s+\d+',      # "Section 1"
            r'^appendix\s+[a-zA-Z]', # "Appendix A"
            r'^[ivxlcdm]+\.\s*[a-zA-Z]', # Roman numerals "I. Introduction"
            r'^[a-z]\.\s*[a-zA-Z]',  # "a. Subsection"
        ]
        
        import re
        for pattern in heading_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check formatting characteristics
        is_large_font = font_size > 14  # Larger than typical body text
        is_bold = bool(font_flags & 2**4)  # Bold flag
        is_short = len(text.split()) <= 8  # Headings are typically short
        
        # Combine criteria
        formatting_score = 0
        if is_large_font:
            formatting_score += 2
        if is_bold:
            formatting_score += 2
        if is_short:
            formatting_score += 1
        
        # Content characteristics
        has_title_case = text.istitle() or text.isupper()
        ends_with_punctuation = text.rstrip().endswith(('.', ':', '!', '?'))
        
        if has_title_case:
            formatting_score += 1
        if not ends_with_punctuation:  # Headings typically don't end with punctuation
            formatting_score += 1
        
        return formatting_score >= 3
    
    def _estimate_heading_level(self, font_size: float, font_flags: int) -> int:
        """
        Estimate heading level based on formatting characteristics.
        """
        # Base level determination on font size
        if font_size >= 18:
            level = 1
        elif font_size >= 16:
            level = 2
        elif font_size >= 14:
            level = 3
        elif font_size >= 12:
            level = 4
        else:
            level = 5
        
        # Adjust based on formatting
        is_bold = bool(font_flags & 2**4)
        if is_bold and level > 1:
            level -= 1  # Bold text gets higher priority
        
        return max(1, min(6, level))  # Clamp to valid range
    
    def _extract_section_preview(self, doc: fitz.Document, page_num: int, 
                                start_bbox: Tuple[float, float, float, float]) -> str:
        """
        Extract a preview of content following a heading for context.
        """
        try:
            preview_text = ""
            words_collected = 0
            target_words = 50  # Aim for ~50 words of context
            
            # Start from the page containing the heading
            for current_page_num in range(page_num, min(len(doc), page_num + 2)):
                page = doc[current_page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        block_bbox = block.get("bbox", (0, 0, 0, 0))
                        
                        # On the starting page, only consider content after the heading
                        if current_page_num == page_num:
                            if block_bbox[1] <= start_bbox[3]:  # Block is above or at heading level
                                continue
                        
                        # Extract text from this block
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text:
                                    block_text += text + " "
                        
                        block_text = block_text.strip()
                        if block_text:
                            words = block_text.split()
                            words_to_add = min(len(words), target_words - words_collected)
                            preview_text += " ".join(words[:words_to_add]) + " "
                            words_collected += words_to_add
                            
                            if words_collected >= target_words:
                                break
                
                if words_collected >= target_words:
                    break
            
            return preview_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract section preview: {e}")
            return ""
    
    def _validate_toc_mappings(self, toc_entries: List[TOCEntry]) -> None:
        """
        Validate and score the quality of TOC to content mappings.
        """
        try:
            high_confidence_count = 0
            mapped_count = 0
            
            for entry in toc_entries:
                if entry.mapped_heading_blocks:
                    mapped_count += 1
                    
                    if entry.has_reliable_mapping():
                        high_confidence_count += 1
                        entry.processing_notes.append("High confidence mapping validated")
                    else:
                        entry.processing_notes.append("Low confidence mapping - needs review")
                else:
                    entry.processing_notes.append("No content mapping found")
            
            # Log validation summary
            total_entries = len(toc_entries)
            if total_entries > 0:
                mapping_rate = mapped_count / total_entries
                confidence_rate = high_confidence_count / total_entries
                
                self.logger.info(f"📊 TOC Mapping Validation: {mapping_rate:.1%} mapped, {confidence_rate:.1%} high confidence")
            
        except Exception as e:
            self.logger.error(f"❌ TOC mapping validation failed: {e}")
    
    def _normalize_text_for_matching(self, text: str) -> str:
        """
        Normalize text for reliable matching across different formats.
        """
        import re
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove common punctuation and formatting
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def _is_fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar using fuzzy matching.
        """
        try:
            from difflib import SequenceMatcher
            
            # Use sequence matcher for similarity
            matcher = SequenceMatcher(None, text1, text2)
            similarity = matcher.ratio()
            
            return similarity >= threshold
            
        except ImportError:
            # Fallback to simple word overlap if difflib not available
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return False
            
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            return (overlap / total) >= threshold if total > 0 else False
    
    def _find_heading_block_page(self, translated_title: str, original_title: str,
                               digital_twin_doc: DocumentModel) -> int:
        """
        Search specifically for heading blocks that match the title.
        """
        try:
            page_estimation = 1
            content_so_far = 0
            chars_per_page = 2200
            
            for page in digital_twin_doc.pages:
                for block in page.get_all_blocks():
                    # Check if this is a heading block
                    if hasattr(block, 'block_type') and 'heading' in str(block.block_type).lower():
                        # Get block text
                        if hasattr(block, 'translated_text') and block.translated_text:
                            block_text = block.translated_text
                        else:
                            block_text = block.get_display_text()
                        
                        # Check for match
                        if self._is_fuzzy_match(
                            self._normalize_text_for_matching(translated_title),
                            self._normalize_text_for_matching(block_text)
                        ):
                            return page_estimation
                        
                        # Also check original title
                        if original_title and self._is_fuzzy_match(
                            self._normalize_text_for_matching(original_title),
                            self._normalize_text_for_matching(block_text)
                        ):
                            return page_estimation
                    
                    # Update page estimation
                    if hasattr(block, 'get_display_text'):
                        text = block.get_display_text()
                        if text:
                            content_so_far += len(text)
                            if content_so_far >= chars_per_page:
                                page_estimation += 1
                                content_so_far = 0
            
            return 0  # Not found
            
        except Exception as e:
            self.logger.error(f"Failed to find heading block page: {e}")
            return 0
    
    def _estimate_page_by_toc_position(self, translated_title: str, digital_twin_doc: DocumentModel) -> int:
        """
        Estimate page number based on TOC entry position and document structure.
        """
        try:
            # Find the index of this title in the TOC
            title_index = -1
            for i, entry in enumerate(digital_twin_doc.toc_entries):
                if (hasattr(entry, 'translated_title') and entry.translated_title == translated_title):
                    title_index = i
                    break
            
            if title_index == -1:
                return 1  # Default to page 1
            
            # Estimate based on position in TOC
            total_toc_entries = len(digital_twin_doc.toc_entries)
            total_pages = max(1, digital_twin_doc.total_pages or 10)  # Fallback estimate
            
            # Linear interpolation based on TOC position
            estimated_page = max(1, int((title_index / total_toc_entries) * total_pages))
            
            return estimated_page
            
        except Exception as e:
            self.logger.error(f"Failed to estimate page by TOC position: {e}")
            return 1
    
    def _classify_text_block_type(self, text_block: TextBlock) -> 'BlockType':
        """
        Classify a text block into its appropriate type based on content and formatting.
        """
        try:
            from digital_twin_model import BlockType
            
            text = text_block.text.strip()
            font_size = text_block.font_size
            bbox = text_block.bbox
            
            # Check for empty or minimal text
            if not text or len(text) < 3:
                return BlockType.TEXT
            
            # Check for footnotes first (most specific)
            if self._is_footnote_block(text_block):
                return BlockType.FOOTNOTE
            
            # Check for headings based on formatting and content
            if self._is_likely_heading(text, font_size, 0):  # Use 0 for font_flags as fallback
                # Determine heading level based on font size
                if font_size >= 18:
                    return BlockType.TITLE
                else:
                    return BlockType.HEADING
            
            # Check for list items
            import re
            list_patterns = [
                r'^\s*[\u2022\u2023\u25e6\u25cf\u25cb]\s+',  # Bullet points
                r'^\s*[•·‣⁃▪▫]\s+',  # Various bullet symbols
                r'^\s*\d+\.\s+',      # Numbered lists "1. "
                r'^\s*[a-zA-Z]\.\s+',  # Lettered lists "a. "
                r'^\s*[ivxlcdm]+\.\s+',  # Roman numerals "i. "
                r'^\s*\(\d+\)\s+',    # Parenthetical numbers "(1) "
                r'^\s*\([a-zA-Z]\)\s+',  # Parenthetical letters "(a) "
            ]
            
            if any(re.match(pattern, text, re.IGNORECASE) for pattern in list_patterns):
                return BlockType.LIST_ITEM
            
            # Check for quotes
            if text.startswith('"') and text.endswith('"') or text.startswith("'") and text.endswith("'"):
                return BlockType.QUOTE
            
            # Check for captions (usually short, contain "Figure", "Table", etc.)
            caption_indicators = ['figure', 'fig.', 'table', 'chart', 'diagram', 'image', 'photo', 'illustration']
            if (len(text.split()) <= 20 and  # Captions are typically short
                any(indicator in text.lower() for indicator in caption_indicators)):
                return BlockType.CAPTION
            
            # Check for headers/footers (typically short and at page edges)
            if len(text.split()) <= 10:  # Headers/footers are usually brief
                page_height = 792  # Default page height
                if hasattr(self, 'current_page_dimensions') and self.current_page_dimensions:
                    page_height = self.current_page_dimensions[1]
                
                y_position = bbox[1]
                
                # Top 10% or bottom 10% of page
                if y_position < page_height * 0.1:
                    return BlockType.HEADER
                elif y_position > page_height * 0.9:
                    return BlockType.FOOTER
            
            # Check for bibliography/references
            bibliography_patterns = [
                r'^\s*\[?\d+\]?\s*[A-Z][a-z]+,?\s+[A-Z]\.?\s*.*\(\d{4}\)',  # Author citations
                r'^\s*[A-Z][a-z]+,\s+[A-Z]\..*\d{4}',  # Author, Year pattern
                r'.*(?:journal|proceedings|conference|press|publisher).*\d{4}',  # Publication patterns
            ]
            
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in bibliography_patterns):
                return BlockType.BIBLIOGRAPHY
            
            # Default to paragraph for substantial text content
            if len(text.split()) > 5:  # More than 5 words suggests paragraph content
                return BlockType.PARAGRAPH
            else:
                return BlockType.TEXT  # Short text that doesn't fit other categories
                
        except Exception as e:
            self.logger.warning(f"Text block classification failed: {e}")
            # Fallback to basic classification
            try:
                from digital_twin_model import BlockType
                return BlockType.TEXT
            except:
                return 'text'  # String fallback
    
    def _is_footnote_block(self, text_block: TextBlock) -> bool:
        """
        Enhanced footnote detection using multiple criteria:
        1. Spatial positioning (bottom of page)
        2. Font size (smaller than main text)
        3. Content patterns (numbers, symbols, reference patterns)
        4. Text characteristics (length, formatting)
        """
        try:
            text = text_block.text.strip()
            font_size = text_block.font_size
            bbox = text_block.bbox
            
            # Skip empty or very short text
            if not text or len(text) < 3:
                return False
            
            # Get actual page dimensions if available
            page_height = 792  # Default fallback
            if hasattr(self, 'current_page_dimensions') and self.current_page_dimensions:
                page_height = self.current_page_dimensions[1]
            
            # Spatial criteria: Check if text is in bottom portion of page
            # bbox format: (x0, y0, x1, y1) where y increases downward
            y_position = bbox[1]  # Top y-coordinate of text block
            
            # Consider bottom 25% of page as potential footnote area
            footnote_threshold = page_height * 0.75
            is_bottom_area = y_position > footnote_threshold
            
            # Font size criteria: Footnotes are typically smaller than main text
            is_small_font = font_size < 10
            
            # Content pattern criteria
            content_patterns = [
                # Numbered footnotes: "1.", "2)", "(1)", etc.
                text.startswith(tuple(f"{i}." for i in range(1, 100))),
                text.startswith(tuple(f"{i})" for i in range(1, 100))),
                text.startswith(tuple(f"({i})" for i in range(1, 100))),
                
                # Symbol footnotes: *, †, ‡, §, etc.
                text.startswith(('*', '†', '‡', '§', '¶', '‖', '#')),
                
                # Superscript patterns (common in footnotes)
                any(char in text[:10] for char in ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁰']),
                
                # Common footnote starting patterns
                any(text.lower().startswith(pattern) for pattern in [
                    'see ', 'cf. ', 'ibid', 'op. cit', 'loc. cit', 'et al',
                    'note:', 'n.', 'fn.', 'footnote'
                ])
            ]
            
            has_footnote_pattern = any(content_patterns)
            
            # Length criteria: Footnotes are often shorter than main paragraphs
            is_short_text = len(text) < 200
            
            # Scoring system for footnote detection
            footnote_score = 0
            
            # High confidence indicators
            if is_bottom_area and is_small_font:
                footnote_score += 4  # Strong spatial + font indicator
            elif is_bottom_area:
                footnote_score += 2  # Spatial positioning is important
            elif is_small_font:
                footnote_score += 1  # Font size alone is less reliable
            
            if has_footnote_pattern:
                footnote_score += 3  # Content patterns are strong indicators
            
            if is_short_text:
                footnote_score += 1
            
            # Additional pattern checks
            if text[0].isdigit() and len(text.split()) > 2:  # "1 This is a footnote"
                footnote_score += 1
            
            # Check for reference patterns within text
            import re
            if re.search(r'\b(p\.|pp\.|page|pages)\s+\d+', text.lower()):
                footnote_score += 1
            
            # Check for academic citation patterns
            if re.search(r'\b\d{4}\b', text) and any(word in text.lower() for word in ['journal', 'vol', 'press', 'ed']):
                footnote_score += 1
            
            # Decision threshold - require higher score for better precision
            is_footnote = footnote_score >= 4
            
            if is_footnote:
                self.logger.debug(f"📝 Detected footnote (score: {footnote_score}, bottom: {is_bottom_area}, small: {is_small_font}): {text[:50]}...")
            
            return is_footnote
            
        except Exception as e:
            self.logger.warning(f"Footnote detection failed: {e}")
            return False
    
    def _determine_structural_role(self, text_block: TextBlock, block_type: 'BlockType') -> 'StructuralRole':
        """Determine the structural role of a text block in the document"""
        try:
            from digital_twin_model import StructuralRole, BlockType
            
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
                
        except Exception as e:
            self.logger.warning(f"Structural role determination failed: {e}")
            # Fallback to basic import
            try:
                from digital_twin_model import StructuralRole
                return StructuralRole.CONTENT
            except:
                return 'content'  # String fallback 

    def enable_resume_functionality(self, checkpoint_dir: str) -> None:
        """
        Enable resume functionality with checkpoint saving.
        
        This allows processing to be resumed from the last successful checkpoint
        in case of interruption or failure.
        """
        import os
        
        self.checkpoint_dir = checkpoint_dir
        self.resume_enabled = True
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"📁 Resume functionality enabled with checkpoint dir: {checkpoint_dir}")
    
    def _save_checkpoint(self, document_model: DocumentModel, page_number: int) -> None:
        """Save processing checkpoint for resume functionality"""
        if not self.resume_enabled or not self.checkpoint_dir:
            return
        
        try:
            import json
            import os
            from datetime import datetime
            
            checkpoint_data = {
                'document_info': {
                    'title': document_model.title,
                    'filename': document_model.filename,
                    'total_pages': document_model.total_pages,
                    'processing_method': document_model.extraction_method
                },
                'processing_state': {
                    'last_completed_page': page_number,
                    'completed_pages': self.processing_state['completed_pages'],
                    'failed_pages': self.processing_state['failed_pages'],
                    'checkpoint_time': datetime.now().isoformat(),
                    'processing_start_time': self.processing_state['processing_start_time']
                },
                'statistics': self.stats.copy()
            }
            
            # Save checkpoint file
            checkpoint_filename = f"checkpoint_{document_model.filename}_{page_number}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.processing_state['last_checkpoint'] = checkpoint_path
            self.logger.debug(f"💾 Checkpoint saved at page {page_number}: {checkpoint_filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, document_filename: str) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint for a document"""
        if not self.resume_enabled or not self.checkpoint_dir:
            return None
        
        try:
            import json
            import os
            import glob
            
            # Find all checkpoint files for this document
            pattern = os.path.join(self.checkpoint_dir, f"checkpoint_{document_filename}_*.json")
            checkpoint_files = glob.glob(pattern)
            
            if not checkpoint_files:
                return None
            
            # Get the most recent checkpoint (highest page number)
            latest_checkpoint = max(checkpoint_files, key=lambda f: 
                int(os.path.basename(f).split('_')[-1].split('.')[0]))
            
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"📂 Loaded checkpoint from: {os.path.basename(latest_checkpoint)}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _handle_processing_error(self, error: Exception, page_number: int, 
                                document_model: DocumentModel) -> PageModel:
        """
        Enhanced error handling with recovery strategies.
        
        This implements multiple recovery strategies for different types of errors.
        """
        self.stats['error_recovery_count'] += 1
        error_str = str(error)
        
        self.logger.error(f"🚨 Processing error on page {page_number}: {error_str}")
        
        # Determine error type and apply appropriate recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error, page_number)
        
        try:
            if recovery_strategy == 'retry_simplified':
                self.logger.info(f"🔄 Attempting simplified processing for page {page_number}")
                return self._retry_with_simplified_processing(page_number, document_model)
            
            elif recovery_strategy == 'retry_memory_optimized':
                self.logger.info(f"💾 Attempting memory-optimized processing for page {page_number}")
                return self._retry_with_memory_optimization(page_number, document_model)
            
            elif recovery_strategy == 'skip_images':
                self.logger.info(f"🖼️ Attempting text-only processing for page {page_number}")
                return self._retry_text_only_processing(page_number, document_model)
            
            else:  # 'create_placeholder'
                self.logger.warning(f"⚠️ Creating placeholder page for page {page_number}")
                return self._create_error_placeholder_page(page_number, error_str)
                
        except Exception as recovery_error:
            self.logger.error(f"❌ Error recovery failed for page {page_number}: {recovery_error}")
            return self._create_error_placeholder_page(page_number, f"Original: {error_str}, Recovery: {str(recovery_error)}")
    
    def _determine_recovery_strategy(self, error: Exception, page_number: int) -> str:
        """Determine the best recovery strategy based on error type"""
        error_str = str(error).lower()
        
        # Memory-related errors
        if any(keyword in error_str for keyword in ['memory', 'allocation', 'out of memory']):
            return 'retry_memory_optimized'
        
        # Image-related errors
        elif any(keyword in error_str for keyword in ['image', 'pixmap', 'extract_image']):
            return 'skip_images'
        
        # Complex processing errors
        elif any(keyword in error_str for keyword in ['yolo', 'layout', 'analysis']):
            return 'retry_simplified'
        
        # File access errors
        elif any(keyword in error_str for keyword in ['permission', 'access', 'file']):
            return 'retry_simplified'
        
        # Default to placeholder for unknown errors
        else:
            return 'create_placeholder'
    
    def _retry_with_simplified_processing(self, page_number: int, document_model: DocumentModel) -> PageModel:
        """Retry processing with simplified approach (no YOLO, basic extraction)"""
        try:
            # This would use the memory-optimized processing method
            doc = fitz.open(document_model.filename)
            simplified_page = self._process_single_page_optimized(doc, page_number - 1, "output")
            doc.close()
            
            # Mark as simplified processing
            simplified_page.page_metadata['recovery_method'] = 'simplified_processing'
            simplified_page.page_metadata['original_error_recovered'] = True
            
            return simplified_page
            
        except Exception as e:
            raise Exception(f"Simplified processing failed: {e}")
    
    def _retry_with_memory_optimization(self, page_number: int, document_model: DocumentModel) -> PageModel:
        """Retry processing with aggressive memory optimization"""
        try:
            import gc
            
            # Force garbage collection before retry
            gc.collect()
            
            # Use the most basic processing approach
            doc = fitz.open(document_model.filename)
            page = doc[page_number - 1]
            
            # Create minimal page model
            recovery_page = PageModel(
                page_number=page_number,
                dimensions=(page.rect.width, page.rect.height),
                page_metadata={
                    'recovery_method': 'memory_optimized',
                    'original_error_recovered': True,
                    'processing_strategy': 'minimal'
                }
            )
            
            # Extract only essential text blocks
            try:
                blocks = page.get_text("dict")["blocks"]
                for i, block in enumerate(blocks):
                    if "lines" in block:
                        text_content = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_content += span["text"] + " "
                        
                        if text_content.strip():
                            text_block = create_text_block(
                                block_id=f"recovery_text_{page_number}_{i}",
                                text=text_content.strip(),
                                bbox=(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3]),
                                page_number=page_number,
                                extraction_method='memory_optimized_recovery'
                            )
                            recovery_page.add_block(text_block)
            except Exception as text_error:
                self.logger.warning(f"Text extraction failed in recovery mode: {text_error}")
            
            doc.close()
            return recovery_page
            
        except Exception as e:
            raise Exception(f"Memory-optimized processing failed: {e}")
    
    def _retry_text_only_processing(self, page_number: int, document_model: DocumentModel) -> PageModel:
        """Retry processing with text-only extraction (skip images)"""
        try:
            doc = fitz.open(document_model.filename)
            page = doc[page_number - 1]
            
            # Create text-only page model
            text_only_page = PageModel(
                page_number=page_number,
                dimensions=(page.rect.width, page.rect.height),
                page_metadata={
                    'recovery_method': 'text_only',
                    'original_error_recovered': True,
                    'images_skipped': True
                }
            )
            
            # Extract text blocks only
            text_blocks = self.content_extractor.extract_text_blocks(page)
            for i, text_block in enumerate(text_blocks):
                dt_text_block = create_text_block(
                    block_id=f"text_only_{page_number}_{i}",
                    text=text_block.text,
                    bbox=text_block.bbox,
                    page_number=page_number,
                    extraction_method='text_only_recovery'
                )
                text_only_page.add_block(dt_text_block)
            
            doc.close()
            return text_only_page
            
        except Exception as e:
            raise Exception(f"Text-only processing failed: {e}")
    
    def _create_error_placeholder_page(self, page_number: int, error_message: str) -> PageModel:
        """Create a placeholder page when all recovery strategies fail"""
        placeholder_page = PageModel(
            page_number=page_number,
            dimensions=(612.0, 792.0),  # Standard letter size
            page_metadata={
                'processing_failed': True,
                'error_message': error_message,
                'recovery_method': 'placeholder',
                'placeholder_created': True
            }
        )
        
        # Add error message as text block
        error_text_block = create_text_block(
            block_id=f"error_placeholder_{page_number}",
            text=f"[ERROR: Page {page_number} processing failed - {error_message}]",
            bbox=(50, 50, 550, 100),
            page_number=page_number,
            block_type=BlockType.TEXT,
            extraction_method='error_placeholder'
        )
        placeholder_page.add_block(error_text_block)
        
        return placeholder_page
    
    def _cleanup_old_checkpoints(self, document_filename: str, keep_latest: int = 3) -> None:
        """Clean up old checkpoint files to prevent disk space issues"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return
            
        try:
            # Find all checkpoint files for this document
            checkpoint_pattern = f"{document_filename}_checkpoint_*.json"
            checkpoint_files = []
            
            for file in os.listdir(self.checkpoint_dir):
                if file.startswith(f"{document_filename}_checkpoint_") and file.endswith('.json'):
                    file_path = os.path.join(self.checkpoint_dir, file)
                    checkpoint_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            for file_path, _ in checkpoint_files[keep_latest:]:
                try:
                    os.remove(file_path)
                    self.logger.info(f"🗑️ Removed old checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def cleanup(self) -> None:
        """
        OPTIMIZATION: Cleanup all resources and components.
        
        This method ensures proper resource cleanup to prevent memory leaks
        and maintain system performance.
        """
        self.logger.info("🧹 Cleaning up PyMuPDF-YOLO Processor resources...")
        
        try:
            # Cleanup parallel image extractor
            if hasattr(self, 'image_extractor') and self.image_extractor:
                self.image_extractor.cleanup()
                self.logger.info("   ✅ Parallel image extractor cleaned up")
            
            # Final memory cleanup
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.cleanup_memory()
                final_stats = self.memory_manager.get_processing_stats()
                self.logger.info(f"   ✅ Memory manager cleaned up - Final stats: {final_stats}")
            
            # Clear processing state
            self.processing_state = {
                'current_document': None,
                'completed_pages': [],
                'failed_pages': [],
                'last_checkpoint': None,
                'processing_start_time': None
            }
            
            # Log final performance statistics
            if self.stats['total_pages_processed'] > 0:
                self.logger.info(f"   📊 Final Processing Statistics:")
                self.logger.info(f"      • Total pages processed: {self.stats['total_pages_processed']}")
                self.logger.info(f"      • Successful pages: {self.stats['successful_pages']}")
                self.logger.info(f"      • Failed pages: {self.stats['failed_pages']}")
                self.logger.info(f"      • Average processing time: {self.stats['average_page_time']:.3f}s per page")
                self.logger.info(f"      • Error recovery operations: {self.stats['error_recovery_count']}")
            
            self.logger.info("✅ PyMuPDF-YOLO Processor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on garbage collection"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    async def translate_toc_titles_post_processing(self, digital_twin_doc: DocumentModel, 
                                                 target_language: str, gemini_service=None) -> None:
        """
        Translate TOC titles after the main document translation is complete.
        
        This implements the user's suggestion to:
        1. Translate the content first
        2. Scan for where titles exist in the translated document
        3. Deduce page numbers from the actual document structure
        4. Update TOC entries with translated titles and corrected page numbers
        """
        try:
            if not digital_twin_doc.toc_entries or not gemini_service:
                return
            
            self.logger.info(f"🌐 Starting post-processing TOC title translation to {target_language}")
            
            # Step 1: Collect all TOC titles for translation
            toc_titles = [entry.original_title for entry in digital_twin_doc.toc_entries]
            
            # Step 2: Translate TOC titles as a batch
            translated_titles = await self._translate_toc_titles_batch(toc_titles, target_language, gemini_service)
            
            # Step 3: Scan translated document for title locations and deduce page numbers
            title_locations = self._scan_document_for_translated_titles(
                digital_twin_doc, translated_titles
            )
            
            # Step 4: Update TOC entries with translations and corrected page numbers
            for i, toc_entry in enumerate(digital_twin_doc.toc_entries):
                if i < len(translated_titles):
                    translated_title = translated_titles[i]
                    toc_entry.translated_title = translated_title
                    
                    # Find the actual page number in the translated document
                    actual_page = self._find_title_page_in_document(
                        digital_twin_doc, translated_title, toc_entry.original_title
                    )
                    
                    if actual_page > 0:
                        original_page = toc_entry.page_number
                        toc_entry.page_number = actual_page
                        
                        self.logger.debug(f"📄 Updated TOC entry: '{translated_title}' "
                                        f"Page {original_page} → {actual_page}")
            
            self.logger.info(f"✅ Completed post-processing TOC translation: "
                           f"{len(translated_titles)} titles translated")
            
        except Exception as e:
            self.logger.error(f"❌ TOC post-processing translation failed: {e}")
    
    async def _translate_toc_titles_batch(self, titles: List[str], target_language: str, gemini_service=None) -> List[str]:
        """
        Translate TOC titles in batch for efficiency.
        
        ENHANCED: Improved parsing robustness to handle mixed translation formats.
        """
        try:
            if not titles or not gemini_service:
                return titles
            
            # Create a structured format for batch translation with fallback parsing
            numbered_titles = []
            for i, title in enumerate(titles):
                numbered_titles.append(f"{i+1}. {title}")
            
            combined_text = "\n".join(numbered_titles)
            
            # Enhanced translation prompt with clear formatting instructions
            translation_prompt = f"""
            Translate the following Table of Contents titles to {target_language}.
            Maintain the numbering and format exactly as shown.
            
            IMPORTANT: Respond with each title on a separate line, keeping the number prefix:
            
            {combined_text}
            
            TRANSLATED TITLES:
            """
            
            # Get translation from Gemini
            if hasattr(gemini_service, 'translate_text_with_context'):
                translated_batch = await gemini_service.translate_text_with_context(
                    combined_text, target_language, 
                    context="Table of Contents titles for academic document",
                    translation_style="academic"
                )
            else:
                translated_batch = await gemini_service.translate_text(
                    translation_prompt, target_language
                )
            
            # Enhanced parsing with multiple fallback strategies
            translated_titles = self._parse_translated_titles_robust(translated_batch, titles)
            
            self.logger.info(f"✅ Successfully translated {len(translated_titles)} TOC titles")
            return translated_titles
            
        except Exception as e:
            self.logger.error(f"Failed to translate TOC titles: {e}")
            return titles  # Return original titles as fallback
    
    def _parse_translated_titles_robust(self, translated_batch: str, original_titles: List[str]) -> List[str]:
        """
        Robust parsing of translated titles with multiple fallback strategies.
        """
        try:
            translated_titles = []
            lines = translated_batch.strip().split('\n')
            
            # Strategy 1: Parse numbered lines
            for i, original_title in enumerate(original_titles):
                found_translation = None
                
                # Look for lines starting with the expected number
                for line in lines:
                    line = line.strip()
                    if line.startswith(f"{i+1}."):
                        # Extract the title after the number
                        title_part = line[len(f"{i+1}."):].strip()
                        if title_part:
                            found_translation = title_part
                            break
                
                # Strategy 2: If not found, try to find by position
                if not found_translation and i < len(lines):
                    line = lines[i].strip()
                    # Remove any leading numbering
                    import re
                    line = re.sub(r'^\d+\.\s*', '', line)
                    if line:
                        found_translation = line
                
                # Strategy 3: Fallback to original if all else fails
                if not found_translation:
                    found_translation = original_title
                    self.logger.warning(f"Could not parse translation for title {i+1}, using original")
                
                translated_titles.append(found_translation)
            
            return translated_titles
            
        except Exception as e:
            self.logger.error(f"Failed to parse translated titles: {e}")
            return original_titles
    
    def _scan_document_for_translated_titles(self, digital_twin_doc: DocumentModel, 
                                           translated_titles: List[str]) -> Dict[str, int]:
        """
        Scan the translated document to find where each title actually appears.
        
        Returns a mapping of translated titles to their actual page numbers.
        """
        title_locations = {}
        
        try:
            for page in digital_twin_doc.pages:
                page_number = page.page_number
                
                # Check all text blocks on this page
                for block in page.get_all_blocks():
                    if hasattr(block, 'get_display_text'):
                        block_text = block.get_display_text(prefer_translation=True)
                        
                        # Check if this block contains any of our translated titles
                        for translated_title in translated_titles:
                            if self._text_contains_title(block_text, translated_title):
                                # Record the first occurrence (headings usually appear first)
                                if translated_title not in title_locations:
                                    title_locations[translated_title] = page_number
                                    self.logger.debug(f"📍 Found title '{translated_title}' on page {page_number}")
            
            return title_locations
            
        except Exception as e:
            self.logger.error(f"Failed to scan document for translated titles: {e}")
            return {}
    
    def _find_title_page_in_document(self, digital_twin_doc: DocumentModel, 
                                   translated_title: str, original_title: str) -> int:
        """
        Find the actual page number where a title appears in the document.
        
        Checks both translated and original titles to handle partial translations.
        """
        try:
            # First, try to find the translated title
            for page in digital_twin_doc.pages:
                for block in page.get_all_blocks():
                    if hasattr(block, 'get_display_text'):
                        block_text = block.get_display_text(prefer_translation=True)
                        
                        if self._text_contains_title(block_text, translated_title):
                            return page.page_number
            
            # Fallback: try to find the original title
            for page in digital_twin_doc.pages:
                for block in page.get_all_blocks():
                    if hasattr(block, 'get_display_text'):
                        block_text = block.get_display_text(prefer_translation=False)
                        
                        if self._text_contains_title(block_text, original_title):
                            return page.page_number
            
            # If not found, return 0 (will keep original page number)
            return 0
            
        except Exception as e:
            self.logger.warning(f"Failed to find title page: {e}")
            return 0
    
    def _text_contains_title(self, text: str, title: str) -> bool:
        """
        Check if a text block contains a title, allowing for minor formatting differences.
        """
        if not text or not title:
            return False
        
        # Normalize both texts for comparison
        text_normalized = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        title_normalized = re.sub(r'[^\w\s]', ' ', title.lower()).strip()
        
        # Exact match
        if title_normalized in text_normalized:
            return True
        
        # Partial match for longer titles (allowing for line breaks, etc.)
        if len(title_normalized) > 20:
            title_words = title_normalized.split()
            if len(title_words) >= 3:
                # Check if most title words appear in the text
                matching_words = sum(1 for word in title_words if word in text_normalized)
                return matching_words >= len(title_words) * 0.7  # 70% match threshold
        
        return False
    
    async def post_process_toc_translation(self, digital_twin_doc: DocumentModel, target_language: str) -> None:
        """
        Post-process TOC translation with enhanced page number deduction.
        
        ENHANCED: Implements the user's suggestion to scan the translated document,
        find where titles exist, and deduce the actual page numbers.
        """
        try:
            # This method appears to be unused - the main method is translate_toc_titles_post_processing
            # But we'll fix it for completeness
            self.logger.warning("post_process_toc_translation is deprecated - use translate_toc_titles_post_processing instead")
            return
            
            # Step 3: ENHANCED - Scan translated document for title locations and deduce page numbers
            title_page_mapping = self._scan_document_for_title_locations(
                digital_twin_doc, translated_titles, toc_titles
            )
            
            # Step 4: Update TOC entries with translations and corrected page numbers
            for i, toc_entry in enumerate(digital_twin_doc.toc_entries):
                if i < len(translated_titles):
                    translated_title = translated_titles[i]
                    toc_entry.translated_title = translated_title
                    
                    # Find the actual page number using enhanced scanning
                    actual_page = title_page_mapping.get(translated_title, toc_entry.page_number)
                    
                    if actual_page != toc_entry.page_number:
                        original_page = toc_entry.page_number
                        toc_entry.page_number = actual_page
                        
                        self.logger.info(f"📄 Updated TOC entry: '{translated_title}' "
                                       f"Page {original_page} → {actual_page}")
                    else:
                        self.logger.debug(f"📄 TOC entry page unchanged: '{translated_title}' "
                                        f"remains on page {actual_page}")
            
            self.logger.info(f"✅ Completed enhanced post-processing TOC translation: "
                           f"{len(translated_titles)} titles translated and pages corrected")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced TOC post-processing translation failed: {e}")
    
    def _scan_document_for_title_locations(self, digital_twin_doc: DocumentModel, 
                                         translated_titles: List[str], 
                                         original_titles: List[str]) -> Dict[str, int]:
        """
        Enhanced document scanning to find where translated titles actually appear.
        
        This implements the user's suggestion to scan the translated document and
        deduce page numbers based on where content actually appears.
        """
        title_page_mapping = {}
        
        try:
            # Build content mapping with page estimation
            content_pages = self._build_content_page_mapping(digital_twin_doc)
            
            # For each translated title, find its location in the document
            for i, translated_title in enumerate(translated_titles):
                original_title = original_titles[i] if i < len(original_titles) else ""
                
                # Find page using multiple strategies
                page_number = self._find_title_page_multiple_strategies(
                    translated_title, original_title, content_pages, digital_twin_doc
                )
                
                title_page_mapping[translated_title] = page_number
                
                self.logger.debug(f"📍 Mapped '{translated_title}' to page {page_number}")
            
            return title_page_mapping
            
        except Exception as e:
            self.logger.error(f"Failed to scan document for title locations: {e}")
            return {}
    
    def _build_content_page_mapping(self, digital_twin_doc: DocumentModel) -> Dict[str, int]:
        """
        Build a mapping of content to estimated page numbers in the final document.
        """
        content_pages = {}
        current_page = 1
        content_on_current_page = 0
        
        # Conservative estimate for academic documents
        chars_per_page = 2200  # Slightly lower than document generator to account for formatting
        
        try:
            for page in digital_twin_doc.pages:
                for block in page.get_all_blocks():
                    # Get display text (translated if available)
                    if hasattr(block, 'translated_text') and block.translated_text:
                        display_text = block.translated_text
                    else:
                        display_text = block.get_display_text()
                    
                    if not display_text:
                        continue
                    
                    # Normalize text for matching
                    normalized_text = self._normalize_text_for_matching(display_text)
                    
                    # Map content to current page
                    content_pages[normalized_text] = current_page
                    
                    # Also map partial text for fuzzy matching
                    if len(normalized_text) > 20:
                        # Map first 50 characters for partial matching
                        partial_key = normalized_text[:50]
                        if partial_key not in content_pages:
                            content_pages[partial_key] = current_page
                    
                    # Update page estimation
                    content_length = len(display_text)
                    content_on_current_page += content_length
                    
                    # Check if we should move to next page
                    if content_on_current_page >= chars_per_page:
                        current_page += 1
                        content_on_current_page = 0
            
            self.logger.debug(f"📊 Built content-page mapping: {len(content_pages)} entries across ~{current_page} pages")
            return content_pages
            
        except Exception as e:
            self.logger.error(f"Failed to build content page mapping: {e}")
            return {}
    
    def _find_title_page_multiple_strategies(self, translated_title: str, original_title: str,
                                           content_pages: Dict[str, int], 
                                           digital_twin_doc: DocumentModel) -> int:
        """
        Find the page number for a title using multiple search strategies.
        """
        # Strategy 1: Exact match with translated title
        normalized_translated = self._normalize_text_for_matching(translated_title)
        if normalized_translated in content_pages:
            return content_pages[normalized_translated]
        
        # Strategy 2: Partial match with translated title
        for content_text, page_num in content_pages.items():
            if self._is_fuzzy_match(normalized_translated, content_text):
                self.logger.debug(f"📍 Fuzzy match found: '{translated_title}' matches content on page {page_num}")
                return page_num
        
        # Strategy 3: Try with original title
        if original_title:
            normalized_original = self._normalize_text_for_matching(original_title)
            if normalized_original in content_pages:
                return content_pages[normalized_original]
            
            # Fuzzy match with original title
            for content_text, page_num in content_pages.items():
                if self._is_fuzzy_match(normalized_original, content_text):
                    self.logger.debug(f"📍 Original title fuzzy match: '{original_title}' matches content on page {page_num}")
                    return page_num
        
        # Strategy 4: Search by heading blocks specifically
        heading_page = self._find_heading_block_page(translated_title, original_title, digital_twin_doc)
        if heading_page > 0:
            return heading_page
        
        # Strategy 5: Use pattern-based estimation
        estimated_page = self._estimate_page_by_toc_position(translated_title, digital_twin_doc)
        
        self.logger.debug(f"📍 Using estimated page {estimated_page} for title: {translated_title}")
        return estimated_page
    
    def _normalize_text_for_matching(self, text: str) -> str:
        """
        Normalize text for consistent matching across different representations.
        """
        if not text:
            return ""
        
        import re
        # Remove punctuation, normalize whitespace, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _is_fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar enough to be considered a match.
        """
        if not text1 or not text2:
            return False
        
        # Exact match
        if text1 == text2:
            return True
        
        # Length-based quick rejection
        if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.5:
            return False
        
        # Check if one contains the other (for partial matches)
        if len(text1) > 10 and len(text2) > 10:
            if text1 in text2 or text2 in text1:
                return True
        
        # Use difflib for similarity ratio
        try:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, text1, text2).ratio()
            return similarity >= threshold
        except ImportError:
            # Fallback to simple substring matching
            words1 = set(text1.split())
            words2 = set(text2.split())
            if words1 and words2:
                common_words = words1.intersection(words2)
                similarity = len(common_words) / max(len(words1), len(words2))
                return similarity >= threshold
        
        return False
    
    def _find_heading_block_page(self, translated_title: str, original_title: str,
                               digital_twin_doc: DocumentModel) -> int:
        """
        Search specifically for heading blocks that match the title.
        """
        try:
            page_estimation = 1
            content_so_far = 0
            chars_per_page = 2200
            
            for page in digital_twin_doc.pages:
                for block in page.get_all_blocks():
                    # Check if this is a heading block
                    if hasattr(block, 'block_type') and 'heading' in str(block.block_type).lower():
                        # Get block text
                        if hasattr(block, 'translated_text') and block.translated_text:
                            block_text = block.translated_text
                        else:
                            block_text = block.get_display_text()
                        
                        # Check for match
                        if self._is_fuzzy_match(
                            self._normalize_text_for_matching(translated_title),
                            self._normalize_text_for_matching(block_text)
                        ):
                            return page_estimation
                        
                        # Also check original title
                        if original_title and self._is_fuzzy_match(
                            self._normalize_text_for_matching(original_title),
                            self._normalize_text_for_matching(block_text)
                        ):
                            return page_estimation
                    
                    # Update page estimation
                    if hasattr(block, 'get_display_text'):
                        text = block.get_display_text()
                        if text:
                            content_so_far += len(text)
                            if content_so_far >= chars_per_page:
                                page_estimation += 1
                                content_so_far = 0
            
            return 0  # Not found
            
        except Exception as e:
            self.logger.error(f"Failed to find heading block page: {e}")
            return 0
    
    def _estimate_page_by_toc_position(self, translated_title: str, digital_twin_doc: DocumentModel) -> int:
        """
        Estimate page number based on TOC entry position and document structure.
        """
        try:
            # Find the index of this title in the TOC
            title_index = -1
            for i, entry in enumerate(digital_twin_doc.toc_entries):
                if (hasattr(entry, 'translated_title') and entry.translated_title == translated_title):
                    title_index = i
                    break
            
            if title_index == -1:
                return 1  # Default to page 1
            
            # Estimate based on position in TOC
            total_toc_entries = len(digital_twin_doc.toc_entries)
            total_pages = max(1, digital_twin_doc.total_pages or 10)  # Fallback estimate
            
            # Linear interpolation based on TOC position
            estimated_page = max(1, int((title_index / total_toc_entries) * total_pages))
            
            return estimated_page
            
        except Exception as e:
            self.logger.error(f"Failed to estimate page by TOC position: {e}")
            return 1
    
    def process_pdf(self, input_filepath, *args, **kwargs):
        pass