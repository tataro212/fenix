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
            
            self.logger.info(f"📄 Extracted {len(text_blocks)} text blocks from page")
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting text blocks: {e}")
            return []
    
    def extract_images(self, page: fitz.Page) -> List[ImageBlock]:
        """Extract native images with coordinates"""
        image_blocks = []
        
        try:
            images = page.get_images()
            
            for img_index, img in enumerate(images):
                bbox = page.get_image_bbox(img)
                
                image_block = ImageBlock(
                    image_index=img_index,
                    bbox=tuple(bbox),
                    block_type='image'
                )
                
                image_blocks.append(image_block)
            
            self.logger.info(f"🖼️ Extracted {len(image_blocks)} image blocks from page")
            return image_blocks
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting images: {e}")
            return []
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a PyMuPDF block"""
        block_text = ""
        
        try:
            for line in block.get("lines", []):
                line_text = ""
                
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                
                if line_text.strip():
                    block_text += line_text + "\n"
            
            return block_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from block: {e}")
            return ""
    
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
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 PyMuPDF-YOLO Processor initialized")
    
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
    
    async def process_page(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Main processing method with intelligent routing - implements user's strategic vision"""
        start_time = time.time()
        
        try:
            # 1. Open PDF and get page
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 2. Quick content scan to determine processing path
            if self._quick_content_scan(page):
                return await self._process_pure_text_fast(page, page_num, start_time)
            else:
                return await self._process_mixed_content_with_coordinates(page, page_num, start_time)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing page {page_num + 1}: {e}", exc_info=True)
            return {
                'error': f"Failed to process page {page_num + 1}: {e}",
                'page_num': page_num
            }
    
    async def _process_pure_text_fast(self, page: fitz.Page, page_num: int, start_time: float) -> Dict[str, Any]:
        """Fast-path processing for pure text pages using only PyMuPDF."""
        try:
            text_blocks = self.content_extractor.extract_text_blocks(page)
            
            # Create a consistent mapped_content structure
            mapped_content = {}
            for i, block in enumerate(text_blocks):
                area_id = f"text_area_{page_num}_{i}"
                layout_area = LayoutArea(
                    label='text',
                    bbox=block.bbox,
                    confidence=block.confidence,
                    area_id=area_id,
                    class_id=0 # Generic text class
                )
                mapped_content[area_id] = MappedContent(
                    layout_info=layout_area,
                    text_blocks=[block],
                    image_blocks=[],
                    combined_text=block.text
                )

            strategy = self.classifier.get_processing_strategy(ContentType.PURE_TEXT, mapped_content)

            self.logger.info(f"⚡ Page {page_num + 1}: Fast text processing completed in {time.time() - start_time:.3f}s")
            
            return {
                'page_num': page_num,
                'processing_time': time.time() - start_time,
                'content_type': ContentType.PURE_TEXT,
                'strategy': strategy,
                'mapped_content': mapped_content, # Consistent output
                'text_blocks': text_blocks,
                'image_blocks': [],
                'layout_areas': list(mapped_content.keys())
            }
        except Exception as e:
            self.logger.error(f"❌ Error in fast text processing for page {page_num + 1}: {e}", exc_info=True)
            return {'error': str(e), 'page_num': page_num}
    
    async def _process_mixed_content_with_coordinates(self, page: fitz.Page, page_num: int, start_time: float) -> Dict[str, Any]:
        """Process mixed-content pages using both PyMuPDF and YOLO, with coordinate-based mapping."""
        try:
            # 1. Extract content with PyMuPDF
            text_blocks = self.content_extractor.extract_text_blocks(page)
            image_blocks = self.content_extractor.extract_images(page)
            
            # 2. Render page for YOLO
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            page_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            
            # 3. Analyze layout with YOLO (0.15 confidence)
            layout_areas = self.layout_analyzer.analyze_layout(page_image)
            
            # 4. Map content to layout using coordinates
            mapped_content = self.content_mapper.map_content_to_layout(
                text_blocks, image_blocks, layout_areas
            )
            
            # 5. Classify content type and get strategy
            content_type = self.classifier.classify_mapped_content(mapped_content)
            strategy = self.classifier.get_processing_strategy(content_type, mapped_content)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"🎯 Page {page_num}: Coordinate-based processing completed in {processing_time:.3f}s")
            self.logger.info(f"   Content type: {content_type.value}")
            self.logger.info(f"   Strategy: {strategy.strategy}")
            self.logger.info(f"   Text blocks: {len(text_blocks)}")
            self.logger.info(f"   Image blocks: {len(image_blocks)}")
            self.logger.info(f"   Layout areas: {len(layout_areas)}")
            
            return {
                'mapped_content': mapped_content,
                'content_type': content_type,
                'strategy': strategy,
                'page_num': page_num,
                'processing_time': processing_time,
                'text_blocks': text_blocks,
                'image_blocks': image_blocks,
                'layout_areas': layout_areas,
                'statistics': {
                    'text_blocks': len(text_blocks),
                    'image_blocks': len(image_blocks),
                    'layout_areas': len(layout_areas),
                    'mapped_areas': len(mapped_content),
                    'processing_path': 'coordinate_based_extraction'
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in coordinate-based processing: {e}")
            raise
    
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