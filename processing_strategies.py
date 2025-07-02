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
    
    async def translate_direct_text(self, text_elements: list[dict], target_language: str) -> list[dict]:
        """
        Translates a list of text elements using the new semantic batching.

        This method joins paragraphs with a unique separator, sends a single large
        request to the API per batch, and then reliably splits the response back
        into structured, translated paragraphs.
        """
        # Use the new, smarter batching method.
        batches = self._create_batches(text_elements)
        
        translated_paragraphs = []
        
        for i, batch_of_elements in enumerate(batches):
            self.logger.info(f"  Translating batch {i+1}/{len(batches)}...")
            
            # Use a unique, robust separator to join and split paragraphs.
            separator = "\n<|=|>\n"
            source_text_for_api = separator.join([elem.get('text', '') for elem in batch_of_elements])

            # Call the translation service with the large, context-rich text block.
            translated_blob = await self.gemini_service.translate_text(source_text_for_api, target_language)
            
            # Split the translated blob back into individual paragraph segments.
            translated_segments = translated_blob.split(separator.strip())
            
            # Re-associate translated segments with original element data (like bounding boxes).
            for original_element, translated_text in zip(batch_of_elements, translated_segments):
                translated_paragraphs.append({
                    'type': 'text',
                    'text': translated_text.strip(),
                    'label': 'paragraph',
                    'bbox': original_element.get('bbox') # Preserve original coordinates
                })
                
        self.logger.info(f"🌐 Direct text translation completed. Created {len(translated_paragraphs)} structured paragraphs.")
        return translated_paragraphs
    
    def _create_batches(self, text_elements: list[dict], max_chars_per_batch: int = 4500) -> list[list[dict]]:
        """
        Creates semantically coherent batches of text elements based on paragraphs.

        This logic prioritizes keeping paragraphs intact. It creates a few large,
        context-rich batches instead of many small, fragmented ones.
        The character limit is set high (4500) to maximize context per API call.
        """
        if not text_elements:
            return []

        batches = []
        current_batch_elements = []
        current_batch_char_count = 0

        for element in text_elements:
            element_text = element.get('text', '')
            element_char_count = len(element_text)

            if current_batch_elements and (current_batch_char_count + element_char_count > max_chars_per_batch):
                batches.append(current_batch_elements)
                current_batch_elements = []
                current_batch_char_count = 0
            
            current_batch_elements.append(element)
            current_batch_char_count += element_char_count

        if current_batch_elements:
            batches.append(current_batch_elements)
            
        self.logger.info(f"📦 Created {len(batches)} semantically coherent batches from {len(text_elements)} text elements.")
        return batches


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
        self.minimal_graph_builder = MinimalGraphBuilder(gemini_service)
        self.comprehensive_graph_builder = ComprehensiveGraphBuilder(gemini_service)
        
        self.performance_stats = {
            'pure_text_fast': {'total_time': 0.0, 'count': 0},
            'coordinate_based_extraction': {'total_time': 0.0, 'count': 0},
            'direct_text': {'total_time': 0.0, 'count': 0},
            'minimal_graph': {'total_time': 0.0, 'count': 0},
            'comprehensive_graph': {'total_time': 0.0, 'count': 0}
        }
        
        self.logger.info("🔧 Processing Strategy Executor initialized")
    
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
            return {'translated_text': '', 'blocks': []}
        processing_result = self._ensure_dict_of_areas(processing_result)
        if not processing_result:
            self.logger.error(f"[ERROR] processing_result could not be converted to dict in _process_coordinate_based_extraction: {processing_result}")
            return {'translated_text': '', 'blocks': []}
        start_time = time.time()
        try:
            mapped_content = processing_result.get('mapped_content', {})
            text_areas = []
            non_text_areas = []
            for area_id, area_data in mapped_content.items():
                if hasattr(area_data, 'combined_text') and area_data.combined_text:
                    text_areas.append(area_data)
                else:
                    non_text_areas.append(area_data)
            text_areas.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
            translated_texts = []
            if self.gemini_service:
                batches = self._create_text_batches(text_areas, max_chars_per_batch=500)
                self.logger.info(f"📦 Created {len(batches)} batches from {len(text_areas)} text areas")
                async def translate_batch(batch, batch_index):
                    batch_text = '\n\n'.join([area.combined_text for area in batch])
                    self.logger.info(f"   Translating batch {batch_index + 1}/{len(batches)} ({len(batch_text)} chars)")
                    try:
                        translated_batch = await asyncio.wait_for(
                            self.gemini_service.translate_text(batch_text, target_language, timeout=30.0),
                            timeout=35.0
                        )
                        translated_parts = translated_batch.split('\n\n')
                        if len(translated_parts) != len(batch):
                            translated_parts = translated_batch.split('\n')
                            if len(translated_parts) != len(batch):
                                translated_parts = translated_batch.split('. ')
                                if len(translated_parts) != len(batch):
                                    self.logger.warning(f"⚠️ Batch {batch_index + 1}: Using proportional splitting")
                                    total_chars = sum(len(area.combined_text) for area in batch)
                                    translated_parts = []
                                    current_pos = 0
                                    for area in batch:
                                        area_ratio = len(area.combined_text) / total_chars
                                        area_chars = int(len(translated_batch) * area_ratio)
                                        translated_parts.append(translated_batch[current_pos:current_pos + area_chars])
                                        current_pos += area_chars
                        if len(translated_parts) == len(batch):
                            return translated_parts
                        else:
                            self.logger.warning(f"⚠️ Batch {batch_index + 1}: Translation splitting mismatch, using original")
                            return [area.combined_text for area in batch]
                    except asyncio.TimeoutError:
                        self.logger.error(f"❌ Batch {batch_index + 1} translation timed out after 35s, using original")
                        return [area.combined_text for area in batch]
                    except Exception as e:
                        self.logger.error(f"❌ Batch {batch_index + 1} translation failed: {e}")
                        return [area.combined_text for area in batch]
                batch_tasks = [translate_batch(batch, i) for i, batch in enumerate(batches)]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Batch {i + 1} failed with exception: {result}")
                        batch = batches[i]
                        translated_texts.extend([area.combined_text for area in batch])
                    else:
                        translated_texts.extend(result)
                self.logger.info(f"✅ Completed parallel batch translation: {len(translated_texts)} texts translated")
            else:
                self.logger.warning("⚠️ Gemini service not available for coordinate-based extraction.")
                translated_texts = [area.combined_text for area in text_areas]
            final_content = []
            for i, area in enumerate(text_areas):
                final_content.append({
                    'type': 'text',
                    'original_text': area.combined_text,
                    'translated_text': translated_texts[i] if i < len(translated_texts) else area.combined_text,
                    'layout_info': area.layout_info
                })
            for area in non_text_areas:
                final_content.append({
                    'type': 'visual_element',
                    'layout_info': area.layout_info,
                    'image_blocks': getattr(area, 'image_blocks', [])
                })
            final_content.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            processing_time = time.time() - start_time
            self.performance_stats['coordinate_based_extraction']['total_time'] += processing_time
            self.performance_stats['coordinate_based_extraction']['count'] += 1
            self.logger.info(f"🎯 Coordinate-based extraction completed in {processing_time:.3f}s")
            self.logger.info(f"   Total areas processed: {len(mapped_content)}")
            self.logger.info(f"   Text areas: {len(text_areas)}")
            self.logger.info(f"   Non-text areas: {len(non_text_areas)}")
            self.logger.info(f"   API calls reduced from {len(text_areas)} to {len(batches) if 'batches' in locals() else 0}")
            return ProcessingResult(
                success=True,
                strategy='coordinate_based_extraction',
                processing_time=processing_time,
                content={
                    'text_areas': [{'label': area.label, 'text_content': area.combined_text, 'translated_content': translated_texts[i] if i < len(translated_texts) else area.combined_text} for i, area in enumerate(text_areas)],
                    'non_text_areas': [{'label': area.label, 'bbox': area.bbox} for area in non_text_areas],
                    'coordinate_mapping': mapped_content,
                    'page_num': processing_result.get('page_num', 0)
                },
                statistics={
                    'total_areas': len(mapped_content),
                    'text_areas_count': len(text_areas),
                    'non_text_areas_count': len(non_text_areas),
                    'api_calls_reduction': len(text_areas) - (len(batches) if 'batches' in locals() else 0),
                    'coordinate_precision': 'high',
                    'processing_efficiency': 'batched'
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
    
    def _create_text_batches(self, text_areas: List[Any], max_chars_per_batch: int = 2000) -> List[List[Any]]:
        """Create batches of text areas based on character count"""
        batches = []
        current_batch = []
        current_chars = 0
        
        for area in text_areas:
            area_chars = len(area.combined_text)
            
            # If adding this area would exceed the limit and we already have content, start a new batch
            if current_chars + area_chars > max_chars_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [area]
                current_chars = area_chars
            else:
                # Add to current batch
                current_batch.append(area)
                current_chars += area_chars
        
        # Add the last batch if it has content
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
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

def process_page_worker(page_data_dict: dict, config: Config) -> ProcessResult:
    """
    Robust parallel worker for processing a single page with REAL content extraction.
    Accepts a serializable dict and a config object.
    Always returns a ProcessResult object, capturing either the
    processed PageModel data or a string representation of the error.
    """
    import logging
    import fitz  # PyMuPDF
    import traceback
    logger = logging.getLogger(__name__)
    page_number = page_data_dict.get('page_number', -1)
    
    try:
        # Deserialize and validate input using Pydantic v2 method
        page = PageModel.model_validate(page_data_dict)

        # --- REAL CONTENT EXTRACTION ---
        # Extract actual text from PDF instead of using placeholder content
        pdf_path = page_data_dict.get('pdf_path')
        if not pdf_path:
            raise ValueError("PDF path not provided in page_data_dict")
        
        # Open the PDF document and extract real content
        doc = fitz.open(pdf_path)
        pdf_page = doc.load_page(page_number - 1)  # Convert to 0-based indexing
        
        # Extract text using PyMuPDF's structured text extraction
        page_dict = pdf_page.get_text("dict")
        
        extracted_elements = []
        element_index = 0
        
        # Process text blocks to extract real content
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    # Combine all spans in a line to get the full text
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    
                    if line_text.strip():  # Only add non-empty lines
                        extracted_elements.append(ElementModel(
                            type='text',
                            content=line_text.strip(),  # REAL CONTENT from PDF
                            bbox=line.get("bbox", [0, 0, 0, 0]),
                            confidence=1.0  # High confidence for PyMuPDF extraction
                        ))
                        element_index += 1
        
        doc.close()
        
        # Update the page with extracted real elements
        page.elements = extracted_elements
        
        logger.info(f"Worker extracted {len(extracted_elements)} REAL text elements from page {page_number}")
        if extracted_elements:
            logger.info(f"First element preview: '{extracted_elements[0].content[:100]}...'")
        
        return ProcessResult(page_number=page.page_number, data=page)

    except Exception as e:
        error_str = f"Failed to process page {page_number}: {e}\n{traceback.format_exc()}"
        logger.error(error_str)
        return ProcessResult(page_number=page_number, error=error_str) 