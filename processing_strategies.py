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
    
    async def process_pure_text(self, mapped_content: Dict[str, Any]) -> ProcessingResult:
        """Process text content directly without graph creation"""
        start_time = time.time()
        
        try:
            # Extract all text areas
            text_areas = []
            for area_id, area_data in mapped_content.items():
                if area_data.layout_info.label in ['text', 'paragraph', 'title']:
                    text_areas.append({
                        'content': area_data.combined_text,
                        'bbox': area_data.layout_info.bbox,
                        'label': area_data.layout_info.label,
                        'confidence': area_data.layout_info.confidence
                    })
            
            # Sort by vertical position (reading order)
            text_areas.sort(key=lambda x: x['bbox'][1])
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Pure text processing completed in {processing_time:.3f}s")
            
            return ProcessingResult(
                success=True,
                strategy='direct_text',
                processing_time=processing_time,
                content=text_areas,
                statistics={
                    'text_areas': len(text_areas),
                    'sections': 0,  # No sections in direct text processing
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
    
    async def translate_direct_text(self, document_structure: Dict[str, Any], 
                                  target_language: str = 'en') -> Dict[str, Any]:
        """Translate text directly without graph processing, using intelligent batching"""
        if not self.gemini_service:
            self.logger.warning("⚠️ Gemini service not available for direct text translation.")
            return {
                'error': 'Gemini service not provided',
                'translated_content': document_structure['total_text']
            }
        
        try:
            # Get all text content
            sections = document_structure.get('sections', [])
            if not sections:
                # If no sections, treat the total text as one section
                total_text = document_structure.get('total_text', '')
                if total_text:
                    sections = [{'content': total_text}]
            
            if not sections:
                return {
                    'error': 'No content to translate',
                    'translated_content': ''
                }
            
            # Create batches of sections (max 500 characters per batch for faster processing)
            batches = self._create_section_batches(sections, max_chars_per_batch=500)
            
            self.logger.info(f"📦 Created {len(batches)} batches from {len(sections)} sections")
            
            # Translate batches in parallel instead of sequentially
            async def translate_batch(batch, batch_index):
                batch_text = '\n\n'.join([section['content'] for section in batch])
                self.logger.info(f"   Translating batch {batch_index + 1}/{len(batches)} ({len(batch_text)} chars)")
                
                try:
                    # Add timeout protection for each batch
                    translated_batch = await asyncio.wait_for(
                        self.gemini_service.translate_text(batch_text, target_language, timeout=30.0),
                        timeout=35.0  # 5 seconds extra for processing
                    )
                    return translated_batch
                except asyncio.TimeoutError:
                    self.logger.error(f"❌ Batch {batch_index + 1} translation timed out after 35s, using original")
                    return batch_text
                except Exception as e:
                    self.logger.error(f"❌ Batch {batch_index + 1} translation failed: {e}")
                    # Use original text for failed batch
                    return batch_text
            
            # Execute all batch translations in parallel
            batch_tasks = [translate_batch(batch, i) for i, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine all translated batches
            translated_batches = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Batch {i + 1} failed with exception: {result}")
                    # Use original text for failed batch
                    batch = batches[i]
                    batch_text = '\n\n'.join([section['content'] for section in batch])
                    translated_batches.append(batch_text)
                else:
                    translated_batches.append(result)
            
            # Combine all translated batches
            translated_text = '\n\n'.join(translated_batches)
            
            # Create translated document structure
            translated_structure = {
                'type': 'translated_text_document',
                'original_structure': document_structure,
                'translated_content': translated_text,
                'target_language': target_language,
                'translation_time': 0.0,  # Could add timing if needed
                'total_text': translated_text  # Add this for compatibility
            }
            
            self.logger.info(f"🌐 Direct text translation completed (batched)")
            self.logger.info(f"   Original length: {len(document_structure.get('total_text', ''))}")
            self.logger.info(f"   Translated length: {len(translated_text)}")
            self.logger.info(f"   API calls reduced from {len(sections)} to {len(batches)}")
            
            return translated_structure
            
        except Exception as e:
            self.logger.error(f"❌ Direct text translation failed: {e}")
            return {
                'error': str(e),
                'translated_content': document_structure.get('total_text', '')
            }
    
    def _create_section_batches(self, sections: List[Dict[str, Any]], max_chars_per_batch: int = 2000) -> List[List[Dict[str, Any]]]:
        """Create batches of sections based on character count"""
        batches = []
        current_batch = []
        current_chars = 0
        
        for section in sections:
            section_chars = len(section.get('content', ''))
            
            # If adding this section would exceed the limit and we already have content, start a new batch
            if current_chars + section_chars > max_chars_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [section]
                current_chars = section_chars
            else:
                # Add to current batch
                current_batch.append(section)
                current_chars += section_chars
        
        # Add the last batch if it has content
        if current_batch:
            batches.append(current_batch)
        
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
            layout_info = area_data['layout_info']
            
            # Create node based on area type
            if layout_info['label'] in ['text', 'paragraph', 'title']:
                node_id = graph.add_node(
                    bbox=layout_info['bbox'],
                    class_label=layout_info['label'],
                    confidence=layout_info['confidence'],
                    text=area_data['combined_text'],
                    semantic_role=self._infer_semantic_role(area_data['combined_text']),
                    extra={'area_id': area_id}
                )
            elif layout_info['label'] in ['figure', 'table', 'image']:
                node_id = graph.add_node(
                    bbox=layout_info['bbox'],
                    class_label=layout_info['label'],
                    confidence=layout_info['confidence'],
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
            layout_info = area_data['layout_info']
            yolo_detections.append({
                'label': layout_info['label'],
                'confidence': layout_info['confidence'],
                'bounding_box': list(layout_info['bbox']),
                'class_id': layout_info.get('class_id', 0)
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
        """Execute the appropriate processing strategy - implements user's strategic vision"""
        strategy = processing_result['strategy']
        mapped_content = processing_result['mapped_content']
        
        start_time = time.time()
        
        try:
            if strategy.strategy == 'pure_text_fast':
                # Fast path for pure text (user's strategic vision)
                return await self._process_pure_text_fast(processing_result, target_language)
            elif strategy.strategy == 'coordinate_based_extraction':
                # Coordinate-based extraction for mixed content (user's strategic vision)
                return await self._process_coordinate_based_extraction(processing_result, target_language)
            elif strategy.strategy == 'direct_text':
                # Legacy fallback
                return await self._process_direct_text(mapped_content, target_language)
            elif strategy.strategy == 'minimal_graph':
                return await self._process_minimal_graph(mapped_content, target_language)
            else:  # comprehensive_graph
                return await self._process_comprehensive_graph(processing_result, target_language)
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Strategy execution failed: {e}")
            
            return ProcessingResult(
                success=False,
                strategy=strategy.strategy,
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    async def _process_pure_text_fast(self, processing_result: Dict[str, Any], 
                                    target_language: str) -> ProcessingResult:
        """Process pure text content quickly, then translate."""
        start_time = time.time()
        
        try:
            mapped_content = processing_result.get('mapped_content', {})
            
            # First, process the text to get a structured document
            text_processing_result = await self.direct_text_processor.process_pure_text(mapped_content)

            if not text_processing_result.success:
                return text_processing_result

            # Then, translate the structured document
            document_structure = text_processing_result.content
            translated_content = await self.direct_text_processor.translate_direct_text(
                document_structure, target_language
            )
            
            processing_time = time.time() - start_time
            
            self.performance_stats['pure_text_fast']['total_time'] += processing_time
            self.performance_stats['pure_text_fast']['count'] += 1
            
            self.logger.info(f"⚡ Pure text fast processing completed in {processing_time:.3f}s")
            self.logger.info(f"   Text sections: {len(document_structure.get('sections', []))}")
            self.logger.info(f"   Translation success: {'error' not in translated_content}")
            
            return ProcessingResult(
                success=True,
                strategy='pure_text_fast',
                processing_time=processing_time,
                content=translated_content,
                statistics={
                    **text_processing_result.statistics,
                    'translation_success': 'error' not in translated_content
                }
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Pure text fast processing failed: {e}", exc_info=True)
            
            return ProcessingResult(
                success=False,
                strategy='pure_text_fast',
                processing_time=processing_time,
                content={},
                statistics={},
                error=str(e)
            )
    
    async def _process_coordinate_based_extraction(self, processing_result: Dict[str, Any], 
                                                 target_language: str) -> ProcessingResult:
        """Process mixed content using coordinate-based extraction with intelligent batching"""
        start_time = time.time()
        
        try:
            mapped_content = processing_result.get('mapped_content', {})
            text_areas = []
            non_text_areas = []

            for area_id, area_data in mapped_content.items():
                # Use attribute access for dataclass, not .get()
                if hasattr(area_data, 'combined_text') and area_data.combined_text:
                    text_areas.append(area_data)
                else:
                    non_text_areas.append(area_data)

            # Sort text areas by reading order (top to bottom, left to right)
            text_areas.sort(key=lambda x: (x.layout_info.bbox[1], x.layout_info.bbox[0]))
            
            # Create intelligent batches instead of individual API calls
            translated_texts = []
            if self.gemini_service:
                # Create batches of text areas (max 500 characters per batch for faster processing)
                batches = self._create_text_batches(text_areas, max_chars_per_batch=500)
                
                self.logger.info(f"📦 Created {len(batches)} batches from {len(text_areas)} text areas")
                
                # Translate batches in parallel instead of sequentially
                async def translate_batch(batch, batch_index):
                    batch_text = '\n\n'.join([area.combined_text for area in batch])
                    self.logger.info(f"   Translating batch {batch_index + 1}/{len(batches)} ({len(batch_text)} chars)")
                    
                    try:
                        # Add timeout protection for each batch
                        translated_batch = await asyncio.wait_for(
                            self.gemini_service.translate_text(batch_text, target_language, timeout=30.0),
                            timeout=35.0  # 5 seconds extra for processing
                        )
                        
                        # Split translated batch back into individual texts using a more reliable method
                        # First try the original separator
                        translated_parts = translated_batch.split('\n\n')
                        
                        # If splitting failed, try alternative separators
                        if len(translated_parts) != len(batch):
                            # Try single newline
                            translated_parts = translated_batch.split('\n')
                            if len(translated_parts) != len(batch):
                                # Try period + space as separator
                                translated_parts = translated_batch.split('. ')
                                if len(translated_parts) != len(batch):
                                    # Last resort: split by character count proportionally
                                    self.logger.warning(f"⚠️ Batch {batch_index + 1}: Using proportional splitting")
                                    total_chars = sum(len(area.combined_text) for area in batch)
                                    translated_parts = []
                                    current_pos = 0
                                    for area in batch:
                                        area_ratio = len(area.combined_text) / total_chars
                                        area_chars = int(len(translated_batch) * area_ratio)
                                        translated_parts.append(translated_batch[current_pos:current_pos + area_chars])
                                        current_pos += area_chars
                        
                        # Ensure we have the right number of parts
                        if len(translated_parts) == len(batch):
                            return translated_parts
                        else:
                            # Fallback: use original text if splitting failed
                            self.logger.warning(f"⚠️ Batch {batch_index + 1}: Translation splitting mismatch, using original")
                            return [area.combined_text for area in batch]
                            
                    except asyncio.TimeoutError:
                        self.logger.error(f"❌ Batch {batch_index + 1} translation timed out after 35s, using original")
                        return [area.combined_text for area in batch]
                    except Exception as e:
                        self.logger.error(f"❌ Batch {batch_index + 1} translation failed: {e}")
                        # Use original text for failed batch
                        return [area.combined_text for area in batch]
                
                # Execute all batch translations in parallel
                batch_tasks = [translate_batch(batch, i) for i, batch in enumerate(batches)]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Combine all results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Batch {i + 1} failed with exception: {result}")
                        # Use original text for failed batch
                        batch = batches[i]
                        translated_texts.extend([area.combined_text for area in batch])
                    else:
                        translated_texts.extend(result)
                
                self.logger.info(f"✅ Completed parallel batch translation: {len(translated_texts)} texts translated")
                
            else:
                self.logger.warning("⚠️ Gemini service not available for coordinate-based extraction.")
                translated_texts = [area.combined_text for area in text_areas]

            # Combine translated text with original layout info
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
                    'image_blocks': area.image_blocks
                })
            
            # Sort final content by reading order
            final_content.sort(key=lambda x: (x['layout_info'].bbox[1], x['layout_info'].bbox[0]))

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
                    'text_areas': [{'label': area.layout_info.label, 'text_content': area.combined_text, 'translated_content': translated_texts[i] if i < len(translated_texts) else area.combined_text} for i, area in enumerate(text_areas)],
                    'non_text_areas': [{'label': area.layout_info.label, 'bbox': area.layout_info.bbox} for area in non_text_areas],
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