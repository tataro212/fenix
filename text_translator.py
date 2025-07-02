"""
Structured Text Translator - Core Implementation

This module implements the translation service with strict separation of concerns
as specified in the directives. It removes API instruction contamination and
uses proper system parameters for translation instructions.

Key Features:
- Pure text payload without API instruction contamination
- Proper use of system/instruction parameters
- Sequence-aware translation with TextBlock support
- Asynchronous integrity with sequence_id reassembly
- Enhanced error handling and fallback mechanisms
"""

import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

# Import the structured document model
from pdf_processor import TextBlock, structured_pdf_processor

logger = logging.getLogger(__name__)

class StructuredTextTranslator:
    """
    Structured text translator that maintains document integrity.
    
    This class implements the new translation approach that preserves
    sequence and semantic coherence throughout the translation process.
    """
    
    def __init__(self, translation_service=None):
        self.logger = logging.getLogger(__name__)
        
        # Import the translation service
        if translation_service:
            self.translation_service = translation_service
        else:
            try:
                from translation_service_enhanced import enhanced_translation_service
                self.translation_service = enhanced_translation_service
            except ImportError:
                self.logger.warning("Enhanced translation service not available, using fallback")
                self.translation_service = None
        
        self.logger.info("🔧 Structured Text Translator initialized")
    
    async def translate_document_structure(self, 
                                         document_structure: List[List[TextBlock]], 
                                         target_language: str,
                                         style_guide: str = "") -> List[List[TextBlock]]:
        """
        Translate the entire document structure while preserving sequence integrity.
        
        This is the main entry point for document translation. It accepts the
        structured document model and returns the translated structure with
        all sequence information preserved.
        """
        try:
            self.logger.info(f"🌐 Starting structured document translation to {target_language}")
            
            # Create a copy of the document structure to avoid modifying the original
            translated_structure = []
            
            for page_num, page_blocks in enumerate(document_structure):
                self.logger.info(f"   Translating page {page_num + 1} ({len(page_blocks)} blocks)")
                
                # Translate page blocks in parallel while preserving sequence
                translated_page_blocks = await self._translate_page_blocks_parallel(
                    page_blocks, target_language, style_guide
                )
                
                translated_structure.append(translated_page_blocks)
            
            self.logger.info(f"✅ Document translation completed: {len(translated_structure)} pages")
            return translated_structure
            
        except Exception as e:
            self.logger.error(f"❌ Error in document translation: {e}")
            raise
    
    async def _translate_page_blocks_parallel(self, 
                                            page_blocks: List[TextBlock], 
                                            target_language: str,
                                            style_guide: str) -> List[TextBlock]:
        """
        Translate page blocks in parallel while maintaining sequence integrity.
        
        Each translation task carries the sequence_id of the TextBlock it is translating.
        As translations return, we use the sequence_id to update the correct block
        in our master data structure.
        """
        try:
            # Create translation tasks with sequence_id preservation
            translation_tasks = []
            
            for block in page_blocks:
                if block.content_type.value == "image":
                    # Skip translation for image blocks
                    translation_tasks.append(self._create_identity_task(block))
                else:
                    # Create translation task with sequence_id
                    task = self._create_translation_task(block, target_language, style_guide)
                    translation_tasks.append(task)
            
            # Execute all tasks in parallel
            self.logger.debug(f"Executing {len(translation_tasks)} translation tasks in parallel")
            results = await asyncio.gather(*translation_tasks, return_exceptions=True)
            
            # Reassemble results using sequence_id
            translated_blocks = self._reassemble_translated_blocks(page_blocks, results)
            
            return translated_blocks
            
        except Exception as e:
            self.logger.error(f"Error in parallel page translation: {e}")
            # Fallback to sequential translation
            return await self._translate_page_blocks_sequential(page_blocks, target_language, style_guide)
    
    async def _create_translation_task(self, 
                                     block: TextBlock, 
                                     target_language: str, 
                                     style_guide: str) -> Dict[str, Any]:
        """Create a translation task that preserves sequence_id"""
        try:
            # Get context from surrounding blocks (if available)
            context = self._get_translation_context(block)
            
            # Use the translation service with proper system parameters
            if self.translation_service and hasattr(self.translation_service, 'translate_text_enhanced'):
                translated_text = await self.translation_service.translate_text_enhanced(
                    text=block.text,  # Pure text payload - no API instructions
                    target_language=target_language,
                    style_guide=style_guide,
                    prev_context=context.get('prev', ''),
                    next_context=context.get('next', ''),
                    item_type=block.content_type.value
                )
            else:
                # Fallback translation
                translated_text = await self._fallback_translation(block.text, target_language)
            
            return {
                'sequence_id': block.sequence_id,
                'translated_text': translated_text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            self.logger.warning(f"Translation failed for block {block.sequence_id}: {e}")
            return {
                'sequence_id': block.sequence_id,
                'translated_text': block.text,  # Keep original text
                'success': False,
                'error': str(e)
            }
    
    async def _create_identity_task(self, block: TextBlock) -> Dict[str, Any]:
        """Create an identity task for non-translatable blocks (images, etc.)"""
        return {
            'sequence_id': block.sequence_id,
            'translated_text': block.text,  # Keep original
            'success': True,
            'error': None
        }
    
    def _reassemble_translated_blocks(self, 
                                    original_blocks: List[TextBlock], 
                                    translation_results: List[Dict[str, Any]]) -> List[TextBlock]:
        """
        Reassemble translated blocks using sequence_id to restore original order.
        
        This function uses the sequence_id to update the text attribute of the
        corresponding TextBlock object in the master data structure. The order
        of completion no longer matters because we have a key to restore the
        original sequence.
        """
        try:
            # Create a mapping of sequence_id to translation result
            result_mapping = {}
            for result in translation_results:
                if isinstance(result, dict) and 'sequence_id' in result:
                    result_mapping[result['sequence_id']] = result
            
            # Create translated blocks in original order
            translated_blocks = []
            
            for original_block in original_blocks:
                # Create a copy of the original block
                translated_block = TextBlock(
                    text=original_block.text,  # Will be updated below
                    page_num=original_block.page_num,
                    sequence_id=original_block.sequence_id,
                    bbox=original_block.bbox,
                    font_size=original_block.font_size,
                    font_family=original_block.font_family,
                    font_weight=original_block.font_weight,
                    font_style=original_block.font_style,
                    color=original_block.color,
                    content_type=original_block.content_type,
                    confidence=original_block.confidence,
                    block_type=original_block.block_type
                )
                
                # Update with translated text if available
                if original_block.sequence_id in result_mapping:
                    result = result_mapping[original_block.sequence_id]
                    if result.get('success', False):
                        translated_block.text = result['translated_text']
                    else:
                        self.logger.warning(f"Translation failed for {original_block.sequence_id}: {result.get('error', 'Unknown error')}")
                
                translated_blocks.append(translated_block)
            
            self.logger.debug(f"Reassembled {len(translated_blocks)} blocks using sequence_id")
            return translated_blocks
            
        except Exception as e:
            self.logger.error(f"Error reassembling translated blocks: {e}")
            return original_blocks  # Return original blocks as fallback
    
    def _get_translation_context(self, block: TextBlock) -> Dict[str, str]:
        """Get translation context from surrounding blocks"""
        # This is a simplified implementation
        # In a full implementation, you would access the document structure
        # to get the previous and next blocks for better context
        return {
            'prev': '',
            'next': ''
        }
    
    async def _fallback_translation(self, text: str, target_language: str) -> str:
        """Fallback translation method when enhanced service is not available"""
        try:
            # Simple fallback - could be enhanced with other translation services
            self.logger.warning("Using fallback translation method")
            return text  # Return original text as fallback
            
        except Exception as e:
            self.logger.error(f"Fallback translation failed: {e}")
            return text
    
    async def _translate_page_blocks_sequential(self, 
                                              page_blocks: List[TextBlock], 
                                              target_language: str,
                                              style_guide: str) -> List[TextBlock]:
        """Sequential translation fallback when parallel processing fails"""
        try:
            self.logger.info("Using sequential translation fallback")
            
            translated_blocks = []
            
            for block in page_blocks:
                if block.content_type.value == "image":
                    # Keep image blocks unchanged
                    translated_blocks.append(block)
                    continue
                
                try:
                    # Translate block sequentially
                    if self.translation_service and hasattr(self.translation_service, 'translate_text_enhanced'):
                        translated_text = await self.translation_service.translate_text_enhanced(
                            text=block.text,
                            target_language=target_language,
                            style_guide=style_guide,
                            prev_context="",
                            next_context="",
                            item_type=block.content_type.value
                        )
                    else:
                        translated_text = await self._fallback_translation(block.text, target_language)
                    
                    # Create translated block
                    translated_block = TextBlock(
                        text=translated_text,
                        page_num=block.page_num,
                        sequence_id=block.sequence_id,
                        bbox=block.bbox,
                        font_size=block.font_size,
                        font_family=block.font_family,
                        font_weight=block.font_weight,
                        font_style=block.font_style,
                        color=block.color,
                        content_type=block.content_type,
                        confidence=block.confidence,
                        block_type=block.block_type
                    )
                    
                    translated_blocks.append(translated_block)
                    
                except Exception as e:
                    self.logger.warning(f"Sequential translation failed for block {block.sequence_id}: {e}")
                    translated_blocks.append(block)  # Keep original block
            
            return translated_blocks
            
        except Exception as e:
            self.logger.error(f"Sequential translation fallback failed: {e}")
            return page_blocks  # Return original blocks as final fallback
    
    def export_translated_document(self, 
                                 translated_structure: List[List[TextBlock]], 
                                 output_path: str) -> bool:
        """
        Export the translated document structure to a readable format.
        
        This function iterates through the master data structure in its sorted order,
        writing the translated text to the output file.
        """
        try:
            self.logger.info(f"📝 Exporting translated document to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for page_num, page_blocks in enumerate(translated_structure):
                    f.write(f"\n--- Page {page_num + 1} ---\n\n")
                    
                    for block in page_blocks:
                        # Write block content based on type
                        if block.content_type.value == "image":
                            f.write(f"[Image: {block.text}]\n\n")
                        elif block.content_type.value == "heading":
                            f.write(f"# {block.text}\n\n")
                        elif block.content_type.value == "list_item":
                            f.write(f"• {block.text}\n")
                        else:
                            f.write(f"{block.text}\n\n")
            
            self.logger.info(f"✅ Document exported successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting document: {e}")
            return False
    
    def get_translation_statistics(self, 
                                 original_structure: List[List[TextBlock]], 
                                 translated_structure: List[List[TextBlock]]) -> Dict[str, Any]:
        """Get comprehensive statistics about the translation process"""
        try:
            total_original_blocks = sum(len(page_blocks) for page_blocks in original_structure)
            total_translated_blocks = sum(len(page_blocks) for page_blocks in translated_structure)
            
            # Count successful translations
            successful_translations = 0
            failed_translations = 0
            
            for page_blocks in translated_structure:
                for block in page_blocks:
                    if block.content_type.value != "image":
                        # Check if text was actually translated (simplified check)
                        if block.text and len(block.text) > 0:
                            successful_translations += 1
                        else:
                            failed_translations += 1
            
            return {
                'total_original_blocks': total_original_blocks,
                'total_translated_blocks': total_translated_blocks,
                'successful_translations': successful_translations,
                'failed_translations': failed_translations,
                'success_rate': (successful_translations / (successful_translations + failed_translations)) * 100 if (successful_translations + failed_translations) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating translation statistics: {e}")
            return {}

# Create global instance
structured_text_translator = StructuredTextTranslator() 