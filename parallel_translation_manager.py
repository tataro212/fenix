#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Translation Manager for Intelligent Content Batching

This module integrates the existing AsyncTranslationService with the new
IntelligentContentBatcher to provide true parallel processing of translation batches.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import our new intelligent content batcher
from intelligent_content_batcher import IntelligentContentBatcher, ContentBatch, ContentItem

# Import existing async translation service
try:
    from async_translation_service import AsyncTranslationService
    ASYNC_TRANSLATION_AVAILABLE = True
except ImportError:
    ASYNC_TRANSLATION_AVAILABLE = False
    logger.warning("AsyncTranslationService not available")

logger = logging.getLogger(__name__)


@dataclass
class TranslationBatchResult:
    """Result from translating a content batch"""
    batch_id: str
    original_text: str
    translated_text: str
    success: bool
    processing_time: float
    error: Optional[str] = None
    content_types: List[str] = None
    item_count: int = 0


@dataclass
class ParallelTranslationResult:
    """Result from parallel translation of multiple batches"""
    total_batches: int
    successful_batches: int
    failed_batches: int
    total_processing_time: float
    average_batch_time: float
    batch_results: List[TranslationBatchResult]
    api_calls_reduction: float
    performance_stats: Dict[str, Any]


class ParallelTranslationManager:
    """
    Manages parallel translation of content batches using the existing
    AsyncTranslationService and new IntelligentContentBatcher.
    """
    
    def __init__(self, max_concurrent_batches: int = 5):
        self.max_concurrent_batches = max_concurrent_batches
        self.content_batcher = IntelligentContentBatcher(max_batch_chars=12000)
        
        # Initialize async translation service if available
        if ASYNC_TRANSLATION_AVAILABLE:
            self.async_translator = AsyncTranslationService()
            logger.info(f"🚀 ParallelTranslationManager initialized with {max_concurrent_batches} concurrent batches")
        else:
            self.async_translator = None
            logger.warning("⚠️ AsyncTranslationService not available, falling back to sequential processing")
        
        # Performance tracking
        self.stats = {
            'total_batches_processed': 0,
            'total_api_calls': 0,
            'total_processing_time': 0.0,
            'average_batch_time': 0.0,
            'success_rate': 0.0,
            'api_calls_reduction': 0.0
        }
    
    async def translate_content_parallel(self, mapped_content: Dict[str, Any], 
                                       target_language: str = 'en') -> ParallelTranslationResult:
        """
        Translate content using intelligent batching and parallel processing.
        
        Args:
            mapped_content: Content from PyMuPDF-YOLO mapping
            target_language: Target language for translation
            
        Returns:
            ParallelTranslationResult with comprehensive results
        """
        start_time = time.time()
        
        try:
            # Step 1: Create content items from mapped content
            logger.info("📝 Step 1: Creating content items...")
            content_items = self.content_batcher.create_content_items(mapped_content)
            
            if not content_items:
                logger.warning("⚠️ No content items created")
                return self._create_empty_result()
            
            # Step 2: Create intelligent batches
            logger.info("📦 Step 2: Creating intelligent batches...")
            batches = self.content_batcher.create_intelligent_batches(content_items)
            
            if not batches:
                logger.warning("⚠️ No batches created")
                return self._create_empty_result()
            
            # Step 3: Translate batches in parallel
            logger.info(f"🌐 Step 3: Translating {len(batches)} batches in parallel...")
            batch_results = await self._translate_batches_parallel(batches, target_language)
            
            # Step 4: Calculate results
            total_time = time.time() - start_time
            result = self._calculate_translation_result(batch_results, total_time, len(content_items))
            
            # Update global stats
            self._update_global_stats(result)
            
            logger.info(f"✅ Parallel translation completed:")
            logger.info(f"   Total batches: {result.total_batches}")
            logger.info(f"   Successful: {result.successful_batches}")
            logger.info(f"   Failed: {result.failed_batches}")
            logger.info(f"   Total time: {result.total_processing_time:.3f}s")
            logger.info(f"   Average batch time: {result.average_batch_time:.3f}s")
            logger.info(f"   API calls reduction: {result.api_calls_reduction:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Parallel translation failed: {e}")
            return self._create_error_result(str(e))
    
    async def _translate_batches_parallel(self, batches: List[ContentBatch], 
                                        target_language: str) -> List[TranslationBatchResult]:
        """Translate batches in parallel with concurrency control"""
        if not self.async_translator:
            # Fallback to sequential processing
            return await self._translate_batches_sequential(batches, target_language)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        # Create translation tasks
        async def translate_batch_with_semaphore(batch: ContentBatch) -> TranslationBatchResult:
            async with semaphore:
                return await self._translate_single_batch(batch, target_language)
        
        # Execute all tasks in parallel
        tasks = [translate_batch_with_semaphore(batch) for batch in batches]
        
        logger.info(f"🔄 Executing {len(tasks)} translation tasks with concurrency limit of {self.max_concurrent_batches}")
        
        # Use asyncio.gather for parallel execution
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch {i+1} failed with exception: {result}")
                # Create error result for failed batch
                error_result = TranslationBatchResult(
                    batch_id=f"batch_{i}",
                    original_text=batches[i].get_combined_text(),
                    translated_text=batches[i].get_combined_text(),  # Use original as fallback
                    success=False,
                    processing_time=0.0,
                    error=str(result),
                    content_types=[t.value for t in batches[i].content_types],
                    item_count=len(batches[i].items)
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _translate_batches_sequential(self, batches: List[ContentBatch], 
                                          target_language: str) -> List[TranslationBatchResult]:
        """Fallback sequential translation when async service is not available"""
        logger.warning("⚠️ Using sequential translation (async service not available)")
        
        batch_results = []
        for i, batch in enumerate(batches):
            logger.info(f"   Translating batch {i+1}/{len(batches)}")
            result = await self._translate_single_batch(batch, target_language)
            batch_results.append(result)
        
        return batch_results
    
    async def _translate_single_batch(self, batch: ContentBatch, 
                                    target_language: str) -> TranslationBatchResult:
        """Translate a single content batch"""
        start_time = time.time()
        
        try:
            # Get combined text for translation
            original_text = batch.get_combined_text()
            
            if not original_text.strip():
                return TranslationBatchResult(
                    batch_id=batch.batch_id,
                    original_text="",
                    translated_text="",
                    success=True,
                    processing_time=time.time() - start_time,
                    content_types=[t.value for t in batch.content_types],
                    item_count=len(batch.items)
                )
            
            # Use async translation service if available
            if self.async_translator:
                translated_text = await self.async_translator.translate_text(
                    original_text, target_language
                )
            else:
                # Mock translation for testing
                translated_text = f"[TRANSLATED TO {target_language.upper()}] {original_text}"
            
            processing_time = time.time() - start_time
            
            return TranslationBatchResult(
                batch_id=batch.batch_id,
                original_text=original_text,
                translated_text=translated_text,
                success=True,
                processing_time=processing_time,
                content_types=[t.value for t in batch.content_types],
                item_count=len(batch.items)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Translation failed for batch {batch.batch_id}: {e}")
            
            return TranslationBatchResult(
                batch_id=batch.batch_id,
                original_text=batch.get_combined_text(),
                translated_text=batch.get_combined_text(),  # Use original as fallback
                success=False,
                processing_time=processing_time,
                error=str(e),
                content_types=[t.value for t in batch.content_types],
                item_count=len(batch.items)
            )
    
    def _calculate_translation_result(self, batch_results: List[TranslationBatchResult], 
                                    total_time: float, total_items: int) -> ParallelTranslationResult:
        """Calculate comprehensive translation result"""
        successful_batches = [r for r in batch_results if r.success]
        failed_batches = [r for r in batch_results if not r.success]
        
        # Calculate API calls reduction
        old_api_calls = total_items  # Individual item processing
        new_api_calls = len(batch_results)  # Batch processing
        api_calls_reduction = ((old_api_calls - new_api_calls) / old_api_calls) * 100 if old_api_calls > 0 else 0
        
        # Calculate average batch time
        total_batch_time = sum(r.processing_time for r in batch_results)
        average_batch_time = total_batch_time / len(batch_results) if batch_results else 0
        
        # Performance stats
        performance_stats = {
            'total_items_processed': total_items,
            'items_per_batch': total_items / len(batch_results) if batch_results else 0,
            'character_utilization': sum(len(r.original_text) for r in batch_results) / (len(batch_results) * 12000) * 100,
            'success_rate': len(successful_batches) / len(batch_results) * 100 if batch_results else 0
        }
        
        return ParallelTranslationResult(
            total_batches=len(batch_results),
            successful_batches=len(successful_batches),
            failed_batches=len(failed_batches),
            total_processing_time=total_time,
            average_batch_time=average_batch_time,
            batch_results=batch_results,
            api_calls_reduction=api_calls_reduction,
            performance_stats=performance_stats
        )
    
    def _create_empty_result(self) -> ParallelTranslationResult:
        """Create empty result when no content to process"""
        return ParallelTranslationResult(
            total_batches=0,
            successful_batches=0,
            failed_batches=0,
            total_processing_time=0.0,
            average_batch_time=0.0,
            batch_results=[],
            api_calls_reduction=0.0,
            performance_stats={}
        )
    
    def _create_error_result(self, error: str) -> ParallelTranslationResult:
        """Create error result when processing fails"""
        return ParallelTranslationResult(
            total_batches=0,
            successful_batches=0,
            failed_batches=1,
            total_processing_time=0.0,
            average_batch_time=0.0,
            batch_results=[TranslationBatchResult(
                batch_id="error",
                original_text="",
                translated_text="",
                success=False,
                processing_time=0.0,
                error=error
            )],
            api_calls_reduction=0.0,
            performance_stats={'error': error}
        )
    
    def _update_global_stats(self, result: ParallelTranslationResult):
        """Update global performance statistics"""
        self.stats['total_batches_processed'] += result.total_batches
        self.stats['total_api_calls'] += result.total_batches
        self.stats['total_processing_time'] += result.total_processing_time
        
        # Update averages
        if self.stats['total_batches_processed'] > 0:
            self.stats['average_batch_time'] = (
                self.stats['total_processing_time'] / self.stats['total_batches_processed']
            )
            self.stats['success_rate'] = (
                (self.stats['total_batches_processed'] - result.failed_batches) 
                / self.stats['total_batches_processed'] * 100
            )
        
        self.stats['api_calls_reduction'] = result.api_calls_reduction
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'global_stats': self.stats.copy(),
            'batcher_stats': self.content_batcher.get_batching_report(),
            'efficiency_metrics': {
                'api_calls_reduction_avg': self.stats['api_calls_reduction'],
                'success_rate_avg': self.stats['success_rate'],
                'average_batch_time': self.stats['average_batch_time']
            }
        }


def create_parallel_translation_manager(max_concurrent_batches: int = 5) -> ParallelTranslationManager:
    """Factory function to create a parallel translation manager"""
    return ParallelTranslationManager(max_concurrent_batches=max_concurrent_batches) 