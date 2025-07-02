"""
Asynchronous Translation Service for Ultimate PDF Translator

Implements concurrent API calls, two-tier caching, and performance optimization
to dramatically reduce translation time while maintaining quality.
"""

import asyncio
import aiohttp
import time
import logging
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config_manager import config_manager
from advanced_caching import advanced_cache_manager

# Optional imports for enhanced error handling
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logging.warning("Tenacity not available - using basic retry logic")

# Import markdown translator (with fallback if not available)
try:
    from markdown_aware_translator import markdown_translator
    MARKDOWN_AWARE_AVAILABLE = True
except ImportError:
    MARKDOWN_AWARE_AVAILABLE = False
    logging.warning("Markdown-aware translator not available")

# Import markdown content processor
try:
    from markdown_content_processor import markdown_processor
    MARKDOWN_PROCESSOR_AVAILABLE = True
except ImportError:
    MARKDOWN_PROCESSOR_AVAILABLE = False
    logging.warning("Markdown content processor not available")

# Import structured document model
from document_model import Document, Page, ContentBlock, Heading, Paragraph, Footnote, Table

logger = logging.getLogger(__name__)

@dataclass
class TranslationTask:
    """Represents a single translation task"""
    text: str
    target_language: str
    context_before: str = ""
    context_after: str = ""
    item_type: str = "text"
    priority: int = 1  # 1=high, 2=medium, 3=low
    task_id: str = ""
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = hashlib.md5(
                (self.text + self.target_language).encode('utf-8')
            ).hexdigest()[:8]

class InMemoryCache:
    """Fast in-memory cache for current translation session"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Remove least recently used item"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
    
    def stats(self) -> Dict:
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
        }

class AsyncTranslationService:
    """
    High-performance asynchronous translation service with two-tier caching
    and concurrent API call optimization.
    """
    
    def __init__(self):
        self.settings = config_manager.gemini_settings
        self.translation_settings = config_manager.translation_enhancement_settings
        
        # Enhanced concurrency settings with adaptive scaling
        try:
            import psutil
            import multiprocessing
            
            # Get system resources
            cpu_count = multiprocessing.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Adaptive concurrency calculation based on system resources
            # Base concurrency on CPU cores and available memory
            base_concurrency = min(cpu_count, int(memory_gb / 2))  # 1 task per 2GB RAM
            
            # Try to get from config, with enhanced defaults
            if hasattr(config_manager, 'translation_enhancement_settings'):
                config_concurrency = config_manager.translation_enhancement_settings.get('max_concurrent_tasks', base_concurrency)
            elif hasattr(config_manager, 'gemini_api_settings'):
                config_concurrency = config_manager.gemini_api_settings.get('max_concurrent_api_calls', base_concurrency)
            elif hasattr(config_manager, 'get_config_value'):
                # Try TranslationEnhancements first
                config_concurrency = config_manager.get_config_value('TranslationEnhancements', 'max_concurrent_tasks', base_concurrency, int)
                if config_concurrency == base_concurrency:  # Default value, try GeminiAPI
                    config_concurrency = config_manager.get_config_value('GeminiAPI', 'max_concurrent_api_calls', base_concurrency, int)
            else:
                config_concurrency = base_concurrency

            # Use the higher of system-based or config-based concurrency
            self.max_concurrent = max(config_concurrency, base_concurrency)
            
            # Cap at reasonable limits to avoid overwhelming the API
            self.max_concurrent = min(self.max_concurrent, 15)  # Increased from 5 to 15
            
            # Adaptive request delay based on concurrency
            if self.max_concurrent <= 5:
                self.request_delay = 0.1  # 100ms for low concurrency
            elif self.max_concurrent <= 10:
                self.request_delay = 0.05  # 50ms for medium concurrency
            else:
                self.request_delay = 0.02  # 20ms for high concurrency

        except Exception as e:
            logger.warning(f"Could not get async config: {e}, using enhanced defaults")
            self.max_concurrent = 10  # Increased default from 5 to 10
            self.request_delay = 0.05
        
        # Two-tier caching with enhanced memory cache
        try:
            if hasattr(config_manager, 'get_value'):
                cache_size = config_manager.get_value('async_optimization', 'memory_cache_size', 2000)  # Increased from 1000
            elif hasattr(config_manager, 'get_config_value'):
                cache_size = config_manager.get_config_value('AsyncOptimization', 'memory_cache_size', 2000, int)
            else:
                cache_size = 2000  # Increased default
        except Exception:
            cache_size = 2000

        self.memory_cache = InMemoryCache(max_size=cache_size)
        self.persistent_cache = advanced_cache_manager
        
        # Enhanced performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits_memory': 0,
            'cache_hits_persistent': 0,
            'api_calls': 0,
            'total_time': 0.0,
            'concurrent_batches': 0,
            'average_response_time': 0.0,
            'peak_concurrency': 0,
            'system_resources': {
                'cpu_count': multiprocessing.cpu_count() if hasattr(multiprocessing, 'cpu_count') else 4,
                'memory_gb': psutil.virtual_memory().total / (1024**3) if hasattr(psutil, 'virtual_memory') else 8.0
            }
        }
        
        # Enhanced semaphore with adaptive control
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Performance monitoring for adaptive scaling
        self.response_times = []
        self.max_response_time_history = 100
        
        logger.info(f"🚀 Enhanced AsyncTranslationService initialized:")
        logger.info(f"   • Max concurrent: {self.max_concurrent} (adaptive)")
        logger.info(f"   • Request delay: {self.request_delay*1000:.0f}ms")
        logger.info(f"   • Memory cache size: {self.memory_cache.max_size}")
        logger.info(f"   • System resources: {self.stats['system_resources']['cpu_count']} CPUs, {self.stats['system_resources']['memory_gb']:.1f}GB RAM")
    
    async def translate_batch_concurrent(self, tasks: List[TranslationTask]) -> List[str]:
        """
        Translate multiple tasks concurrently with intelligent batching
        and two-tier caching.
        """
        start_time = time.time()
        self.stats['total_requests'] += len(tasks)
        self.stats['concurrent_batches'] += 1
        
        logger.info(f"🔄 Starting concurrent translation of {len(tasks)} tasks...")
        
        # Check caches first
        cached_results = {}
        remaining_tasks = []
        
        for task in tasks:
            cache_key = self._generate_cache_key(task)
            
            # Check memory cache first
            result = self.memory_cache.get(cache_key)
            if result:
                cached_results[task.task_id] = result
                self.stats['cache_hits_memory'] += 1
                logger.debug(f"Memory cache hit for task {task.task_id}")
                continue
            
            # Check persistent cache
            result = self.persistent_cache.get_cached_translation(
                task.text, task.target_language, self.settings['model_name'],
                task.context_before, task.context_after
            )
            if result:
                cached_results[task.task_id] = result
                # Also store in memory cache for faster future access
                self.memory_cache.set(cache_key, result)
                self.stats['cache_hits_persistent'] += 1
                logger.debug(f"Persistent cache hit for task {task.task_id}")
                continue
            
            remaining_tasks.append(task)
        
        logger.info(f"📊 Cache performance: {len(cached_results)} hits, {len(remaining_tasks)} API calls needed")
        
        # Translate remaining tasks concurrently
        api_results = {}
        if remaining_tasks:
            api_results = await self._translate_tasks_concurrent(remaining_tasks)
        
        # Combine results in original order
        results = []
        for task in tasks:
            if task.task_id in cached_results:
                results.append(cached_results[task.task_id])
            elif task.task_id in api_results:
                results.append(api_results[task.task_id])
            else:
                logger.warning(f"No result for task {task.task_id}, using original text")
                results.append(task.text)
        
        # Update performance stats
        elapsed = time.time() - start_time
        self.stats['total_time'] += elapsed
        
        logger.info(f"✅ Batch translation completed in {elapsed:.2f}s")
        logger.info(f"   • Cache hits: {len(cached_results)} ({len(cached_results)/len(tasks)*100:.1f}%)")
        logger.info(f"   • API calls: {len(remaining_tasks)}")
        
        return results
    
    async def _translate_tasks_concurrent(self, tasks: List[TranslationTask]) -> Dict[str, str]:
        """Translate tasks concurrently with adaptive scaling and performance monitoring"""
        
        # Track current concurrency and performance
        current_concurrency = len(tasks)
        self.stats['peak_concurrency'] = max(self.stats['peak_concurrency'], current_concurrency)
        
        # Adaptive semaphore based on current performance
        adaptive_semaphore = asyncio.Semaphore(self._get_adaptive_concurrency_limit())
        
        async def translate_with_semaphore(task):
            start_time = time.time()
            async with adaptive_semaphore:
                try:
                    # Add adaptive delay based on current load
                    if len(self.response_times) > 10:
                        avg_response_time = sum(self.response_times[-10:]) / 10
                        if avg_response_time > 5.0:  # If average response > 5s, add delay
                            await asyncio.sleep(self.request_delay * 2)
                    
                    result = await self._translate_single_task(task)
                    
                    # Record response time for adaptive scaling
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    if len(self.response_times) > self.max_response_time_history:
                        self.response_times.pop(0)
                    
                    # Update average response time
                    self.stats['average_response_time'] = sum(self.response_times) / len(self.response_times)
                    
                    return result
                except Exception as e:
                    # Record failed response time for adaptive scaling
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    if len(self.response_times) > self.max_response_time_history:
                        self.response_times.pop(0)
                    raise e

        # Execute with enhanced error handling and progress tracking
        try:
            from tqdm.asyncio import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False

        logger.info(f"⚡ Starting adaptive concurrent translation of {len(tasks)} tasks...")
        logger.info(f"   • Adaptive concurrency limit: {self._get_adaptive_concurrency_limit()}")
        logger.info(f"   • Current average response time: {self.stats['average_response_time']:.2f}s")

        # Execute tasks with progress tracking
        if TQDM_AVAILABLE and len(tasks) > 5:
            results = await tqdm.gather(
                *[translate_with_semaphore(task) for task in tasks],
                desc="🌐 Translating (adaptive)",
                unit="task",
                colour="green"
            )
        else:
            results = await asyncio.gather(
                *[translate_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )

        # Process results and maintain order
        translated_results = {}
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(results):
            task = tasks[i]
            if isinstance(result, Exception):
                logger.warning(f"Translation failed for task {task.task_id}: {str(result)}")
                # Keep original text on failure
                translated_results[task.task_id] = task.text
                failed_count += 1
            else:
                translated_results[task.task_id] = result
                successful_count += 1

        # Log performance metrics
        logger.info(f"✅ Adaptive concurrent translation completed:")
        logger.info(f"   • Successful: {successful_count}")
        logger.info(f"   • Failed: {failed_count}")
        logger.info(f"   • Average response time: {self.stats['average_response_time']:.2f}s")
        logger.info(f"   • Peak concurrency: {self.stats['peak_concurrency']}")

        return translated_results

    def _get_adaptive_concurrency_limit(self) -> int:
        """Calculate adaptive concurrency limit based on performance metrics"""
        base_limit = self.max_concurrent
        
        # Adjust based on recent response times
        if len(self.response_times) >= 5:
            recent_avg = sum(self.response_times[-5:]) / 5
            
            if recent_avg > 10.0:  # Very slow responses
                return max(1, base_limit // 3)  # Reduce to 1/3
            elif recent_avg > 5.0:  # Slow responses
                return max(2, base_limit // 2)  # Reduce to 1/2
            elif recent_avg < 1.0:  # Fast responses
                return min(base_limit + 2, 20)  # Increase by 2, cap at 20
            elif recent_avg < 2.0:  # Good responses
                return min(base_limit + 1, 18)  # Increase by 1, cap at 18
        
        return base_limit
    
    async def _translate_single_task(self, task: TranslationTask) -> str:
        """Translate a single task using the Gemini API with robust error handling"""
        if TENACITY_AVAILABLE:
            return await self._translate_with_tenacity(task)
        else:
            return await self._translate_with_basic_retry(task)

    async def _translate_with_tenacity(self, task: TranslationTask) -> str:
        """Translate with tenacity retry logic"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception))
        )
        async def _do_translation():
            try:
                # Import here to avoid circular imports
                from translation_service import translation_service

                # Use the async version directly instead of the sync wrapper
                result = await translation_service.translate_text(
                    task.text,
                    task.target_language,
                    "",  # style_guide
                    task.context_before,
                    task.context_after,
                    task.item_type
                )

                self.stats['api_calls'] += 1
                logger.debug(f"API translation completed for task {task.task_id}")
                return result

            except Exception as e:
                logger.warning(f"Translation attempt failed for task {task.task_id}: {e}")
                raise

        try:
            return await _do_translation()
        except Exception as e:
            logger.error(f"All retry attempts failed for task {task.task_id}: {e}")
            raise

    async def _translate_with_basic_retry(self, task: TranslationTask) -> str:
        """Translate with basic retry logic when tenacity is not available"""
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                # Import here to avoid circular imports
                from translation_service import translation_service

                # Use the async version directly instead of the sync wrapper
                result = await translation_service.translate_text(
                    task.text,
                    task.target_language,
                    "",  # style_guide
                    task.context_before,
                    task.context_after,
                    task.item_type
                )

                self.stats['api_calls'] += 1
                logger.debug(f"API translation completed for task {task.task_id}")
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Translation attempt {attempt + 1} failed for task {task.task_id}: {e}")
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for task {task.task_id}: {e}")
                    raise
    
    def _generate_cache_key(self, task: TranslationTask) -> str:
        """Generate cache key for memory cache"""
        key_data = f"{task.text}|{task.target_language}|{task.context_before}|{task.context_after}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _cache_result(self, task: TranslationTask, result: str):
        """Cache successful translation result in both tiers"""
        # Memory cache
        cache_key = self._generate_cache_key(task)
        self.memory_cache.set(cache_key, result)
        
        # Persistent cache
        self.persistent_cache.cache_translation(
            task.text, task.target_language, self.settings['model_name'],
            result, task.context_before, task.context_after
        )
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        total_requests = max(self.stats['total_requests'], 1)
        cache_hit_rate = (self.stats['cache_hits_memory'] + self.stats['cache_hits_persistent']) / total_requests
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'avg_time_per_batch': self.stats['total_time'] / max(self.stats['concurrent_batches'], 1),
            'memory_cache_stats': self.memory_cache.stats(),
            'persistent_cache_stats': self.persistent_cache.get_cache_statistics()
        }
    
    def clear_session_cache(self):
        """Clear the in-memory cache (useful between documents)"""
        self.memory_cache.clear()
        logger.info("🧹 Session cache cleared")

    def create_tasks_from_content(self, content_items: List[Dict], target_language: str) -> List[TranslationTask]:
        """Create translation tasks from content items with context awareness"""
        tasks = []

        for i, item in enumerate(content_items):
            if item.get('type') == 'image':
                continue  # Skip image items

            text = item.get('text', '').strip()
            if not text:
                continue

            # Get context from surrounding items
            context_before = ""
            context_after = ""

            # Look for context in previous items
            for j in range(max(0, i-2), i):
                prev_item = content_items[j]
                if prev_item.get('type') != 'image':
                    prev_text = prev_item.get('text', '').strip()
                    if prev_text:
                        context_before = prev_text[-200:] + " " + context_before

            # Look for context in next items
            for j in range(i+1, min(len(content_items), i+3)):
                next_item = content_items[j]
                if next_item.get('type') != 'image':
                    next_text = next_item.get('text', '').strip()
                    if next_text:
                        context_after = context_after + " " + next_text[:200]

            # Determine priority based on content type
            priority = 1  # High priority by default
            if item.get('type') in ['h1', 'h2', 'h3']:
                priority = 1  # Headers are high priority
            elif item.get('type') == 'paragraph':
                priority = 2  # Paragraphs are medium priority
            else:
                priority = 3  # Other content is low priority

            task = TranslationTask(
                text=text,
                target_language=target_language,
                context_before=context_before.strip(),
                context_after=context_after.strip(),
                item_type=item.get('type', 'text'),
                priority=priority
            )

            tasks.append(task)

        logger.info(f"📝 Created {len(tasks)} translation tasks from {len(content_items)} content items")
        return tasks

    async def translate_content_items_async(self, content_items: List[Dict], target_language: str) -> List[Dict]:
        """
        High-level method to translate content items asynchronously
        with automatic task creation and result integration.
        """
        logger.info(f"🚀 Starting async translation of {len(content_items)} content items...")

        # Create translation tasks
        tasks = self.create_tasks_from_content(content_items, target_language)

        if not tasks:
            logger.warning("No translation tasks created")
            return content_items

        # Execute translations concurrently
        translated_texts = await self.translate_batch_concurrent(tasks)

        # Integrate results back into content items
        result_items = []
        task_index = 0

        for item in content_items:
            if item.get('type') == 'image':
                # Keep image items unchanged
                result_items.append(item.copy())
            else:
                text = item.get('text', '').strip()
                if text and task_index < len(translated_texts):
                    # Update with translated text
                    updated_item = item.copy()
                    updated_item['text'] = translated_texts[task_index]
                    result_items.append(updated_item)
                    task_index += 1
                else:
                    # Keep original if no translation available
                    result_items.append(item.copy())

        logger.info(f"✅ Async translation completed: {task_index} items translated")

        # Post-process translated content to convert Markdown back to structured format
        if MARKDOWN_PROCESSOR_AVAILABLE:
            logger.info("🔄 Post-processing translated Markdown content...")
            result_items = markdown_processor.process_translated_content_items(result_items)

            # Validate the structured content
            validation_stats = markdown_processor.validate_structured_content(result_items)

            if validation_stats['total_headers'] > 0:
                logger.info(f"✅ Found {validation_stats['total_headers']} headers for TOC generation")
            else:
                logger.warning("⚠️ No headers found after processing - TOC may be empty")

        return result_items

    async def translate_document_structured(self, document: Document, target_language: str) -> Document:
        """
        NEW STRUCTURED APPROACH: Translate a structured Document object.

        This method operates on ContentBlock objects, preserving their type
        and metadata while translating only the content text.
        """
        logger.info(f"🏗️ Starting structured document translation: {document.title}")
        logger.info(f"   • Target language: {target_language}")
        logger.info(f"   • Total blocks: {len(document.get_all_content_blocks())}")

        # Create translation tasks from content blocks
        tasks = self._create_tasks_from_content_blocks(document.get_all_content_blocks(), target_language)

        if not tasks:
            logger.warning("No translation tasks created from document")
            return document

        # Execute translations concurrently
        translated_texts = await self.translate_batch_concurrent(tasks)

        # Create new document with translated content
        translated_document = self._apply_translations_to_document(document, tasks, translated_texts)

        logger.info(f"✅ Structured document translation completed")
        return translated_document

    def _create_tasks_from_content_blocks(self, content_blocks: List[ContentBlock],
                                        target_language: str) -> List[TranslationTask]:
        """Create translation tasks from structured content blocks"""
        tasks = []

        for i, block in enumerate(content_blocks):
            # Skip footnotes entirely - they will be kept in original language
            if isinstance(block, Footnote):
                continue

            text = block.content.strip()
            if not text:
                continue

            # Get context from surrounding blocks
            context_before = ""
            context_after = ""

            # Look for context in previous blocks
            for j in range(max(0, i-2), i):
                prev_block = content_blocks[j]
                if not isinstance(prev_block, Footnote):
                    prev_text = prev_block.content.strip()
                    if prev_text:
                        context_before = prev_text[-200:] + " " + context_before

            # Look for context in next blocks
            for j in range(i+1, min(len(content_blocks), i+3)):
                next_block = content_blocks[j]
                if not isinstance(next_block, Footnote):
                    next_text = next_block.content.strip()
                    if next_text:
                        context_after = context_after + " " + next_text[:200]

            # Determine priority based on block type
            priority = self._get_block_translation_priority(block)

            # Create task with block ID for tracking
            task = TranslationTask(
                text=text,
                target_language=target_language,
                context_before=context_before.strip(),
                context_after=context_after.strip(),
                item_type=block.get_content_type().value,
                priority=priority,
                task_id=block.block_id  # Use block ID for tracking
            )

            tasks.append(task)

        logger.info(f"📝 Created {len(tasks)} translation tasks from {len(content_blocks)} content blocks")
        return tasks

    def _get_block_translation_priority(self, block: ContentBlock) -> int:
        """Determine translation priority based on content block type"""
        if isinstance(block, Heading):
            return 1  # Headings are highest priority for TOC
        elif isinstance(block, Table):
            return 1  # Tables need careful translation
        elif isinstance(block, Paragraph):
            return 2  # Paragraphs are medium priority
        else:
            return 3  # Other content is lower priority

    def _apply_translations_to_document(self, original_document: Document,
                                      tasks: List[TranslationTask],
                                      translated_texts: List[str]) -> Document:
        """Apply translated texts back to document structure"""
        # Create mapping of block IDs to translated texts
        translation_map = {}
        for task, translated_text in zip(tasks, translated_texts):
            translation_map[task.task_id] = translated_text

        # Create new document with same structure
        translated_document = Document(
            title=original_document.title,
            document_metadata=original_document.document_metadata.copy()
        )

        # Process each page
        for original_page in original_document.pages:
            translated_page = Page(
                page_number=original_page.page_number,
                page_metadata=original_page.page_metadata.copy()
            )

            # Process each block on the page
            for block in original_page.content_blocks:
                translated_block = self._translate_content_block(block, translation_map)
                translated_page.add_block(translated_block)

            translated_document.add_page(translated_page)

        return translated_document

    def _translate_content_block(self, block: ContentBlock,
                               translation_map: Dict[str, str]) -> ContentBlock:
        """Create a translated copy of a content block"""
        # Get translated text if available
        translated_text = translation_map.get(block.block_id, block.content)

        # Create new block of the same type with translated content
        block_dict = block.to_dict()
        block_dict['content'] = translated_text

        # Import the creation function
        from document_model import create_content_block_from_dict
        translated_block = create_content_block_from_dict(block_dict)

        return translated_block if translated_block else block

    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate a single text string to the target language"""
        if not text.strip():
            return text
        
        try:
            # Create a simple translation task
            task = TranslationTask(
                text=text,
                target_language=target_language,
                item_type="text",
                priority=2
            )
            
            # Use the existing translation infrastructure
            result = await self._translate_single_task(task)
            return result
            
        except Exception as e:
            logger.error(f"❌ Translation failed for text: {e}")
            return text  # Return original text on failure

class PreemptiveImageFilter:
    """
    Fast local image filtering to avoid sending decorative/simple images
    to expensive AI analysis APIs.
    """

    def __init__(self):
        try:
            if hasattr(config_manager, 'get_value'):
                self.min_complexity_threshold = config_manager.get_value('image_filtering', 'min_complexity_threshold', 0.3)
                self.min_size_bytes = config_manager.get_value('image_filtering', 'min_size_bytes', 10000)
                self.max_simple_colors = config_manager.get_value('image_filtering', 'max_simple_colors', 5)
            elif hasattr(config_manager, 'get_config_value'):
                self.min_complexity_threshold = config_manager.get_config_value('ImageFiltering', 'min_complexity_threshold', 0.3, float)
                self.min_size_bytes = config_manager.get_config_value('ImageFiltering', 'min_size_bytes', 10000, int)
                self.max_simple_colors = config_manager.get_config_value('ImageFiltering', 'max_simple_colors', 5, int)
            else:
                self.min_complexity_threshold = 0.3
                self.min_size_bytes = 10000
                self.max_simple_colors = 5
        except Exception as e:
            logger.warning(f"Could not get image filtering config: {e}, using defaults")
            self.min_complexity_threshold = 0.3
            self.min_size_bytes = 10000
            self.max_simple_colors = 5

        logger.info(f"🔍 PreemptiveImageFilter initialized:")
        logger.info(f"   • Min complexity: {self.min_complexity_threshold}")
        logger.info(f"   • Min size: {self.min_size_bytes} bytes")
        logger.info(f"   • Max simple colors: {self.max_simple_colors}")

    def should_analyze_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Quickly determine if an image is worth sending to AI analysis.
        Returns (should_analyze, reason)
        """
        try:
            import os
            from PIL import Image

            # Check file size first (fastest check)
            file_size = os.path.getsize(image_path)
            if file_size < self.min_size_bytes:
                return False, f"Too small ({file_size} bytes)"

            # Load image for analysis
            with Image.open(image_path) as img:
                # Check dimensions
                width, height = img.size
                if width < 50 or height < 50:
                    return False, f"Dimensions too small ({width}x{height})"

                # Check color complexity
                if img.mode in ['RGB', 'RGBA']:
                    # Sample the image to check color diversity
                    sampled = img.resize((50, 50))  # Small sample for speed
                    colors = sampled.getcolors(maxcolors=256)

                    if colors and len(colors) <= self.max_simple_colors:
                        return False, f"Too few colors ({len(colors)})"

                    # Check for single-color or very simple images
                    if colors:
                        dominant_color_ratio = max(count for count, _ in colors) / (50 * 50)
                        if dominant_color_ratio > 0.9:
                            return False, f"Single color dominates ({dominant_color_ratio:.1%})"

                # Check aspect ratio for likely decorative elements
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 10:  # Very thin/wide images are often decorative
                    return False, f"Extreme aspect ratio ({aspect_ratio:.1f})"

                # If we get here, the image passed all quick filters
                return True, "Passed complexity checks"

        except Exception as e:
            logger.warning(f"Error analyzing image {image_path}: {e}")
            # When in doubt, analyze it
            return True, "Error in filtering - defaulting to analyze"

    def filter_image_list(self, image_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter a list of images into those worth analyzing and those to skip.
        Returns (analyze_list, skip_list)
        """
        analyze_list = []
        skip_list = []

        logger.info(f"🔍 Filtering {len(image_paths)} images...")

        for image_path in image_paths:
            should_analyze, reason = self.should_analyze_image(image_path)

            if should_analyze:
                analyze_list.append(image_path)
                logger.debug(f"✅ Will analyze: {image_path} - {reason}")
            else:
                skip_list.append(image_path)
                logger.debug(f"⏭️ Skipping: {image_path} - {reason}")

        logger.info(f"📊 Filtering results:")
        logger.info(f"   • Will analyze: {len(analyze_list)} images")
        logger.info(f"   • Skipping: {len(skip_list)} images")
        logger.info(f"   • Reduction: {len(skip_list)/len(image_paths)*100:.1f}%")

        return analyze_list, skip_list

# Global instances
async_translation_service = AsyncTranslationService()
preemptive_image_filter = PreemptiveImageFilter()
