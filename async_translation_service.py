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
import re
from typing import List, Dict, Optional, Tuple, Any
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
# try:
#     from markdown_content_processor import markdown_processor
#     MARKDOWN_PROCESSOR_AVAILABLE = True
# except ImportError:
#     MARKDOWN_PROCESSOR_AVAILABLE = False
#     logging.warning("Markdown content processor not available")

# Since this is optional and doesn't exist, disable it for now
MARKDOWN_PROCESSOR_AVAILABLE = False

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

@dataclass
class TranslationBatch:
    """Represents a batch of text blocks for translation with contextual continuity"""
    batch_id: str
    text_blocks: List[TranslationTask]
    combined_text: str
    total_chars: int
    context_from_previous: str = ""
    target_language: str = ""
    batch_type: str = "content"
    page_number: int = 1
    processing_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = hashlib.md5(
                (self.combined_text + self.target_language).encode('utf-8')
            ).hexdigest()[:8]
        if self.processing_metadata is None:
            self.processing_metadata = {}

class IntelligentBatcher:
    """
    Enhanced intelligent batching system that preserves paragraph continuity
    and prevents content fragmentation across batch boundaries.
    """
    
    def __init__(self, max_batch_size: int = 12000, min_batch_size: int = 1000):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.logger = logging.getLogger(__name__)
        
        # Enhanced paragraph detection patterns
        self.paragraph_indicators = [
            r'\n\s*\n',  # Double line breaks
            r'\.\s*\n\s*[A-Z]',  # Sentence end followed by new sentence
            r'\.\s*\n\s*\d+\.',  # Numbered paragraphs
            r'\n\s*•\s*',  # Bullet points
            r'\n\s*-\s*',  # Dash points
            r'\n\s*\d+\.\s*',  # Numbered lists
        ]
        
        # ENHANCED: Paragraph boundary markers for translation preservation
        self.paragraph_start_marker = "[PARAGRAPH_START]"
        self.paragraph_end_marker = "[PARAGRAPH_END]"
        self.paragraph_break_marker = "[PARAGRAPH_BREAK]"
        
        # Content type priorities for batching
        self.content_type_priorities = {
            'heading': 1,
            'title': 1,
            'paragraph': 2,
            'list_item': 2,
            'caption': 3,
            'footnote': 4,  # Process footnotes separately
            'table': 5
        }

    def create_content_aware_batches(self, translation_tasks: List[TranslationTask]) -> List[TranslationBatch]:
        """
        Create batches with enhanced paragraph and content awareness.
        MODIFIED: Now strictly preserves the original order of translation_tasks and sends special roles as single-item batches.
        """
        if not translation_tasks:
            return []
        # Step 1: Add paragraph boundary markers to tasks
        self._add_paragraph_markers_to_tasks(translation_tasks)
        # Define special roles
        special_roles = {'footnote', 'title', 'header', 'heading', 'caption', 'list_item', 'bullet'}
        batches = []
        current_batch = []
        current_size = 0
        for i, task in enumerate(translation_tasks):
            role = task.item_type.lower()
            # If this is a special role, send as a single-item batch
            if any(special in role for special in special_roles):
                # Finalize current batch if any
                if current_batch:
                    batch = self._create_batch_from_tasks(current_batch, len(batches))
                    batches.append(batch)
                    current_batch = []
                    current_size = 0
                # Single-item batch for special role
                batch = self._create_batch_from_tasks([task], len(batches))
                batches.append(batch)
            else:
                task_size = len(task.text)
                if current_size + task_size > self.max_batch_size and current_batch:
                    batch = self._create_batch_from_tasks(current_batch, len(batches))
                    batches.append(batch)
                    current_batch = []
                    current_size = 0
                current_batch.append(task)
                current_size += task_size
        # Add final batch if any
        if current_batch:
            batch = self._create_batch_from_tasks(current_batch, len(batches))
            batches.append(batch)
        self.logger.info(f"📦 Created {len(batches)} order-preserving batches (special roles as single-item batches)")
        return batches

    def _should_start_new_batch(self, current_batch: List[TranslationTask], current_size: int, 
                               new_task: TranslationTask, new_task_size: int, 
                               current_page: Optional[int], new_page: int) -> bool:
        """
        Determine if we should start a new batch based on content awareness.
        """
        # Don't start new batch if current batch is empty
        if not current_batch:
            return False
        
        # Always start new batch if it would exceed max size
        if current_size + new_task_size > self.max_batch_size:
            return True
        
        # Start new batch for page breaks (but allow some flexibility)
        if current_page is not None and new_page != current_page:
            # Only break if current batch is reasonably sized
            if current_size > self.min_batch_size:
                return True
        
        # Check for paragraph breaks within the same page
        if current_page == new_page:
            last_task = current_batch[-1]
            if self._is_paragraph_break(last_task, new_task):
                # Only break if it creates reasonably sized batches
                if current_size > self.min_batch_size and new_task_size > 200:
                    return True
        
        # Check for content type changes that should trigger new batch
        if self._should_break_on_content_type_change(current_batch, new_task):
            return True
        
        return False

    def _is_paragraph_break(self, task1: TranslationTask, task2: TranslationTask) -> bool:
        """
        Detect if there's a paragraph break between two tasks.
        """
        # Check for explicit paragraph indicators
        text1_end = task1.text.strip()
        text2_start = task2.text.strip()
        
        # Strong paragraph indicators
        if text1_end.endswith('.') and text2_start and text2_start[0].isupper():
            return True
        
        # Check for numbered paragraphs
        if re.match(r'^\d+\.', text2_start):
            return True
        
        # Check for bullet points
        if text2_start.startswith(('•', '-', '*')):
            return True
        
        # Check for heading-like content
        if 'heading' in task2.item_type.lower() or 'title' in task2.item_type.lower():
            return True
        
        return False

    def _should_break_on_content_type_change(self, current_batch: List[TranslationTask], 
                                           new_task: TranslationTask) -> bool:
        """
        Determine if content type change should trigger a new batch.
        """
        if not current_batch:
            return False
        
        last_task = current_batch[-1]
        last_type = last_task.item_type.split('_')[0]
        new_type = new_task.item_type.split('_')[0]
        
        # Always break for headings and titles
        if new_type in ['heading', 'title']:
            return True
        
        # Break when transitioning from headings to content
        if last_type in ['heading', 'title'] and new_type in ['paragraph', 'text']:
            return True
        
        # Break for tables (keep tables separate)
        if new_type == 'table' or last_type == 'table':
            return True
        
        return False

    def _create_footnote_batches(self, footnote_tasks: List[TranslationTask], 
                                start_batch_id: int) -> List[TranslationBatch]:
        """
        Create separate batches for footnotes, grouped by page.
        """
        if not footnote_tasks:
            return []
        
        # Group footnotes by page
        footnotes_by_page = {}
        for task in footnote_tasks:
            page = self._extract_page_number(task)
            if page not in footnotes_by_page:
                footnotes_by_page[page] = []
            footnotes_by_page[page].append(task)
        
        # Create batches for each page's footnotes
        batches = []
        batch_id = start_batch_id
        
        for page, page_footnotes in sorted(footnotes_by_page.items()):
            # Sort footnotes by position on page
            page_footnotes.sort(key=lambda t: self._extract_position(t))
            
            # Create batch for this page's footnotes
            combined_text = self._create_combined_text_from_tasks(page_footnotes)
            batch = TranslationBatch(
                batch_id=f"footnotes_page_{page}_{batch_id}",
                text_blocks=page_footnotes,
                combined_text=combined_text,
                total_chars=sum(len(task.text) for task in page_footnotes),
                batch_type="footnotes",
                page_number=page,
                processing_metadata={
                    'content_type': 'footnotes',
                    'page_specific': True,
                    'requires_special_formatting': True
                }
            )
            batches.append(batch)
            batch_id += 1
        
        return batches

    def _extract_page_number(self, task: TranslationTask) -> int:
        """Extract page number from task metadata."""
        # Try to extract from task_id or context
        if hasattr(task, 'page_number'):
            return task.page_number
        
        # Try to extract from task_id pattern
        if '_page_' in task.task_id:
            try:
                page_part = task.task_id.split('_page_')[1].split('_')[0]
                return int(page_part)
            except (IndexError, ValueError):
                pass
        
        # Try to extract from context
        if task.context_before and 'page' in task.context_before.lower():
            match = re.search(r'page\s*(\d+)', task.context_before.lower())
            if match:
                return int(match.group(1))
        
        return 1  # Default to page 1

    def _extract_position(self, task: TranslationTask) -> float:
        """Extract position information from task for sorting."""
        # Try to extract from task_id
        if '_pos_' in task.task_id:
            try:
                pos_part = task.task_id.split('_pos_')[1].split('_')[0]
                return float(pos_part)
            except (IndexError, ValueError):
                pass
        
        # Try to extract from context or metadata
        if hasattr(task, 'bbox') and task.bbox:
            # Use y-coordinate for vertical position
            return task.bbox[1] if isinstance(task.bbox, (list, tuple)) else 0
        
        return 0.0

    def _create_batch_from_tasks(self, tasks: List[TranslationTask], batch_id: int) -> TranslationBatch:
        """Create a translation batch from a list of tasks."""
        total_chars = sum(len(task.text) for task in tasks)
        
        # Determine batch type from content
        content_types = [task.item_type.split('_')[0] for task in tasks]
        most_common_type = max(set(content_types), key=content_types.count)
        
        # Determine page number
        page_numbers = [self._extract_page_number(task) for task in tasks]
        primary_page = max(set(page_numbers), key=page_numbers.count)
        
        # Create combined text for the batch
        combined_text = self._create_combined_text_from_tasks(tasks)
        
        return TranslationBatch(
            batch_id=f"content_batch_{batch_id}",
            text_blocks=tasks,
            combined_text=combined_text,
            total_chars=total_chars,
            batch_type=most_common_type,
            page_number=primary_page,
            processing_metadata={
                'content_types': content_types,
                'paragraph_aware': True,
                'preserves_flow': True
            }
        )

    def _create_combined_text_from_tasks(self, tasks: List[TranslationTask]) -> str:
        """Create combined text from tasks with proper XML formatting."""
        xml_segments = []
        
        for i, task in enumerate(tasks):
            # Escape XML special characters
            escaped_text = task.text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            xml_segments.append(f'<seg id="{i}">{escaped_text}</seg>')
        
        return '\n'.join(xml_segments)

    def parse_batch_translation(self, batch: TranslationBatch, translated_text) -> List[str]:
        """
        Parse translated batch text back into individual translations.
        Enhanced to preserve paragraph structure and handle various response formats.
        """
        translations = []
        # Handle case where translated_text might be a list (fallback from failed translation)
        if isinstance(translated_text, list):
            self.logger.warning(f"⚠️ Received list instead of string for batch translation, using fallback")
            return translated_text[:len(batch.text_blocks)] if len(translated_text) >= len(batch.text_blocks) else [task.text for task in batch.text_blocks]
        # Ensure we have a string
        if not isinstance(translated_text, str):
            self.logger.error(f"❌ Invalid translated_text type: {type(translated_text)}")
            return [task.text for task in batch.text_blocks]
        # Log the raw response for debugging
        self.logger.info(f"🔍 Parsing batch response for {len(batch.text_blocks)} segments")
        self.logger.info(f"Raw response length: {len(translated_text)} chars")
        self.logger.info(f"🔍 RAW GEMINI RESPONSE (first 500 chars):\n{translated_text[:500]}")
        # Try multiple parsing strategies for robustness
        translation_map = {}
        
        # Strategy 1: Standard XML seg tags (most common)
        pattern1 = r'<seg id=["\']?(\d+)["\']?[^>]*>(.*?)</seg>'
        matches1 = re.findall(pattern1, translated_text, re.DOTALL | re.IGNORECASE)
        if matches1:
            self.logger.info(f"✅ Strategy 1: Found {len(matches1)} segments with standard XML")
            for match in matches1:
                seg_id = int(match[0])
                content = match[1].strip()
                translation_map[seg_id] = self._clean_translated_content(content)
                self.logger.debug(f"   Segment {seg_id}: {content[:100]}...")
        
        # Strategy 2: Try without quotes around ID
        if not translation_map:
            pattern2 = r'<seg id=(\d+)[^>]*>(.*?)</seg>'
            matches2 = re.findall(pattern2, translated_text, re.DOTALL | re.IGNORECASE)
            if matches2:
                self.logger.info(f"✅ Strategy 2: Found {len(matches2)} segments without quotes")
                for match in matches2:
                    seg_id = int(match[0])
                    content = match[1].strip()
                    translation_map[seg_id] = self._clean_translated_content(content)
                    self.logger.debug(f"   Segment {seg_id}: {content[:100]}...")
        
        # Strategy 3: Try single-line segments (no DOTALL)
        if not translation_map:
            pattern3 = r'<seg[^>]*id=["\']?(\d+)["\']?[^>]*>(.*?)</seg>'
            matches3 = re.findall(pattern3, translated_text, re.IGNORECASE)
            if matches3:
                self.logger.info(f"✅ Strategy 3: Found {len(matches3)} single-line segments")
                for match in matches3:
                    seg_id = int(match[0])
                    content = match[1].strip()
                    translation_map[seg_id] = self._clean_translated_content(content)
                    self.logger.debug(f"   Segment {seg_id}: {content[:100]}...")
        
        # Strategy 4: Look for numbered segments without XML tags (fallback)
        if not translation_map:
            self.logger.warning(f"🚨 No XML segments found, trying numbered lines...")
            # Split by lines and look for patterns like "1. content" or "Segment 0: content"
            lines = translated_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Pattern: "0: content" or "0. content" or "Segment 0: content"
                number_match = re.match(r'(?:Segment\s+)?(\d+)[:.\-]\s*(.*)', line, re.IGNORECASE)
                if number_match:
                    seg_id = int(number_match.group(1))
                    content = number_match.group(2).strip()
                    if content:
                        translation_map[seg_id] = self._clean_translated_content(content)
                        self.logger.debug(f"   Numbered segment {seg_id}: {content[:100]}...")
            
            if translation_map:
                self.logger.info(f"✅ Strategy 4: Found {len(translation_map)} numbered segments")
        
        # Strategy 5: Split by lines and match sequentially (last resort)
        if not translation_map:
            self.logger.warning(f"🚨 No structured segments found, trying sequential line matching...")
            lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
            # Filter out lines that look like XML tags or metadata
            content_lines = []
            for line in lines:
                if not line.startswith('<') and not line.startswith('[') and len(line) > 5:
                    content_lines.append(line)
                    self.logger.debug(f"   Content line: {line[:100]}...")
            
            if content_lines:
                self.logger.info(f"✅ Strategy 5: Using {len(content_lines)} sequential lines")
                for i, line in enumerate(content_lines):
                    if i < len(batch.text_blocks):
                        translation_map[i] = self._clean_translated_content(line)
        
        # If still no translations found, log the full response for debugging
        if not translation_map:
            self.logger.error(f"🚨 CRITICAL: No translations found in Gemini response!")
            self.logger.error(f"🔍 FULL GEMINI RESPONSE:\n{translated_text}")
            self.logger.error(f"🔍 ORIGINAL BATCH CONTENT:\n{batch.combined_text[:1000]}")
        
        # Reconstruct translations in original order
        for i, task in enumerate(batch.text_blocks):
            if i in translation_map:
                translations.append(translation_map[i])
                self.logger.debug(f"✅ Segment {i}: Translated successfully")
            else:
                self.logger.warning(f"❌ Missing translation for segment {i}, using original text")
                # Clean the original text of any paragraph markers we added
                clean_original = task.text.replace('[PARAGRAPH_START]', '').replace('[PARAGRAPH_END]', '').replace('[PARAGRAPH_BREAK]', '').strip()
                translations.append(clean_original)
        self.logger.info(f"📊 Batch parsing results: {len(translation_map)}/{len(batch.text_blocks)} segments successfully translated")
        return translations

    def _clean_translated_content(self, content: str) -> str:
        """Clean and normalize translated content"""
        # Unescape XML entities
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        content = content.replace('&apos;', "'")
        
        # Remove any XML tags that might have leaked through
        content = re.sub(r'<[^>]+>', '', content)
        
        # Preserve paragraph structure
        content = self._preserve_paragraph_structure(content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content

    def _preserve_paragraph_structure(self, text: str) -> str:
        """
        Preserve paragraph structure in translated text.
        
        ENHANCED: Now processes paragraph boundary markers to reconstruct proper paragraphs.
        """
        # Step 1: Process paragraph boundary markers
        text = self._process_paragraph_markers(text)
        
        # Step 2: Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Step 3: Ensure proper spacing after sentences
        text = re.sub(r'\.(\S)', r'. \1', text)
        
        # Step 4: Clean up excessive whitespace
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _add_paragraph_markers_to_tasks(self, translation_tasks: List[TranslationTask]) -> None:
        """
        Add paragraph boundary markers to translation tasks based on content analysis.
        
        This method analyzes the text content and adds markers that will survive translation
        to help reconstruct proper paragraph structure.
        """
        try:
            for i, task in enumerate(translation_tasks):
                original_text = task.text
                
                # Detect if this task starts a new paragraph
                if self._is_paragraph_start(task, translation_tasks, i):
                    task.text = f"{self.paragraph_start_marker} {original_text}"
                
                # Detect if this task ends a paragraph
                if self._is_paragraph_end(task, translation_tasks, i):
                    task.text = f"{task.text} {self.paragraph_end_marker}"
                
                # Detect paragraph breaks within the task
                if self._contains_paragraph_break(original_text):
                    task.text = self._insert_paragraph_break_markers(task.text)
                
                self.logger.debug(f"Task {i}: Added paragraph markers to text: {task.text[:100]}...")
                
        except Exception as e:
            self.logger.error(f"Failed to add paragraph markers: {e}")
    
    def _is_paragraph_start(self, task: TranslationTask, all_tasks: List[TranslationTask], index: int) -> bool:
        """
        Determine if a task represents the start of a new paragraph.
        """
        # First task is always a paragraph start
        if index == 0:
            return True
        
        # Check if previous task ended a paragraph
        if index > 0:
            prev_task = all_tasks[index - 1]
            
            # Different pages likely mean new paragraph
            if self._extract_page_number(task) != self._extract_page_number(prev_task):
                return True
            
            # Check text content for paragraph indicators
            prev_text = prev_task.text.strip()
            current_text = task.text.strip()
            
            # Previous text ends with sentence terminator and current starts with capital
            if (prev_text.endswith(('.', '!', '?')) and 
                current_text and current_text[0].isupper()):
                return True
            
            # Check for explicit paragraph patterns
            if any(re.search(pattern, prev_text + '\n' + current_text) 
                   for pattern in self.paragraph_indicators):
                return True
        
        return False
    
    def _is_paragraph_end(self, task: TranslationTask, all_tasks: List[TranslationTask], index: int) -> bool:
        """
        Determine if a task represents the end of a paragraph.
        """
        # Last task is always a paragraph end
        if index == len(all_tasks) - 1:
            return True
        
        # Check if next task starts a new paragraph
        if index < len(all_tasks) - 1:
            next_task = all_tasks[index + 1]
            return self._is_paragraph_start(next_task, all_tasks, index + 1)
        
        return False
    
    def _contains_paragraph_break(self, text: str) -> bool:
        """
        Check if text contains internal paragraph breaks.
        """
        # Look for double line breaks or other paragraph indicators
        return any(re.search(pattern, text) for pattern in self.paragraph_indicators)
    
    def _insert_paragraph_break_markers(self, text: str) -> str:
        """
        Insert paragraph break markers within text that contains paragraph breaks.
        """
        # Replace double line breaks with paragraph break markers
        text = re.sub(r'\n\s*\n', f' {self.paragraph_break_marker} ', text)
        
        # Replace other paragraph indicators
        for pattern in self.paragraph_indicators[1:]:  # Skip double line breaks (already handled)
            text = re.sub(pattern, f' {self.paragraph_break_marker} ', text)
        
        return text
    
    def _process_paragraph_markers(self, text: str) -> str:
        """
        Process paragraph markers in translated text to reconstruct proper paragraph structure.
        """
        try:
            # Replace paragraph start markers
            text = text.replace(self.paragraph_start_marker, '')
            
            # Replace paragraph end markers
            text = text.replace(self.paragraph_end_marker, '')
            
            # Replace paragraph break markers with double line breaks
            text = text.replace(self.paragraph_break_marker, '\n\n')
            
            # Clean up any double markers or excessive whitespace
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'^\s+|\s+$', '', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to process paragraph markers: {e}")
            return text

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
    High-performance asynchronous translation service with intelligent batching,
    contextual continuity, and two-tier caching.
    """
    
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.settings = config_manager.gemini_settings
        self.translation_settings = config_manager.translation_enhancement_settings
        
        # Initialize intelligent batcher
        self.batcher = IntelligentBatcher(max_batch_size=12000, min_batch_size=1000)
        
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
            'batches_created': 0,
            'tasks_per_batch': 0,
            'character_utilization': 0.0,
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
        
        logger.info(f"🚀 Enhanced AsyncTranslationService initialized with intelligent batching:")
        logger.info(f"   • Max batch size: {self.batcher.max_batch_size} characters")
        logger.info(f"   • Min batch size: {self.batcher.min_batch_size} characters")
        logger.info(f"   • Max concurrent: {self.max_concurrent} (adaptive)")
        logger.info(f"   • Request delay: {self.request_delay*1000:.0f}ms")
        logger.info(f"   • Memory cache size: {self.memory_cache.max_size}")
        logger.info(f"   • System resources: {self.stats['system_resources']['cpu_count']} CPUs, {self.stats['system_resources']['memory_gb']:.1f}GB RAM")

    async def translate_batch_concurrent(self, tasks: List[TranslationTask]) -> List[str]:
        """
        Translate multiple tasks using intelligent batching with contextual continuity.
        This dramatically reduces API calls while maintaining translation quality.
        """
        start_time = time.time()
        self.stats['total_requests'] += len(tasks)
        
        logger.info(f"🔄 Starting intelligent batch translation of {len(tasks)} tasks...")
        
        # Create intelligent batches with contextual continuity
        batches = self.batcher.create_content_aware_batches(tasks)
        
        if not batches:
            logger.warning("No batches created from tasks")
            return [task.text for task in tasks]
        
        # Update statistics
        self.stats['batches_created'] += len(batches)
        self.stats['tasks_per_batch'] = len(tasks) / len(batches)
        self.stats['character_utilization'] = sum(b.total_chars for b in batches) / (len(batches) * 14000) * 100
        
        logger.info(f"📊 Batch optimization results:")
        logger.info(f"   • {len(batches)} batches created from {len(tasks)} tasks")
        logger.info(f"   • Average tasks per batch: {self.stats['tasks_per_batch']:.1f}")
        logger.info(f"   • Character utilization: {self.stats['character_utilization']:.1f}%")
        logger.info(f"   • API calls reduced by: {(1 - len(batches) / len(tasks)) * 100:.1f}%")
        
        # Check cache for batches
        cached_results = {}
        remaining_batches = []
        
        for batch in batches:
            cache_key = self._generate_batch_cache_key(batch)
            
            # Check memory cache first
            result = self.memory_cache.get(cache_key)
            if result:
                cached_results[batch.batch_id] = result
                self.stats['cache_hits_memory'] += 1
                continue
            
            # Check persistent cache (simplified for batches)
            result = self.persistent_cache.get_cached_translation(
                batch.combined_text[:200], batch.target_language, self.settings['model_name']
            )
            if result:
                cached_results[batch.batch_id] = result
                self.memory_cache.set(cache_key, result)
                self.stats['cache_hits_persistent'] += 1
                continue
            
            remaining_batches.append(batch)
        
        logger.info(f"📊 Batch cache performance: {len(cached_results)} hits, {len(remaining_batches)} API calls needed")
        
        # Translate remaining batches concurrently
        batch_results = {}
        if remaining_batches:
            batch_results = await self._translate_batches_concurrent(remaining_batches)
        
        # Parse batch results back to individual translations
        all_translations = []
        for batch in batches:
            if batch.batch_id in cached_results:
                translated_text = cached_results[batch.batch_id]
            elif batch.batch_id in batch_results:
                translated_text = batch_results[batch.batch_id]
            else:
                logger.warning(f"No result for batch {batch.batch_id}, using original text")
                translated_text = batch.combined_text
            
            # Parse batch translation back to individual translations
            batch_translations = self.batcher.parse_batch_translation(batch, translated_text)
            all_translations.extend(batch_translations)
        
        # Update performance stats
        elapsed = time.time() - start_time
        self.stats['total_time'] += elapsed
        
        logger.info(f"✅ Intelligent batch translation completed in {elapsed:.2f}s")
        logger.info(f"   • Total API calls: {len(remaining_batches)} (reduced from {len(tasks)})")
        logger.info(f"   • Performance improvement: {len(tasks) / max(len(remaining_batches), 1):.1f}x faster")
        
        return all_translations
    
    def _generate_batch_cache_key(self, batch: TranslationBatch) -> str:
        """Generate cache key for a translation batch"""
        key_data = f"{batch.combined_text[:500]}_{batch.target_language}_{batch.context_from_previous[:100]}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    async def _translate_batches_concurrent(self, batches: List[TranslationBatch]) -> Dict[str, str]:
        """Translate batches concurrently with adaptive scaling"""
        
        # Adaptive semaphore based on current performance
        adaptive_semaphore = asyncio.Semaphore(self._get_adaptive_concurrency_limit())
        
        async def translate_batch_with_semaphore(batch):
            async with adaptive_semaphore:
                try:
                    # Add adaptive delay
                    await asyncio.sleep(self.request_delay)
                    
                    result = await self._translate_single_batch(batch)
                    
                    # Cache the result
                    cache_key = self._generate_batch_cache_key(batch)
                    self.memory_cache.set(cache_key, result)
                    
                    return result
                except Exception as e:
                    logger.error(f"Batch translation failed for {batch.batch_id}: {e}")
                    return batch.combined_text  # Return original on failure
        
        # Execute with progress tracking
        try:
            from tqdm.asyncio import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
        
        logger.info(f"⚡ Starting concurrent batch translation of {len(batches)} batches...")
        
        if TQDM_AVAILABLE and len(batches) > 2:
            results = await tqdm.gather(
                *[translate_batch_with_semaphore(batch) for batch in batches],
                desc="🌐 Translating batches",
                unit="batch",
                colour="green"
            )
        else:
            results = await asyncio.gather(
                *[translate_batch_with_semaphore(batch) for batch in batches],
                return_exceptions=True
            )
        
        # Process results
        batch_results = {}
        for i, result in enumerate(results):
            batch = batches[i]
            if isinstance(result, Exception):
                logger.warning(f"Batch translation failed for {batch.batch_id}: {str(result)}")
                batch_results[batch.batch_id] = batch.combined_text
            else:
                batch_results[batch.batch_id] = result
        
        return batch_results
    
    async def _translate_single_batch(self, batch: TranslationBatch) -> str:
        """Translate a single batch using the Gemini API with enhanced Greek support"""
        try:
            # Import the Gemini service directly for enhanced Greek translation
            from gemini_service import GeminiService
            
            # Create Gemini service instance
            gemini_service = GeminiService()
            
            # Create enhanced translation prompt for the batch
            prompt = f"""You are a professional academic translator. You MUST translate ALL text from English to GREEK (ελληνικά).

CRITICAL REQUIREMENTS:
1. TARGET LANGUAGE: GREEK (ελληνικά) - You MUST use Greek language ONLY
2. Do NOT translate to any other language (not French, German, Russian, Chinese, Spanish, etc.)
3. Every translation MUST be in GREEK characters (α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω)

XML FORMAT REQUIREMENTS:
1. You MUST return translations in this EXACT XML format
2. Each translation MUST be wrapped in <seg id="X">GREEK_TRANSLATION</seg> tags
3. Use EXACTLY this format: <seg id="NUMBER">GREEK_TRANSLATION</seg>
4. Include ALL segments (0 through {len(batch.text_blocks)-1})
5. Do NOT add any other text outside the <seg> tags

EXAMPLE - Translate to GREEK:
Input: "Hello world"
CORRECT Output: <seg id="0">Γεια σας κόσμε</seg>
WRONG Output: <seg id="0">Bonjour le monde</seg> (This is French - FORBIDDEN!)

Now translate these {len(batch.text_blocks)} segments to GREEK (ελληνικά):

{batch.combined_text}

Remember: 
- Use ONLY GREEK language (ελληνικά)
- Return ONLY the <seg id="X">GREEK_TRANSLATION</seg> format
- NO French, German, Russian, Chinese, Spanish or any other language"""

            # Make the translation request
            translated_text = await gemini_service.translate_text(
                prompt, 
                batch.target_language
            )
            
            if not translated_text or not translated_text.strip():
                self.logger.error(f"❌ Empty response from Gemini for batch of {len(batch.text_blocks)} segments")
                return batch.combined_text  # Return original combined text as string
            
            # Return the raw translated text string (not parsed)
            return translated_text
            
        except Exception as e:
            self.logger.error(f"❌ Batch translation failed: {e}")
            return batch.combined_text  # Return original combined text as string

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
    
    def _generate_cache_key(self, task: TranslationTask) -> str:
        """Generate cache key for memory cache (backward compatibility)"""
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
            'avg_time_per_batch': self.stats['total_time'] / max(self.stats['batches_created'], 1),
            'memory_cache_stats': self.memory_cache.stats(),
            'persistent_cache_stats': self.persistent_cache.get_cache_statistics()
        }
    
    def clear_session_cache(self):
        """Clear the in-memory cache (useful between documents)"""
        self.memory_cache.clear()
        logger.info("🧹 Session cache cleared")

    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate a single text string to the target language with enhanced Greek support"""
        if not text.strip():
            return text
        
        try:
            # For single text translation, use enhanced Gemini service directly
            from gemini_service import GeminiService
            
            gemini_service = GeminiService()
            
            # Use enhanced translation method for better Greek output
            if hasattr(gemini_service, 'translate_text_with_context'):
                result = await gemini_service.translate_text_with_context(
                    text, 
                    target_language,
                    context="Single text translation",
                    translation_style="academic"
                )
                return result
            else:
                # Fallback to basic method
                return await gemini_service.translate_text(text, target_language)
            
        except Exception as e:
            logger.error(f"❌ Translation failed for text: {e}")
            return text  # Return original text on failure

# Global instances
async_translation_service = AsyncTranslationService()
