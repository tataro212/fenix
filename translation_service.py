"""
Translation Service Module for Ultimate PDF Translator

Handles translation API calls, batching, caching, and quality assessment
"""

import asyncio
import json
import os
import time
import logging
import hashlib
from collections import defaultdict
import google.generativeai as genai
import psutil
import multiprocessing

from config_manager import config_manager
from utils import get_cache_key

logger = logging.getLogger(__name__)

# Import structured document model for Document translation
try:
    from structured_document_model import Document, ContentType, Heading, Paragraph, ListItem, Footnote, Caption
    STRUCTURED_MODEL_AVAILABLE = True
except ImportError:
    STRUCTURED_MODEL_AVAILABLE = False
    # Create dummy classes for type checking
    class Document: pass
    class Heading: pass
    class Paragraph: pass
    class ListItem: pass
    class Footnote: pass
    class Caption: pass
    logger.warning("Structured document model not available")

# Import markdown translator (with fallback if not available)
try:
    from markdown_aware_translator import markdown_translator
    MARKDOWN_AWARE_AVAILABLE = True
except ImportError:
    MARKDOWN_AWARE_AVAILABLE = False
    logger.warning("Markdown-aware translator not available")

class TranslationCache:
    """Manages translation caching functionality"""
    
    def __init__(self):
        self.settings = config_manager.translation_enhancement_settings
        self.cache = {}
        self.cache_file = self.settings['translation_cache_file_path']
        self.enabled = self.settings['use_translation_cache']
        
        if self.enabled:
            self.load_cache()
    
    def load_cache(self):
        """Load translation cache from file"""
        if not self.enabled or not self.cache_file:
            return
            
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached translations")
            except Exception as e:
                logger.error(f"Error loading translation cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save translation cache to file"""
        if not self.enabled or not self.cache_file:
            return
            
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.cache)} cached translations")
        except Exception as e:
            logger.error(f"Error saving translation cache: {e}")
    
    def get_cached_translation(self, text, target_language, model_name):
        """Get cached translation if available"""
        if not self.enabled:
            return None
            
        cache_key = get_cache_key(text, target_language, model_name)
        return self.cache.get(cache_key)
    
    def cache_translation(self, text, target_language, model_name, translation):
        """Cache a translation"""
        if not self.enabled:
            return
            
        cache_key = get_cache_key(text, target_language, model_name)
        self.cache[cache_key] = translation

class GlossaryManager:
    """Manages translation glossary functionality"""
    
    def __init__(self):
        self.settings = config_manager.translation_enhancement_settings
        self.glossary = {}
        self.glossary_file = self.settings['glossary_file_path']
        self.enabled = self.settings['use_glossary']
        
        if self.enabled:
            self.load_glossary()
    
    def load_glossary(self):
        """Load glossary from file"""
        if not self.enabled or not self.glossary_file:
            return
            
        if os.path.exists(self.glossary_file):
            try:
                with open(self.glossary_file, 'r', encoding='utf-8') as f:
                    self.glossary = json.load(f)
                logger.info(f"Loaded {len(self.glossary)} glossary terms")
            except Exception as e:
                logger.error(f"Error loading glossary: {e}")
                self.glossary = {}
    
    def get_glossary_terms_in_text(self, text):
        """Find glossary terms present in the text"""
        if not self.enabled or not self.glossary:
            return {}
            
        found_terms = {}
        text_lower = text.lower()
        
        for source_term, target_term in self.glossary.items():
            if source_term.lower() in text_lower:
                found_terms[source_term] = target_term
        
        return found_terms

class QuotaManager:
    """Manages API quota and implements intelligent retry strategies"""
    
    def __init__(self):
        self.request_times = []
        self.max_requests_per_minute = 60
        self.retry_delays = [1, 2, 5, 10, 30]
        
    def can_make_request(self):
        """Check if we can make a request without hitting quota limits"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        return len(self.request_times) < self.max_requests_per_minute
    
    def record_request(self):
        """Record that a request was made"""
        self.request_times.append(time.time())
    
    def get_retry_delay(self, attempt):
        """Get retry delay for failed requests"""
        if attempt < len(self.retry_delays):
            return self.retry_delays[attempt]
        return self.retry_delays[-1]

class EnhancedErrorRecovery:
    """Enhanced error recovery with progressive retry and graceful degradation"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.max_retries = 3
        self.backoff_multiplier = 2
        
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                self.error_counts[error_type] += 1
                
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) * self.backoff_multiplier
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")
        
        raise last_exception

class DocumentTypeDetector:
    """Detects document type and provides specialized translation guidance"""
    
    DOCUMENT_TYPES = {
        'academic': {
            'keywords': ['abstract', 'methodology', 'conclusion', 'hypothesis', 'research', 'study', 'analysis'],
            'style': 'formal academic',
            'temperature': 0.05
        },
        'technical': {
            'keywords': ['specification', 'procedure', 'manual', 'configuration', 'system', 'protocol'],
            'style': 'technical precision',
            'temperature': 0.1
        },
        'legal': {
            'keywords': ['whereas', 'hereby', 'pursuant', 'jurisdiction', 'contract', 'clause'],
            'style': 'legal formal',
            'temperature': 0.05
        },
        'business': {
            'keywords': ['proposal', 'strategy', 'market', 'revenue', 'business', 'corporate'],
            'style': 'business professional',
            'temperature': 0.1
        },
        'literary': {
            'keywords': ['narrative', 'character', 'story', 'dialogue', 'chapter', 'novel'],
            'style': 'literary creative',
            'temperature': 0.15
        }
    }
    
    def detect_document_type(self, text_sample):
        """Detect document type from text sample"""
        text_lower = text_sample.lower()
        scores = {}
        
        for doc_type, config in self.DOCUMENT_TYPES.items():
            score = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            scores[doc_type] = score
        
        if not scores or max(scores.values()) == 0:
            return 'general', 'neutral', 0.1
        
        best_type = max(scores, key=scores.get)
        config = self.DOCUMENT_TYPES[best_type]
        
        return best_type, config['style'], config['temperature']

class EnhancedTranslationPromptGenerator:
    """Generates enhanced translation prompts based on document analysis"""
    
    def __init__(self):
        self.document_detector = DocumentTypeDetector()
        
    def generate_translation_prompt(self, text_to_translate, target_language,
                                  style_guide="", glossary_terms=None,
                                  prev_context="", next_context="",
                                  item_type="text block"):
        """Generate enhanced translation prompt with separator preservation instructions"""

        # Detect document type if no style guide provided
        if not style_guide:
            doc_type, detected_style, _ = self.document_detector.detect_document_type(text_to_translate)
            style_guide = f"Document type: {doc_type}. Style: {detected_style}"

        prompt_parts = [
            f"Translate the following {item_type} from English to {target_language}.",
            f"Style guide: {style_guide}",
            ""
        ]

        # Check if text contains item separators and add specific instructions
        if "%%%%ITEM_BREAK%%%%" in text_to_translate:
            prompt_parts.extend([
                "⚠️ CRITICAL INSTRUCTION: The text contains special separators '%%%%ITEM_BREAK%%%%'.",
                "These separators MUST be preserved EXACTLY as they are - do NOT translate, modify, or remove them.",
                "They are technical markers used for document processing.",
                ""
            ])

        # Add context if available
        if prev_context:
            prompt_parts.extend([
                "Previous context:",
                f"```{prev_context[:200]}...```",
                ""
            ])

        # Add glossary terms if available
        if glossary_terms:
            prompt_parts.extend([
                "Use these specific translations for key terms:",
                "\n".join([f"- {source} → {target}" for source, target in glossary_terms.items()]),
                ""
            ])

        # Add main content
        prompt_parts.extend([
            "Text to translate:",
            "```",
            text_to_translate,
            "```",
            "",
            f"Provide only the {target_language} translation, maintaining the original formatting and structure."
        ])

        # Add separator preservation reminder if needed
        if "%%%%ITEM_BREAK%%%%" in text_to_translate:
            prompt_parts.extend([
                "",
                "REMINDER: Keep all '%%%%ITEM_BREAK%%%%' separators exactly as they are!"
            ])

        # Add next context if available
        if next_context:
            prompt_parts.extend([
                "",
                "Next context (for reference only):",
                f"```{next_context[:200]}...```"
            ])

        return "\n".join(prompt_parts)

class TranslationService:
    """Main translation service that coordinates all translation functionality"""

    def __init__(self):
        self.gemini_settings = config_manager.gemini_settings
        self.translation_settings = config_manager.translation_enhancement_settings

        # Initialize components
        self.cache = TranslationCache()
        self.glossary = GlossaryManager()
        self.quota_manager = QuotaManager()
        self.error_recovery = EnhancedErrorRecovery()
        self.prompt_generator = EnhancedTranslationPromptGenerator()

        # Initialize advanced caching if available
        try:
            from advanced_caching import advanced_cache_manager
            self.advanced_cache = advanced_cache_manager
            self.use_advanced_cache = True
            logger.info("Advanced contextual caching enabled")
        except ImportError:
            self.advanced_cache = None
            self.use_advanced_cache = False
            logger.info("Using basic caching")

        # Initialize Gemini model
        if self.gemini_settings['api_key']:
            self.model = genai.GenerativeModel(self.gemini_settings['model_name'])
        else:
            self.model = None
            logger.warning("No API key available - translation will not work")
    
    async def translate_text(self, text, target_language=None, style_guide="",
                           prev_context="", next_context="", item_type="text block"):
        """Translate a single text item with advanced caching and Markdown awareness"""

        if not self.model:
            raise Exception("Translation service not properly initialized - no API key")

        if target_language is None:
            target_language = self.translation_settings['target_language']

        # Check if content is Markdown and use appropriate translation method
        if MARKDOWN_AWARE_AVAILABLE and markdown_translator.is_markdown_content(text):
            logger.debug("🔄 Using Markdown-aware translation")

            # Create async wrapper for this translation service
            async def translate_func(content, lang, style, prev_ctx, next_ctx, content_type):
                return await self._translate_raw_text(content, lang, style, prev_ctx, next_ctx, content_type)

            return await markdown_translator.translate_markdown_content(
                text, translate_func, target_language, prev_context, next_context
            )

        # Use standard translation for non-Markdown content
        return await self._translate_raw_text(text, target_language, style_guide,
                                            prev_context, next_context, item_type)

    async def _translate_raw_text(self, text, target_language, style_guide="",
                                prev_context="", next_context="", item_type="text block"):
        """Internal method for raw text translation (without Markdown processing)"""

        # Check advanced cache first if available
        if self.use_advanced_cache and self.advanced_cache:
            cached_result = self.advanced_cache.get_cached_translation(
                text, target_language, self.gemini_settings['model_name'],
                prev_context, next_context
            )
            if cached_result:
                logger.debug("Using advanced cached translation")
                return cached_result

        # Fallback to basic cache
        cached_result = self.cache.get_cached_translation(
            text, target_language, self.gemini_settings['model_name']
        )
        if cached_result:
            logger.debug("Using basic cached translation")
            return cached_result

        # Get glossary terms
        glossary_terms = self.glossary.get_glossary_terms_in_text(text)

        # Generate prompt
        prompt = self.prompt_generator.generate_translation_prompt(
            text, target_language, style_guide, glossary_terms,
            prev_context, next_context, item_type
        )

        # Wait for quota if needed
        while not self.quota_manager.can_make_request():
            await asyncio.sleep(1)

        # Make translation request with retry
        async def make_request():
            self.quota_manager.record_request()

            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.gemini_settings['temperature']
                )
            )

            # Enhanced response validation with detailed error handling
            if not response:
                raise Exception("No response received from translation API")

            # Check for safety/content filtering issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 2:  # SAFETY
                        raise Exception(f"Content filtered by safety settings. Text: {text[:100]}...")
                    elif finish_reason == 3:  # RECITATION
                        raise Exception(f"Content blocked due to recitation. Text: {text[:100]}...")
                    elif finish_reason == 4:  # OTHER
                        raise Exception(f"Content blocked for other reasons. Text: {text[:100]}...")

            # Check if response has valid text
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to extract text from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                        if text_parts:
                            return ''.join(text_parts).strip()

            # If we get here, no valid text was found
            raise Exception(f"Invalid response: The response contains no valid text. Finish reason: {getattr(getattr(response.candidates[0], 'finish_reason', None), 'name', 'unknown') if hasattr(response, 'candidates') and response.candidates else 'no candidates'}")

        try:
            translation = await self.error_recovery.execute_with_retry(make_request)

            # Cache the result in both caches
            self.cache.cache_translation(
                text, target_language, self.gemini_settings['model_name'], translation
            )

            if self.use_advanced_cache and self.advanced_cache:
                self.advanced_cache.cache_translation(
                    text, target_language, self.gemini_settings['model_name'],
                    translation, prev_context, next_context
                )

            return translation

        except Exception as e:
            # Enhanced error logging with more details
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'text_length': len(text),
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'target_language': target_language,
                'model_name': self.gemini_settings['model_name']
            }

            logger.error(f"Translation failed: {error_details['error_type']}: {error_details['error_message']}")
            logger.error(f"  Text preview: {error_details['text_preview']}")
            logger.error(f"  Target language: {error_details['target_language']}")
            logger.error(f"  Model: {error_details['model_name']}")
            raise

    async def translate_document(self, document, target_language=None, style_guide=""):
        """
        Translate a structured Document object.
        Only translates content blocks that should be translated (headings, paragraphs, etc.)
        Preserves non-translatable blocks (images, tables, code, etc.) unchanged.
        """
        if not STRUCTURED_MODEL_AVAILABLE:
            raise Exception("Structured document model not available for Document translation")

        if not isinstance(document, Document):
            raise ValueError(f"Expected Document object, got {type(document)}")

        if target_language is None:
            target_language = self.translation_settings['target_language']

        logger.info(f"🌐 Translating Document '{document.title}' to {target_language}")
        logger.info(f"📊 Document has {len(document.content_blocks)} content blocks")

        # Get translatable and non-translatable blocks
        translatable_blocks = document.get_translatable_blocks()
        non_translatable_blocks = document.get_non_translatable_blocks()

        logger.info(f"📝 Translating {len(translatable_blocks)} blocks, preserving {len(non_translatable_blocks)} blocks")

        # Translate blocks in parallel for much faster processing
        translated_blocks = await self._translate_blocks_parallel(
            translatable_blocks, target_language, style_guide
        )

        # Create new Document with translated content blocks
        # Merge translated and non-translatable blocks in original order
        all_translated_blocks = []
        translated_iter = iter(translated_blocks)

        for original_block in document.content_blocks:
            if original_block.block_type in {ContentType.HEADING, ContentType.PARAGRAPH,
                                           ContentType.LIST_ITEM, ContentType.CAPTION}:
                # Use translated version
                try:
                    all_translated_blocks.append(next(translated_iter))
                except StopIteration:
                    # Fallback to original if we run out of translated blocks
                    all_translated_blocks.append(original_block)
            elif original_block.block_type == ContentType.FOOTNOTE:
                # Keep footnotes in original language
                all_translated_blocks.append(original_block)
            else:
                # Keep non-translatable block as-is
                all_translated_blocks.append(original_block)

        # Create translated document
        translated_document = Document(
            title=f"{document.title} ({target_language})",
            content_blocks=all_translated_blocks,
            source_filepath=document.source_filepath,
            total_pages=document.total_pages,
            metadata={
                **document.metadata,
                'translated_to': target_language,
                'translation_method': 'structured_document_model',
                'original_title': document.title
            }
        )

        logger.info(f"✅ Document translation completed: {len(translated_blocks)} blocks translated")
        return translated_document

    async def _translate_blocks_parallel(self, translatable_blocks, target_language, style_guide):
        """
        Translate content blocks in parallel with enhanced concurrency and adaptive scaling.
        """
        import asyncio
        import psutil
        import multiprocessing
        
        try:
            from tqdm.asyncio import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False

        logger.info(f"🚀 Starting enhanced parallel translation of {len(translatable_blocks)} blocks...")

        # Enhanced concurrency calculation
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calculate optimal concurrency based on system resources
        base_concurrency = min(cpu_count, int(memory_gb / 1.5))  # 1 task per 1.5GB RAM
        enhanced_concurrency = min(max(base_concurrency, 8), 20)  # Min 8, max 20
        
        # Use adaptive semaphore based on system performance
        semaphore = asyncio.Semaphore(enhanced_concurrency)

        logger.info(f"⚡ Enhanced parallel processing configuration:")
        logger.info(f"   • System resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        logger.info(f"   • Base concurrency: {base_concurrency}")
        logger.info(f"   • Enhanced concurrency: {enhanced_concurrency}")
        logger.info(f"   • Blocks to process: {len(translatable_blocks)}")

        # Create translation tasks with enhanced error handling
        tasks = []
        for i, block in enumerate(translatable_blocks):
            task = self._translate_single_block_enhanced(block, target_language, style_guide, i, semaphore)
            tasks.append(task)

        # Execute with enhanced progress tracking and error handling
        results = []
        if TQDM_AVAILABLE and len(tasks) > 5:
            logger.info(f"📊 Enhanced progress tracking enabled for {len(tasks)} translation tasks")
            results = await tqdm.gather(*tasks,
                                      desc="🌐 Translating blocks (enhanced)",
                                      unit="block",
                                      colour="green")
        else:
            logger.info(f"📊 Processing {len(tasks)} translation tasks with enhanced concurrency...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Enhanced progress logging for smaller batches
            if len(tasks) <= 10:
                for i in range(0, len(tasks), max(1, len(tasks) // 5)):
                    progress = (i / len(tasks)) * 100
                    logger.info(f"📊 Translation progress: {progress:.1f}% ({i}/{len(tasks)} blocks)")

        # Process results with enhanced error handling and statistics
        translated_blocks = []
        successful_translations = 0
        failed_translations = 0
        total_processing_time = 0.0

        for i, result in enumerate(results):
            original_block = translatable_blocks[i]
            if isinstance(result, Exception):
                logger.warning(f"Translation failed for block {i+1}: {str(result)}")
                translated_blocks.append(original_block)  # Keep original on failure
                failed_translations += 1
            else:
                translated_blocks.append(result)
                successful_translations += 1
                # Track processing time if available
                if hasattr(result, 'processing_time'):
                    total_processing_time += result.processing_time

        # Enhanced completion statistics
        avg_processing_time = total_processing_time / successful_translations if successful_translations > 0 else 0
        success_rate = (successful_translations / len(translatable_blocks)) * 100

        logger.info(f"✅ Enhanced parallel translation completed:")
        logger.info(f"   • Successful: {successful_translations}/{len(translatable_blocks)} ({success_rate:.1f}%)")
        logger.info(f"   • Failed: {failed_translations}")
        logger.info(f"   • Average processing time: {avg_processing_time:.2f}s per block")
        logger.info(f"   • Concurrency used: {enhanced_concurrency}")
        logger.info(f"   • System efficiency: {successful_translations / enhanced_concurrency:.1f} blocks per concurrent task")

        return translated_blocks

    async def _translate_single_block_enhanced(self, block, target_language, style_guide, block_index, semaphore):
        """
        Enhanced single block translation with semaphore control and performance tracking.
        """
        import asyncio
        import time
        
        start_time = time.time()
        max_retries = 3  # Increased from 2 to 3
        retry_delay = 1.0

        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"Translating block {block_index+1}: {type(block).__name__}")

                    # Get content to translate based on block type
                    if isinstance(block, Heading):
                        content_to_translate = block.content
                    elif isinstance(block, Paragraph):
                        content_to_translate = block.content
                    elif isinstance(block, ListItem):
                        content_to_translate = block.content
                    elif isinstance(block, Footnote):
                        content_to_translate = block.content
                    elif isinstance(block, Caption):
                        content_to_translate = block.content
                    else:
                        # Fallback to original_text
                        content_to_translate = block.original_text

                    # Skip empty content
                    if not content_to_translate or not content_to_translate.strip():
                        logger.debug(f"Skipping empty content block: {type(block).__name__}")
                        return block

                    # Translate the content with improved prompt structure
                    translated_content = await self._translate_content_block(
                        content_to_translate,
                        target_language,
                        style_guide,
                        type(block).__name__.lower()
                    )

                    # Create new block with translated content and processing time
                    translated_block = self._create_translated_block(block, translated_content)
                    
                    # Add processing time metadata
                    processing_time = time.time() - start_time
                    if hasattr(translated_block, 'metadata'):
                        translated_block.metadata = getattr(translated_block, 'metadata', {})
                        translated_block.metadata['processing_time'] = processing_time
                    else:
                        translated_block.processing_time = processing_time
                    
                    return translated_block

                except Exception as e:
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.debug(f"Translation attempt {attempt + 1} failed for block {block_index+1}, retrying in {delay}s: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All translation attempts failed for block {block_index+1}: {str(e)}")
                        raise e

    async def _translate_content_block(self, content, target_language, style_guide, block_type):
        """Translate content with improved prompt structure to prevent leakage"""

        # Create structured prompt with clear delimiters
        system_prompt = f"You are a professional translator. Translate the user-provided {block_type} to {target_language}."
        if style_guide:
            system_prompt += f" Style guide: {style_guide}"

        user_prompt = f"<TEXT_TO_TRANSLATE>\n{content}\n</TEXT_TO_TRANSLATE>"

        # Combine prompts with clear separation
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nProvide only the {target_language} translation, maintaining the original formatting."

        # Use existing translation infrastructure
        return await self._translate_raw_text(
            content, target_language, style_guide, "", "", block_type
        )

    def _create_translated_block(self, original_block, translated_content):
        """Create a new content block with translated content"""
        # Create a copy of the original block with translated content
        if isinstance(original_block, Heading):
            return Heading(
                block_type=original_block.block_type,
                original_text=translated_content,  # Store translation as new original
                page_num=original_block.page_num,
                bbox=original_block.bbox,
                block_num=original_block.block_num,
                formatting=original_block.formatting,
                level=original_block.level,
                content=translated_content
            )
        elif isinstance(original_block, Paragraph):
            return Paragraph(
                block_type=original_block.block_type,
                original_text=translated_content,
                page_num=original_block.page_num,
                bbox=original_block.bbox,
                block_num=original_block.block_num,
                formatting=original_block.formatting,
                content=translated_content
            )
        elif isinstance(original_block, ListItem):
            return ListItem(
                block_type=original_block.block_type,
                original_text=translated_content,
                page_num=original_block.page_num,
                bbox=original_block.bbox,
                block_num=original_block.block_num,
                formatting=original_block.formatting,
                content=translated_content,
                list_type=original_block.list_type,
                level=original_block.level
            )
        elif isinstance(original_block, Footnote):
            return Footnote(
                block_type=original_block.block_type,
                original_text=translated_content,
                page_num=original_block.page_num,
                bbox=original_block.bbox,
                block_num=original_block.block_num,
                formatting=original_block.formatting,
                content=translated_content,
                reference_id=original_block.reference_id,
                source_block=original_block.source_block
            )
        elif isinstance(original_block, Caption):
            return Caption(
                block_type=original_block.block_type,
                original_text=translated_content,
                page_num=original_block.page_num,
                bbox=original_block.bbox,
                block_num=original_block.block_num,
                formatting=original_block.formatting,
                content=translated_content,
                target_type=original_block.target_type
            )
        else:
            # Fallback: return original block (shouldn't happen for translatable blocks)
            return original_block

    def translate_text_sync(self, text, target_language=None, style_guide="",
                          prev_context="", next_context="", item_type="text block"):
        """Synchronous wrapper for translate_text (for compatibility)"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.translate_text(
                    text, target_language, style_guide, prev_context, next_context, item_type
                ))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.translate_text(
                text, target_language, style_guide, prev_context, next_context, item_type
            ))

    def save_caches(self):
        """Save all caches to disk"""
        self.cache.save_cache()

        if self.use_advanced_cache and self.advanced_cache:
            self.advanced_cache.save_cache()

    def get_cache_statistics(self):
        """Get comprehensive cache statistics"""
        stats = {'basic_cache': {'total_entries': len(self.cache.cache)}}

        if self.use_advanced_cache and self.advanced_cache:
            stats['advanced_cache'] = self.advanced_cache.get_cache_statistics()

        return stats

# Global translation service instance
translation_service = TranslationService()
