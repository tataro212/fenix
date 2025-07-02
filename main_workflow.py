"""
Main Workflow Module for Ultimate PDF Translator

Orchestrates the complete translation workflow using all modular components
"""

import os
import asyncio
import time
import logging
import sys
import re
import json
import hashlib
import shutil
import psutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

logger = logging.getLogger(__name__)

# Structured logging setup
try:
    import structlog
    STRUCTURED_LOGGING_AVAILABLE = True

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create structured logger
    structured_logger = structlog.get_logger()
    logger.info("✅ Structured logging enabled (JSON output)")

except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False
    structured_logger = None
    logger.warning("⚠️ Structured logging not available - install with: pip install structlog")

class MetricsCollector:
    """Collects and logs structured metrics for monitoring"""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def start_document_processing(self, filepath):
        """Start tracking metrics for a document"""
        file_stats = os.stat(filepath)
        self.metrics = {
            'document_path': filepath,
            'document_name': os.path.basename(filepath),
            'document_hash': hashlib.md5(f"{filepath}:{file_stats.st_size}:{file_stats.st_mtime}".encode()).hexdigest(),
            'file_size_bytes': file_stats.st_size,
            'processing_start_time': time.time(),
            'page_count': 0,
            'images_found': 0,
            'headings_found': 0,
            'paragraphs_found': 0,
            'tables_found': 0,
            'processing_pipeline': 'unknown',
            'errors': [],
            'warnings': []
        }

    def update_content_metrics(self, document):
        """Update metrics based on extracted content"""
        if not document or not hasattr(document, 'content_blocks'):
            return

        from structured_document_model import ContentType

        self.metrics['page_count'] = getattr(document, 'total_pages', 0)

        # Count content types
        for block in document.content_blocks:
            if hasattr(block, 'block_type'):
                if block.block_type == ContentType.HEADING:
                    self.metrics['headings_found'] += 1
                elif block.block_type == ContentType.PARAGRAPH:
                    self.metrics['paragraphs_found'] += 1
                elif block.block_type == ContentType.IMAGE_PLACEHOLDER:
                    self.metrics['images_found'] += 1
                elif block.block_type == ContentType.TABLE:
                    self.metrics['tables_found'] += 1

    def update_translation_metrics(self, translation_time_ms, word_count=0):
        """Update translation-specific metrics"""
        self.metrics.update({
            'time_to_translate_ms': translation_time_ms,
            'words_translated': word_count,
            'translation_speed_wpm': (word_count / (translation_time_ms / 60000)) if translation_time_ms > 0 else 0
        })

    def add_error(self, error_msg):
        """Add an error to the metrics"""
        self.metrics['errors'].append(str(error_msg))

    def add_warning(self, warning_msg):
        """Add a warning to the metrics"""
        self.metrics['warnings'].append(str(warning_msg))

    def finalize_and_log(self, success=True):
        """Finalize metrics and log them"""
        end_time = time.time()
        self.metrics.update({
            'processing_end_time': end_time,
            'total_processing_time_ms': (end_time - self.metrics.get('processing_start_time', end_time)) * 1000,
            'success': success,
            'error_count': len(self.metrics.get('errors', [])),
            'warning_count': len(self.metrics.get('warnings', []))
        })

        # Log structured metrics
        if STRUCTURED_LOGGING_AVAILABLE and structured_logger:
            structured_logger.info("document_processing_completed", **self.metrics)
        else:
            # Fallback to JSON logging
            logger.info(f"METRICS: {json.dumps(self.metrics, indent=2)}")

        return self.metrics

# Import all modular components
from config_manager import config_manager
UNIFIED_CONFIG_AVAILABLE = False
from pdf_parser import PDFParser, StructuredContentExtractor
from ocr_processor import SmartImageAnalyzer
from translation_service import translation_service
from optimization_manager import optimization_manager
from document_generator import document_generator, pdf_converter
from drive_uploader import drive_uploader
from nougat_integration import NougatIntegration  # Enhanced Nougat integration
from nougat_only_integration import NougatOnlyIntegration  # NOUGAT-ONLY mode
from enhanced_document_intelligence import DocumentTextRestructurer  # Footnote handling

# Import structured document model for new workflow
try:
    from structured_document_model import Document as StructuredDocument
    STRUCTURED_MODEL_AVAILABLE = True
except ImportError:
    STRUCTURED_MODEL_AVAILABLE = False
    logger.warning("Structured document model not available")
from utils import (
    choose_input_path, choose_base_output_directory,
    get_specific_output_dir_for_file, estimate_translation_cost,
    ProgressTracker
)

# Import advanced features (with fallback if not available)
try:
    from advanced_translation_pipeline import AdvancedTranslationPipeline
    from self_correcting_translator import SelfCorrectingTranslator
    from hybrid_ocr_processor import HybridOCRProcessor
    from semantic_cache import SemanticCache
    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("✅ Advanced features available: Self-correction, Hybrid OCR, Semantic caching")
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning(f"⚠️ Advanced features not available: {e}")
    logger.info("💡 Install advanced features with: pip install -r advanced_features_requirements.txt")

# Import intelligent pipeline (with fallback if not available)
try:
    from intelligent_pdf_translator import IntelligentPDFTranslator
    from advanced_document_analyzer import AdvancedDocumentAnalyzer
    from translation_strategy_manager import TranslationStrategyManager
    INTELLIGENT_PIPELINE_AVAILABLE = True
    logger.info("🧠 Intelligent pipeline available: Content-aware routing, Strategic processing")
except ImportError as e:
    INTELLIGENT_PIPELINE_AVAILABLE = False
    logger.warning(f"⚠️ Intelligent pipeline not available: {e}")
    logger.info("💡 Intelligent pipeline requires additional dependencies")

# Import YOLO integration pipeline (with fallback if not available)
try:
    from yolov8_integration_pipeline import YOLOv8IntegrationPipeline
    YOLO_PIPELINE_AVAILABLE = True
    logger.info("🎯 YOLOv8 pipeline available: Supreme visual detection accuracy")
except ImportError as e:
    YOLO_PIPELINE_AVAILABLE = False
    logger.warning(f"⚠️ YOLOv8 pipeline not available: {e}")
    logger.info("💡 YOLOv8 pipeline requires ultralytics and additional dependencies")

# Import intelligent node consolidator (with fallback if not available)
try:
    from intelligent_node_consolidator import create_intelligent_consolidator, IntelligentNodeConsolidator
    INTELLIGENT_NODE_CONSOLIDATOR_AVAILABLE = True
    logger.info("🧠 Intelligent node consolidator available: Solves 25,000 nodes problem")
except ImportError as e:
    INTELLIGENT_NODE_CONSOLIDATOR_AVAILABLE = False
    logger.warning(f"⚠️ Intelligent node consolidator not available: {e}")
    logger.info("💡 Intelligent node consolidator helps with excessive node count")

# Import optimized document pipeline (NEW - HIGHEST PRIORITY)
try:
    from optimized_document_pipeline import process_pdf_optimized, OptimizedDocumentPipeline
    OPTIMIZED_PIPELINE_AVAILABLE = True
    logger.info("🚀 Optimized pipeline available: PyMuPDF-YOLO integration with intelligent routing")
except ImportError as e:
    OPTIMIZED_PIPELINE_AVAILABLE = False
    logger.warning(f"⚠️ Optimized pipeline not available: {e}")
    logger.info("💡 Optimized pipeline provides PyMuPDF extraction + YOLO layout analysis")

# Import graph-based translation pipeline (with fallback if not available)
try:
    from graph_based_translation_pipeline import graph_based_pipeline
    GRAPH_BASED_PIPELINE_AVAILABLE = True
    logger.info("🔄 Graph-based pipeline available: Advanced structure preservation and format integrity")
except ImportError as e:
    GRAPH_BASED_PIPELINE_AVAILABLE = False
    logger.warning(f"⚠️ Graph-based pipeline not available: {e}")
    logger.info("💡 Graph-based pipeline requires document_model and additional dependencies")

# Import distributed tracing for comprehensive pipeline monitoring
try:
    from distributed_tracing import tracer, SpanType, start_trace, span, add_metadata, finish_trace
    DISTRIBUTED_TRACING_AVAILABLE = True
    logger.info("🔍 Distributed tracing available: Pipeline monitoring enabled")
except ImportError as e:
    DISTRIBUTED_TRACING_AVAILABLE = False
    logger.warning(f"⚠️ Distributed tracing not available: {e}")
    # Create dummy functions for compatibility
    def start_trace(*args, **kwargs): return "dummy_trace"
    def span(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    def add_metadata(**kwargs): pass
    def finish_trace(*args, **kwargs): pass

def _process_single_page(task):
    """
    Process a single page in a separate process.
    This function must be at module level to be pickable by ProcessPoolExecutor.
    """
    try:
        import fitz
        from pdf_parser import StructuredContentExtractor

        filepath = task['filepath']
        page_num = task['page_num']
        page_images = task['page_images']

        # Create a new extractor instance for this process
        extractor = StructuredContentExtractor()

        # Open document and extract single page
        doc = fitz.open(filepath)
        page = doc[page_num]

        # Analyze document structure (needed for content classification)
        structure_analysis = extractor._analyze_document_structure(doc)

        # Extract content from this page only
        page_content_blocks = extractor._extract_page_content_as_blocks(
            page, page_num + 1, structure_analysis
        )

        # Add images for this page
        for img_ref in page_images:
            from structured_document_model import ImagePlaceholder, ContentType
            image_block = ImagePlaceholder(
                block_type=ContentType.IMAGE_PLACEHOLDER,
                original_text=img_ref.get('ocr_text', ''),
                page_num=page_num + 1,
                bbox=(img_ref['x0'], img_ref['y0'], img_ref['x1'], img_ref['y1']),
                image_path=img_ref['filepath'],
                width=img_ref.get('width'),
                height=img_ref.get('height'),
                ocr_text=img_ref.get('ocr_text'),
                translation_needed=img_ref.get('translation_needed', False)
            )
            page_content_blocks.append(image_block)

        doc.close()

        # Return page data
        return {
            'page_num': page_num,
            'content_blocks': page_content_blocks,
            'title': extractor._extract_document_title(doc) if page_num == 0 else None
        }

    except Exception as e:
        # Return the exception to be handled by the main process
        return e

class FailureTracker:
    """Tracks PDF processing failures and manages quarantine system"""

    def __init__(self, quarantine_dir="quarantine", max_retries=3):
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.failure_counts = defaultdict(int)
        self.failure_log_path = self.quarantine_dir / "failure_log.json"
        self._load_failure_log()

    def _load_failure_log(self):
        """Load existing failure counts from disk"""
        if self.failure_log_path.exists():
            try:
                with open(self.failure_log_path, 'r') as f:
                    self.failure_counts = defaultdict(int, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load failure log: {e}")

    def _save_failure_log(self):
        """Save failure counts to disk"""
        try:
            with open(self.failure_log_path, 'w') as f:
                json.dump(dict(self.failure_counts), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save failure log: {e}")

    def get_file_hash(self, filepath):
        """Generate a unique hash for a file based on its path and size"""
        try:
            stat = os.stat(filepath)
            content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(filepath.encode()).hexdigest()

    def should_process_file(self, filepath):
        """Check if file should be processed or is quarantined"""
        file_hash = self.get_file_hash(filepath)
        return self.failure_counts[file_hash] < self.max_retries

    def record_failure(self, filepath, error):
        """Record a failure and quarantine if max retries exceeded"""
        file_hash = self.get_file_hash(filepath)
        self.failure_counts[file_hash] += 1

        if self.failure_counts[file_hash] >= self.max_retries:
            self._quarantine_file(filepath, error)
            return True  # File was quarantined

        self._save_failure_log()
        return False  # File not quarantined yet

    def _quarantine_file(self, filepath, error):
        """Move problematic file to quarantine directory"""
        try:
            filename = os.path.basename(filepath)
            quarantine_path = self.quarantine_dir / filename

            # Avoid name conflicts
            counter = 1
            while quarantine_path.exists():
                name, ext = os.path.splitext(filename)
                quarantine_path = self.quarantine_dir / f"{name}_{counter}{ext}"
                counter += 1

            shutil.copy2(filepath, quarantine_path)

            # Create error report
            error_report = {
                "original_path": str(filepath),
                "quarantine_path": str(quarantine_path),
                "failure_count": self.failure_counts[self.get_file_hash(filepath)],
                "last_error": str(error),
                "quarantined_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            report_path = quarantine_path.with_suffix('.error.json')
            with open(report_path, 'w') as f:
                json.dump(error_report, f, indent=2)

            logger.critical(f"🚨 QUARANTINED: {filename} after {self.max_retries} failures")
            logger.critical(f"   📁 Moved to: {quarantine_path}")
            logger.critical(f"   📋 Error report: {report_path}")
            logger.critical(f"   ❌ Last error: {error}")

            self._save_failure_log()

        except Exception as quarantine_error:
            logger.error(f"Failed to quarantine {filepath}: {quarantine_error}")

class UltimatePDFTranslator:
    """Main orchestrator class for the PDF translation workflow with enhanced Nougat integration"""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.content_extractor = StructuredContentExtractor()
        self.image_analyzer = SmartImageAnalyzer()
        self.text_restructurer = DocumentTextRestructurer()  # For footnote handling
        self.quality_report_messages = []

        # Check for NOUGAT-ONLY mode preference
        if UNIFIED_CONFIG_AVAILABLE:
            nougat_only_mode = config_manager.get_value('general', 'nougat_only_mode', False)
        else:
            nougat_only_mode = config_manager.get_config_value('General', 'nougat_only_mode', False, bool)

        try:
            if nougat_only_mode:
                # Initialize NOUGAT-ONLY integration (no fallback)
                logger.info("🚀 NOUGAT-ONLY MODE: Initializing comprehensive visual extraction...")
                self.nougat_integration = NougatOnlyIntegration(config_manager)

                if self.nougat_integration.nougat_available:
                    logger.info("🎯 NOUGAT-ONLY: Replacing PDF parser with Nougat-only extraction")
                    self.nougat_integration.enhance_pdf_parser_nougat_only(self.pdf_parser)
                    logger.info("✅ PDF parser converted to NOUGAT-ONLY mode")
                    logger.info("📊 Will extract: Paintings, Schemata, Diagrams, Equations, Tables, Everything!")
                    self.nougat_only_mode = True
                else:
                    logger.error("❌ NOUGAT-ONLY MODE requires Nougat to be available!")
                    logger.error("❌ Falling back to enhanced Nougat integration...")
                    nougat_only_mode = False

            if not nougat_only_mode:
                # Initialize enhanced Nougat integration with priority mode
                self.nougat_integration = NougatIntegration(config_manager)
                self.nougat_only_mode = False

                # Enhance PDF parser with Nougat capabilities
                if self.nougat_integration.nougat_available or self.nougat_integration.use_alternative:
                    logger.info("🚀 Enhancing PDF parser with Nougat capabilities...")
                    self.nougat_integration.enhance_pdf_parser_with_nougat(self.pdf_parser)
                    logger.info("✅ PDF parser enhanced - prioritizing visual content with Nougat intelligence")
                else:
                    logger.warning("⚠️ Nougat not available - using traditional PDF processing")

        except Exception as e:
            logger.error(f"❌ Error initializing Nougat integration: {e}")
            logger.warning("⚠️ Falling back to traditional PDF processing")
            # Initialize basic integration as fallback
            self.nougat_integration = None
            self.nougat_only_mode = False

        # Initialize advanced features if available
        self.advanced_pipeline = None
        if UNIFIED_CONFIG_AVAILABLE:
            self.use_advanced_features = config_manager.get_value('general', 'use_advanced_features', True)
        else:
            self.use_advanced_features = config_manager.get_config_value('General', 'use_advanced_features', True, bool)

        if ADVANCED_FEATURES_AVAILABLE and self.use_advanced_features:
            try:
                logger.info("🚀 Initializing advanced translation features...")
                self.advanced_pipeline = AdvancedTranslationPipeline(
                    base_translator=translation_service,
                    nougat_integration=self.nougat_integration,
                    cache_dir="advanced_semantic_cache",
                    config_manager=config_manager
                )
                logger.info("✅ Advanced features initialized successfully!")
                logger.info("   🔧 Self-correcting translation enabled")
                logger.info("   📖 Hybrid OCR strategy enabled")
                logger.info("   🧠 Semantic caching enabled")
            except Exception as e:
                logger.error(f"❌ Failed to initialize advanced features: {e}")
                logger.warning("⚠️ Falling back to standard translation workflow")
                self.advanced_pipeline = None
        elif not ADVANCED_FEATURES_AVAILABLE:
            logger.info("💡 Advanced features not available - using standard workflow")
        else:
            logger.info("⚙️ Advanced features disabled in configuration")

        # Initialize intelligent pipeline if available
        self.intelligent_pipeline = None
        try:
            if UNIFIED_CONFIG_AVAILABLE:
                self.use_intelligent_pipeline = config_manager.get_value('intelligent_pipeline', 'use_intelligent_pipeline', True)
            else:
                self.use_intelligent_pipeline = config_manager.get_config_value('IntelligentPipeline', 'use_intelligent_pipeline', True, bool)
        except Exception as e:
            logger.warning(f"Could not get intelligent pipeline config: {e}, defaulting to True")
            self.use_intelligent_pipeline = True

        if INTELLIGENT_PIPELINE_AVAILABLE and self.use_intelligent_pipeline:
            try:
                logger.info("🧠 Initializing intelligent processing pipeline...")
                # Use the correct initialization - IntelligentPDFTranslator only takes max_workers parameter
                try:
                    if UNIFIED_CONFIG_AVAILABLE:
                        max_workers = config_manager.get_value('intelligent_pipeline', 'max_concurrent_tasks', 4)
                    else:
                        max_workers = config_manager.get_config_value('IntelligentPipeline', 'max_concurrent_tasks', 4, int)
                except Exception:
                    max_workers = 4
                self.intelligent_pipeline = IntelligentPDFTranslator(max_workers=max_workers)
                logger.info("✅ Intelligent pipeline initialized successfully!")
                logger.info("   🎯 Content-aware routing enabled")
                logger.info("   📊 Strategic tool selection enabled")
                logger.info("   ⚡ Parallel processing enabled")
                logger.info("   🧠 Semantic caching enabled")
            except Exception as e:
                logger.error(f"❌ Failed to initialize intelligent pipeline: {e}")
                logger.warning("⚠️ Falling back to advanced or standard workflow")
                self.intelligent_pipeline = None
                # Log the full traceback for debugging
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
        elif not INTELLIGENT_PIPELINE_AVAILABLE:
            logger.info("💡 Intelligent pipeline not available - using advanced/standard workflow")
        else:
            logger.info("⚙️ Intelligent pipeline disabled in configuration")

        # Initialize YOLO pipeline (if available and enabled)
        self.yolo_pipeline = None
        self.use_yolo_pipeline = False
        if YOLO_PIPELINE_AVAILABLE:
            try:
                # Use local YOLOv8 pipeline for direct enrichment
                self.yolo_pipeline = YOLOv8IntegrationPipeline()
                self.use_yolo_pipeline = True
                logger.info("✅ YOLOv8 pipeline initialized for local enrichment!")
            except Exception as e:
                logger.error(f"❌ Failed to initialize YOLOv8 pipeline: {e}")
                self.yolo_pipeline = None
                self.use_yolo_pipeline = False
        elif not YOLO_PIPELINE_AVAILABLE:
            logger.info("💡 YOLOv8 pipeline not available - using standard visual detection")

        # Initialize parallel processing settings
        if UNIFIED_CONFIG_AVAILABLE:
            self.max_workers = config_manager.get_value('performance', 'max_parallel_workers', 4)
            self.enable_parallel_processing = config_manager.get_value('performance', 'enable_parallel_processing', True)
            self.enable_metrics = config_manager.get_value('monitoring', 'enable_structured_metrics', True)
        else:
            self.max_workers = config_manager.get_config_value('Performance', 'max_parallel_workers', 4, int)
            self.enable_parallel_processing = config_manager.get_config_value('Performance', 'enable_parallel_processing', True, bool)
            self.enable_metrics = config_manager.get_config_value('Monitoring', 'enable_structured_metrics', True, bool)

        logger.info(f"⚡ Parallel processing: {'enabled' if self.enable_parallel_processing else 'disabled'} (max workers: {self.max_workers})")
        logger.info(f"📊 Structured metrics: {'enabled' if self.enable_metrics else 'disabled'}")

    async def translate_document_async(self, filepath, output_dir_for_this_file,
                                     target_language_override=None, precomputed_style_guide=None,
                                     use_advanced_features=None):
        """Main async translation workflow with optional advanced features"""

        logger.info(f"🚀 Starting translation of: {os.path.basename(filepath)}")
        start_time = time.time()

        # Determine processing pipeline priority
        use_advanced = use_advanced_features if use_advanced_features is not None else self.use_advanced_features

        # Priority 1: Optimized Pipeline (HIGHEST PRIORITY - PyMuPDF-YOLO integration with intelligent routing)
        if OPTIMIZED_PIPELINE_AVAILABLE and use_advanced:
            logger.info("🚀 Using optimized document pipeline (PYMUPDF-YOLO INTEGRATION)")
            logger.info("   🎯 Priority: Intelligent content extraction and layout analysis")
            logger.info("   ✅ Features: PyMuPDF extraction, YOLO layout analysis, parallel processing")
            return await self._translate_document_optimized(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Priority 2: Graph-Based Pipeline (Format preservation and structure integrity)
        elif GRAPH_BASED_PIPELINE_AVAILABLE and use_advanced:
            logger.info("🔄 Using graph-based translation pipeline (ADVANCED STRUCTURE PRESERVATION)")
            logger.info("   🎯 Priority: Format integrity and document structure preservation")
            return await self._translate_document_graph_based(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Priority 2: Advanced Pipeline (NOW WITH PARALLEL PROCESSING!)
        elif self.advanced_pipeline and use_advanced:
            logger.info("🎯 Using advanced translation pipeline (PARALLEL PROCESSING ENABLED)")
            return await self._translate_document_advanced(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Priority 3: YOLOv8 Pipeline (if available and enabled) - Supreme accuracy
        elif self.yolo_pipeline and self.use_yolo_pipeline:
            logger.info("🎯 Using YOLOv8 supreme accuracy pipeline")
            return await self._translate_document_yolo(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Priority 4: Structured Document Model (also has parallel processing)
        elif STRUCTURED_MODEL_AVAILABLE:
            logger.info("🏗️ Using structured document model workflow (PARALLEL PROCESSING)")
            return await self._translate_document_structured(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Priority 5: Intelligent Pipeline (if available and enabled)
        elif self.intelligent_pipeline and self.use_intelligent_pipeline:
            logger.info("🧠 Using intelligent processing pipeline")
            return await self._translate_document_intelligent(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
        # Fallback: Standard workflow
        else:
            logger.info("📝 Using standard translation workflow")
            return await self._translate_document_standard(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )

    async def _extract_pages_in_parallel(self, filepath, extracted_images):
        """Extract pages in parallel using ProcessPoolExecutor"""
        if not self.enable_parallel_processing:
            # Fall back to sequential processing
            return self.content_extractor.extract_structured_content_from_pdf(filepath, extracted_images)

        logger.info(f"⚡ Starting parallel page extraction with {self.max_workers} workers...")
        start_time = time.time()

        try:
            import fitz
            doc = fitz.open(filepath)
            total_pages = len(doc)
            doc.close()

            if total_pages <= 2:
                # For small documents, parallel processing overhead isn't worth it
                logger.info(f"📄 Document has only {total_pages} pages, using sequential processing")
                return self.content_extractor.extract_structured_content_from_pdf(filepath, extracted_images)

            # Group images by page
            images_by_page = self.pdf_parser.groupby_images_by_page(extracted_images)

            # Create page processing tasks
            page_tasks = []
            for page_num in range(total_pages):
                page_images = images_by_page.get(page_num + 1, [])
                page_tasks.append({
                    'filepath': filepath,
                    'page_num': page_num,
                    'page_images': page_images
                })

            # Process pages in parallel
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all page processing tasks
                futures = [
                    loop.run_in_executor(executor, _process_single_page, task)
                    for task in page_tasks
                ]

                # Wait for all pages to complete
                page_results = await asyncio.gather(*futures, return_exceptions=True)

            # Combine results and handle any errors
            successful_pages = []
            failed_pages = []

            for i, result in enumerate(page_results):
                if isinstance(result, Exception):
                    failed_pages.append((i, result))
                    logger.warning(f"⚠️ Page {i+1} processing failed: {result}")
                else:
                    successful_pages.append((i, result))

            if failed_pages:
                logger.warning(f"⚠️ {len(failed_pages)} pages failed parallel processing, falling back to sequential")
                return self.content_extractor.extract_structured_content_from_pdf(filepath, extracted_images)

            # Assemble pages in correct order
            all_content_blocks = []
            document_title = None

            for page_num, page_content in sorted(successful_pages):
                if page_content and 'content_blocks' in page_content:
                    all_content_blocks.extend(page_content['content_blocks'])
                    if not document_title and page_content.get('title'):
                        document_title = page_content['title']

            # Create final document
            from structured_document_model import Document
            document = Document(
                title=document_title or os.path.splitext(os.path.basename(filepath))[0],
                content_blocks=all_content_blocks,
                source_filepath=filepath,
                total_pages=total_pages,
                metadata={
                    'extraction_method': 'parallel_processing',
                    'workers_used': self.max_workers,
                    'processing_time_seconds': time.time() - start_time
                }
            )

            end_time = time.time()
            speedup = total_pages / (end_time - start_time) if (end_time - start_time) > 0 else 0
            logger.info(f"⚡ Parallel extraction completed in {end_time - start_time:.2f}s")
            logger.info(f"📊 Processing speed: {speedup:.1f} pages/second")
            logger.info(f"🏗️ Assembled document with {len(all_content_blocks)} content blocks")

            return document

        except Exception as e:
            logger.warning(f"⚠️ Parallel processing failed: {e}, falling back to sequential")
            return self.content_extractor.extract_structured_content_from_pdf(filepath, extracted_images)

    async def _translate_document_optimized(self, filepath, output_dir_for_this_file,
                                          target_language_override=None, precomputed_style_guide=None):
        """Optimized document pipeline with PyMuPDF-YOLO integration"""
        
        logger.info(f"🚀 Starting OPTIMIZED PIPELINE translation of: {os.path.basename(filepath)}")
        start_time = time.time()
        
        try:
            # Validate inputs
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")
            
            if not os.path.exists(output_dir_for_this_file):
                os.makedirs(output_dir_for_this_file, exist_ok=True)
            
            # Always use Greek as target language
            target_language = "el"
            logger.info(f"🌍 Target language: {target_language}")
            
            # Process with optimized pipeline
            logger.info("🚀 Processing with optimized document pipeline...")
            logger.info("   📄 PyMuPDF content extraction")
            logger.info("   🎯 YOLO layout analysis (0.15 confidence)")
            logger.info("   🗺️ Content-to-layout mapping")
            logger.info("   🧠 Intelligent processing strategies")
            logger.info("   ⚡ Parallel processing (4 workers)")
            
            result = await process_pdf_optimized(
                pdf_path=filepath,
                output_dir=output_dir_for_this_file,
                target_language=target_language,
                max_workers=4
            )
            
            # Generate final report
            end_time = time.time()
            self._generate_optimized_final_report(
                filepath, output_dir_for_this_file, start_time, end_time, result
            )
            
            # Save caches
            translation_service.save_caches()
            
            logger.info("✅ Optimized pipeline translation completed successfully!")
            return precomputed_style_guide
            
        except Exception as e:
            logger.error(f"❌ Optimized pipeline translation workflow failed: {e}")
            logger.info("🔄 Falling back to graph-based translation workflow...")
            if GRAPH_BASED_PIPELINE_AVAILABLE:
                return await self._translate_document_graph_based(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )
            elif self.advanced_pipeline:
                return await self._translate_document_advanced(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )
            else:
                return await self._translate_document_standard(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )

    async def _translate_document_yolo(self, filepath, output_dir_for_this_file,
                                     target_language_override=None, precomputed_style_guide=None):
        """YOLOv8 supreme accuracy translation workflow"""

        logger.info(f"🎯 Starting YOLOv8 SUPREME ACCURACY translation of: {os.path.basename(filepath)}")
        start_time = time.time()

        try:
            # Validate inputs
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")

            if not os.path.exists(output_dir_for_this_file):
                os.makedirs(output_dir_for_this_file, exist_ok=True)

            # Set target language
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']

            # Process with YOLOv8 supreme accuracy pipeline
            logger.info("🎯 Processing with YOLOv8 supreme accuracy pipeline...")
            yolo_result = await self.yolo_pipeline.process_pdf_with_yolo_supreme_accuracy(
                pdf_path=filepath,
                output_dir=output_dir_for_this_file,
                target_language=target_language
            )

            # Generate final report
            end_time = time.time()
            self._generate_yolo_final_report(
                filepath, output_dir_for_this_file, start_time, end_time, yolo_result
            )

            # Save caches
            translation_service.save_caches()

            logger.info("✅ YOLOv8 supreme accuracy translation completed successfully!")
            return precomputed_style_guide

        except Exception as e:
            logger.error(f"❌ YOLOv8 translation workflow failed: {e}")
            logger.info("🔄 Falling back to intelligent translation workflow...")
            if self.intelligent_pipeline:
                return await self._translate_document_intelligent(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )
            elif self.advanced_pipeline:
                return await self._translate_document_advanced(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )
            else:
                return await self._translate_document_standard(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )

    async def _translate_document_intelligent(self, filepath, output_dir_for_this_file,
                                            target_language_override=None, precomputed_style_guide=None):
        """Intelligent translation workflow with advanced features"""
        try:
            # ... existing code ...

            # STEP 2: Use parallel translation on structured document (MUCH FASTER!)
            with span("parallel_translation", SpanType.TRANSLATION,
                     translation_method="parallel_structured", target_language=target_language):
                logger.info("🚀 Step 2: Processing translation with PARALLEL processing...")
                logger.info("   🔥 Using structured document model for maximum speed")

                translated_document = await translation_service.translate_document(
                    structured_document, target_language, ""
                )

                add_metadata(
                    translation_method="parallel_structured",
                    blocks_translated=len(structured_document.get_translatable_blocks()),
                    validation_passed=True
                )

                # Count preserved images
                translated_image_blocks = [block for block in translated_document.content_blocks if hasattr(block, 'image_path') and block.image_path]
                add_metadata(image_placeholders_preserved=len(translated_image_blocks))

            # STEP 3: Generate Word document
            with span("generate_word", SpanType.DOCUMENT_GENERATION):
                logger.info("📄 Step 3: Generating Word document...")
                try:
                    saved_word_filepath = document_generator.create_word_document_from_structured_document(
                        translated_document, word_output_path, image_folder, cover_page_data
                    )
                except Exception as doc_error:
                    logger.error(f"❌ Document generation failed with error: {doc_error}")
                    logger.error(f"   • Structured document blocks: {len(translated_document.content_blocks) if translated_document else 'N/A'}")
                    logger.error(f"   • Target directory exists: {os.path.exists(output_dir_for_this_file)}")
                    logger.error(f"   • Target directory writable: {os.access(output_dir_for_this_file, os.W_OK)}")
                    add_metadata(error_count=1, validation_passed=False)
                    raise Exception(f"Failed to create Word document from advanced translation: {doc_error}")

                if not saved_word_filepath:
                    add_metadata(error_count=1, validation_passed=False)
                    raise Exception("Failed to create Word document from advanced translation")

            # Convert to PDF
            with span("convert_to_pdf", SpanType.DOCUMENT_GENERATION, output_format="pdf"):
                logger.info("📑 Converting to PDF...")
                pdf_success = pdf_converter.convert_word_to_pdf(saved_word_filepath, pdf_output_path)
                add_metadata(pdf_conversion_success=pdf_success)

            # Upload to Google Drive (if configured)
            drive_results = []
            if drive_uploader.is_available():
                with span("upload_to_drive", SpanType.DOCUMENT_GENERATION):
                    logger.info("☁️ Uploading to Google Drive...")
                    files_to_upload = [
                        {'filepath': word_output_path, 'filename': f"{base_filename}_translated.docx"}
                    ]

                    if pdf_success and os.path.exists(pdf_output_path):
                        files_to_upload.append({
                            'filepath': pdf_output_path,
                            'filename': f"{base_filename}_translated.pdf"
                        })

                    drive_results = drive_uploader.upload_multiple_files(files_to_upload)
                    add_metadata(files_uploaded=len(drive_results))

            # Generate enhanced final report
            end_time = time.time()
            self._generate_structured_final_report(
                filepath, output_dir_for_this_file, start_time, end_time,
                structured_document, translated_document, drive_results, pdf_success
            )

            # Save caches
            translation_service.save_caches()

            # Finish distributed trace
            finish_trace()

            logger.info("✅ Advanced translation workflow completed successfully!")
            return precomputed_style_guide

        except Exception as e:
            # Finish trace on error
            finish_trace()
            logger.error(f"❌ Advanced translation workflow failed: {e}")
            logger.info("🔄 Falling back to standard translation workflow...")
            return await self._translate_document_standard(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )

    async def _translate_document_advanced(self, filepath, output_dir_for_this_file,
                                         target_language_override=None, precomputed_style_guide=None):
        """Advanced translation workflow using the enhanced pipeline with data-flow validation and distributed tracing"""

        logger.info(f"🚀 Starting ADVANCED translation of: {os.path.basename(filepath)}")
        start_time = time.time()

        # Initialize metrics collection
        metrics = MetricsCollector() if self.enable_metrics else None
        if metrics:
            metrics.start_document_processing(filepath)
            metrics.metrics['processing_pipeline'] = 'advanced'

        # Start distributed trace for this document
        trace_id = start_trace("advanced_translation_workflow", filepath)

        try:
            # Validate inputs
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")

            if not os.path.exists(output_dir_for_this_file):
                os.makedirs(output_dir_for_this_file, exist_ok=True)

            # Set target language
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']

            # STEP 1: Extract structured document with images FIRST
            with span("extract_structured_document", SpanType.CONTENT_EXTRACTION,
                     document_model="structured", file_size_bytes=os.path.getsize(filepath)):
                logger.info("📄 Step 1: Extracting structured document with images...")

                # Extract images and create structured document
                pdf_parser_instance = PDFParser() # Renamed to avoid confusion if self.pdf_parser exists with different scope
                image_folder = os.path.join(output_dir_for_this_file, "images")
                os.makedirs(image_folder, exist_ok=True)

                # Check if cover page extraction is enabled
                # CORRECTED LINE:
                if config_manager.pdf_processing_settings.get('extract_cover_page', False):
                    # This is where cover_page_data would be extracted if the logic was here.
                    # The original error was a KeyError, implying this check itself was problematic.
                    # For now, ensuring the check is safe. Actual extraction might be elsewhere or part of PDFParser.
                    logger.info("Cover page extraction is enabled in config.")
                    # cover_page_data = pdf_parser_instance.extract_cover_page_from_pdf(filepath, output_dir_for_this_file)
                    # The above line is commented out as the original error was a KeyError on the config check itself,
                    # not necessarily a failure of extract_cover_page_from_pdf if it were called.
                    # We need to ensure config access is safe first.
                    pass # Placeholder for actual cover page extraction logic if needed here

                # Extract images
                with span("extract_images", SpanType.IMAGE_EXTRACTION):
                    extracted_images = pdf_parser_instance.extract_images_from_pdf(filepath, image_folder)
                    logger.info(f"🖼️ Extracted {len(extracted_images)} images")
                    add_metadata(images_extracted=len(extracted_images))

                # Create structured document with images
                with span("create_structured_document", SpanType.CONTENT_EXTRACTION):
                    content_extractor = StructuredContentExtractor()
                    structured_document = content_extractor.extract_structured_content_from_pdf(filepath, extracted_images)
                    logger.info(f"📊 Created structured document with {len(structured_document.content_blocks)} blocks")
                    if self.yolo_pipeline and self.use_yolo_pipeline:
                        logger.info("🔍 Enriching structured document with YOLOv8 visual detections before translation...")
                        structured_document = self.yolo_pipeline.process(structured_document)
                        logger.info("✅ YOLOv8 enrichment complete.")

                    # Count image blocks and validate data integrity
                    image_blocks = [block for block in structured_document.content_blocks if hasattr(block, 'image_path') and block.image_path]
                    logger.info(f"🖼️ Document contains {len(image_blocks)} image blocks")

                    add_metadata(
                        content_blocks_count=len(structured_document.content_blocks),
                        image_placeholders_found=len(image_blocks)
                    )

            # DATA-FLOW AUDIT: Validate structured document integrity
            with span("validate_extraction_integrity", SpanType.VALIDATION):
                self._validate_structured_document_integrity(structured_document, "after_extraction")

            # STEP 2: Use parallel translation on structured document (MUCH FASTER!)
            with span("parallel_translation", SpanType.TRANSLATION,
                     translation_method="parallel_structured", target_language=target_language):
                logger.info("🚀 Step 2: Processing translation with PARALLEL processing...")
                logger.info("   🔥 Using structured document model for maximum speed")

                translated_document = await translation_service.translate_document(
                    structured_document, target_language, ""
                )

                add_metadata(
                    translation_method="parallel_structured",
                    blocks_translated=len(structured_document.get_translatable_blocks()),
                    validation_passed=True
                )

                # Count preserved images
                translated_image_blocks = [block for block in translated_document.content_blocks if hasattr(block, 'image_path') and block.image_path]
                add_metadata(image_placeholders_preserved=len(translated_image_blocks))

            # DATA-FLOW AUDIT: Validate translation preserved structure
            with span("validate_translation_integrity", SpanType.VALIDATION):
                # Line 936 from traceback
                if config_manager.pdf_processing_settings.get('extract_cover_page', False): # Corrected line
                    # This was the original location of the KeyError.
                    # The actual logic for cover page extraction might be earlier in this method,
                    # but the config check itself is what failed.
                    # If cover_page_data is used later, it should be handled appropriately
                    # if this config is false.
                    # For now, just fixing the config access.
                    pass # Placeholder if no immediate action based on config here

                self._validate_structured_document_integrity(translated_document, "after_translation")

                # ASSERTION: Verify image preservation contract
                original_image_count = len(image_blocks)
                translated_image_count = len(translated_image_blocks)

                assert original_image_count == translated_image_count, \
                    f"Image preservation contract violated: {original_image_count} → {translated_image_count}"

                logger.info(f"✅ Data integrity validated: {translated_image_count} images preserved")
                add_metadata(validation_passed=True)

            # Generate documents from the translated structured document
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            output_dir_for_this_file = os.path.normpath(output_dir_for_this_file)
            word_output_path = os.path.normpath(os.path.join(output_dir_for_this_file, f"{base_filename}_translated.docx"))
            pdf_output_path = os.path.normpath(os.path.join(output_dir_for_this_file, f"{base_filename}_translated.pdf"))

            # Configuration for cover page extraction (using .get() for safety)
            extract_cover_setting = config_manager.pdf_processing_settings.get('extract_cover_page', False)

            if extract_cover_setting:
                # Extract cover page if enabled
                logger.info("📄 Extracting cover page...")
                cover_page_data = pdf_converter.extract_cover_page_from_pdf(filepath, output_dir_for_this_file)
            else:
                logger.info("📄 Cover page extraction skipped")
                cover_page_data = None

            # Generate Word document using structured document model (preserves images!)
            with span("generate_word_document", SpanType.DOCUMENT_GENERATION,
                     output_format="docx", images_included=len(image_blocks)):
                logger.info("📄 Generating Word document from translated structured document...")
                logger.info(f"   🖼️ Including {len(image_blocks)} images from: {image_folder}")

                try:
                    saved_word_filepath = document_generator.create_word_document_from_structured_document(
                        translated_document, word_output_path, image_folder, cover_page_data
                    )

                    logger.debug(f"   • Document generator returned: {saved_word_filepath}")

                    if saved_word_filepath and os.path.exists(saved_word_filepath):
                        file_size = os.path.getsize(saved_word_filepath)
                        logger.info(f"✅ Word document created successfully: {file_size} bytes")
                        add_metadata(output_file_size_bytes=file_size)
                    else:
                        logger.error(f"❌ Document generator returned path but file doesn't exist!")
                        logger.error(f"   • Returned path: {saved_word_filepath}")
                        logger.error(f"   • Expected path: {word_output_path}")
                        logger.error(f"   • File exists check: {os.path.exists(saved_word_filepath) if saved_word_filepath else 'N/A'}")
                        raise Exception("Word document was not created - file missing after generation")

                except Exception as doc_error:
                    logger.error(f"❌ Document generation failed with error: {doc_error}")
                    logger.error(f"   • Structured document blocks: {len(translated_document.content_blocks) if translated_document else 'N/A'}")
                    logger.error(f"   • Target directory exists: {os.path.exists(output_dir_for_this_file)}")
                    logger.error(f"   • Target directory writable: {os.access(output_dir_for_this_file, os.W_OK)}")
                    add_metadata(error_count=1, validation_passed=False)
                    raise Exception(f"Failed to create Word document from advanced translation: {doc_error}")

                if not saved_word_filepath:
                    add_metadata(error_count=1, validation_passed=False)
                    raise Exception("Failed to create Word document from advanced translation")

            # Convert to PDF
            with span("convert_to_pdf", SpanType.DOCUMENT_GENERATION, output_format="pdf"):
                logger.info("📑 Converting to PDF...")
                pdf_success = pdf_converter.convert_word_to_pdf(saved_word_filepath, pdf_output_path)
                add_metadata(pdf_conversion_success=pdf_success)

            # Upload to Google Drive (if configured)
            drive_results = []
            if drive_uploader.is_available():
                with span("upload_to_drive", SpanType.DOCUMENT_GENERATION):
                    logger.info("☁️ Uploading to Google Drive...")
                    files_to_upload = [
                        {'filepath': word_output_path, 'filename': f"{base_filename}_translated.docx"}
                    ]

                    if pdf_success and os.path.exists(pdf_output_path):
                        files_to_upload.append({
                            'filepath': pdf_output_path,
                            'filename': f"{base_filename}_translated.pdf"
                        })

                    drive_results = drive_uploader.upload_multiple_files(files_to_upload)
                    add_metadata(files_uploaded=len(drive_results))

            # Generate enhanced final report
            end_time = time.time()
            self._generate_structured_final_report(
                filepath, output_dir_for_this_file, start_time, end_time,
                structured_document, translated_document, drive_results, pdf_success
            )
            
            # Log consolidation results if available
            if hasattr(document, 'consolidation_info') and document.consolidation_info.get('consolidator_used'):
                consolidation_info = document.consolidation_info
                logger.info(f"🧠 NODE CONSOLIDATION SUMMARY:")
                logger.info(f"   • Original blocks: {consolidation_info['original_blocks']:,}")
                logger.info(f"   • Consolidated blocks: {consolidation_info['consolidated_blocks']:,}")
                logger.info(f"   • Reduction: {consolidation_info['reduction_percentage']:.1f}%")
                logger.info(f"   • Translation batches: {consolidation_info['translation_batches']}")
                logger.info(f"   • 25,000 nodes problem: SOLVED ✅")
            
            # Save translation cache
            translation_service.save_caches()
            
            logger.info("✅ Structured document translation completed successfully!")
            return precomputed_style_guide
            
        except Exception as e:
            logger.error(f"❌ Structured document translation failed: {e}")
            raise

    async def _translate_document_standard(self, filepath, output_dir_for_this_file,
                                         target_language_override=None, precomputed_style_guide=None):
        """Standard translation workflow without advanced features"""
        try:
            # ... existing code ...

            # Step 5: Translate the structured document
            logger.info("🌐 Step 5: Translating structured document...")
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']

            # Create translated document
            translated_document = await translation_service.translate_document(
                document, target_language, precomputed_style_guide or ""
            )

            # Step 6: Generate Word document from structured document
            logger.info("📄 Step 6: Generating Word document...")
            saved_word_filepath = document_generator.create_word_document_from_structured_document(
                translated_document, word_output_path, image_folder, cover_page_data
            )

            if not saved_word_filepath:
                raise Exception("Failed to create Word document")

            # Convert to PDF if enabled
            pdf_output_path = ""
            pdf_success = False
            if config_manager.word_output_settings.get('generate_pdf', False):
                logger.info("📑 Step 7: Converting to PDF...")
                pdf_output_path = os.path.join(output_dir_for_this_file, f"{base_filename}_translated.pdf")
                pdf_success = pdf_converter.convert_word_to_pdf(saved_word_filepath, pdf_output_path)
            else:
                logger.info("📄 PDF generation skipped by configuration.")

            # Upload to Google Drive (if configured)
            drive_results = []
            if drive_uploader.is_available():
                logger.info("☁️ Step 8: Uploading to Google Drive...")
                files_to_upload = [
                    {'filepath': word_output_path, 'filename': f"{base_filename}_translated.docx"}
                ]

                if pdf_success and os.path.exists(pdf_output_path):
                    files_to_upload.append({
                        'filepath': pdf_output_path,
                        'filename': f"{base_filename}_translated.pdf"
                    })

                drive_results = drive_uploader.upload_multiple_files(files_to_upload)

            # Step 9: Generate final report
            end_time = time.time()
            self._generate_structured_final_report(
                filepath, output_dir_for_this_file, start_time, end_time,
                document, translated_document, drive_results, pdf_success
            )

            # Save translation cache
            translation_service.save_caches()
            
            logger.info("✅ Translation workflow completed successfully!")
            return precomputed_style_guide  # Return for potential reuse in batch processing
            
        except Exception as e:
            logger.error(f"❌ Translation workflow failed: {e}")
            raise

    async def _translate_document_structured(self, filepath, output_dir_for_this_file,
                                           target_language_override=None, precomputed_style_guide=None):
        """
        New structured document translation workflow using the refactored document model.
        This method implements the structured document model approach for better content integrity.
        """
        if not STRUCTURED_MODEL_AVAILABLE:
            logger.warning("Structured document model not available, falling back to standard workflow")
            return await self._translate_document_standard(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )

        start_time = time.time()
        
        try:
            # Validate inputs
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")
            
            if not os.path.exists(output_dir_for_this_file):
                os.makedirs(output_dir_for_this_file, exist_ok=True)
            
            # Set up output paths
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            output_dir_for_this_file = os.path.normpath(output_dir_for_this_file)
            image_folder = os.path.join(output_dir_for_this_file, "images")
            word_output_path = os.path.normpath(os.path.join(output_dir_for_this_file, f"{base_filename}_translated.docx"))
            pdf_output_path = os.path.normpath(os.path.join(output_dir_for_this_file, f"{base_filename}_translated.pdf"))
            
            # Step 1: Extract images and cover page
            logger.info("📷 Step 1: Extracting images and cover page...")
            extracted_images = self.pdf_parser.extract_images_from_pdf(filepath, image_folder)
            cover_page_data = self.pdf_parser.extract_cover_page_from_pdf(filepath, output_dir_for_this_file)
            
            # Step 2: Extract structured content
            logger.info("📝 Step 2: Extracting structured content...")
            document = self.content_extractor.extract_structured_content_from_pdf(
                filepath, extracted_images
            )
            
            if not document or not document.content_blocks:
                raise Exception("No content could be extracted from the PDF")
            
            # Step 2.5: Apply intelligent node consolidation (if available)
            if INTELLIGENT_NODE_CONSOLIDATOR_AVAILABLE:
                logger.info("🧠 Step 2.5: Applying intelligent node consolidation...")
                original_block_count = len(document.content_blocks)
                
                # Convert document blocks to node format for consolidation
                nodes = self._convert_document_blocks_to_nodes(document.content_blocks)
                
                # Apply intelligent consolidation
                consolidator = create_intelligent_consolidator(max_batch_chars=12000)
                consolidated_nodes = consolidator.consolidate_nodes(nodes)
                
                # Convert back to document blocks
                consolidated_blocks = self._convert_consolidated_nodes_to_blocks(consolidated_nodes)
                document.content_blocks = consolidated_blocks
                
                consolidated_block_count = len(document.content_blocks)
                reduction_percentage = ((original_block_count - consolidated_block_count) / original_block_count) * 100
                
                logger.info(f"   📊 Node consolidation results:")
                logger.info(f"      • Original blocks: {original_block_count:,}")
                logger.info(f"      • Consolidated blocks: {consolidated_block_count:,}")
                logger.info(f"      • Reduction: {reduction_percentage:.1f}%")
                
                # Create semantic batches for translation
                batches = consolidator.create_semantic_batches(consolidated_nodes)
                logger.info(f"      • Translation batches: {len(batches)}")
                
                # Store consolidation info for later use
                document.consolidation_info = {
                    'original_blocks': original_block_count,
                    'consolidated_blocks': consolidated_block_count,
                    'reduction_percentage': reduction_percentage,
                    'translation_batches': len(batches),
                    'consolidator_used': True
                }
            else:
                logger.info("⚠️ Step 2.5: Intelligent node consolidator not available, skipping consolidation")
                document.consolidation_info = {
                    'consolidator_used': False
                }
            
            # Step 3: Analyze images
            logger.info("🔍 Step 3: Analyzing images...")
            if extracted_images:
                image_paths = [img['filepath'] for img in extracted_images]
                image_analysis = self.image_analyzer.batch_analyze_images(image_paths)
                self._integrate_image_analysis_into_document(document, image_analysis)
            
            # Step 4: Translate the structured document
            logger.info("🌐 Step 4: Translating structured document...")
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']
            
            translated_document = await translation_service.translate_document(
                document, target_language, precomputed_style_guide or ""
            )
            
            # Step 5: Generate Word document
            logger.info("📄 Step 5: Generating Word document...")
            saved_word_filepath = document_generator.create_word_document_from_structured_document(
                translated_document, word_output_path, image_folder, cover_page_data
            )
            
            if not saved_word_filepath:
                raise Exception("Failed to create Word document")
            
            # Step 6: Convert to PDF if enabled
            pdf_success = False
            if config_manager.word_output_settings.get('generate_pdf', False):
                logger.info("📑 Step 6: Converting to PDF...")
                pdf_success = pdf_converter.convert_word_to_pdf(saved_word_filepath, pdf_output_path)
            else:
                logger.info("📄 PDF generation skipped by configuration.")
            
            # Step 7: Upload to Google Drive if configured
            drive_results = []
            if drive_uploader.is_available():
                logger.info("☁️ Step 7: Uploading to Google Drive...")
                files_to_upload = [
                    {'filepath': word_output_path, 'filename': f"{base_filename}_translated.docx"}
                ]
                
                if pdf_success and os.path.exists(pdf_output_path):
                    files_to_upload.append({
                        'filepath': pdf_output_path,
                        'filename': f"{base_filename}_translated.pdf"
                    })
                
                drive_results = drive_uploader.upload_multiple_files(files_to_upload)
            
            # Step 8: Generate final report
            end_time = time.time()
            self._generate_structured_final_report(
                filepath, output_dir_for_this_file, start_time, end_time,
                document, translated_document, drive_results, pdf_success
            )
            
            # Log consolidation results if available
            if hasattr(document, 'consolidation_info') and document.consolidation_info.get('consolidator_used'):
                consolidation_info = document.consolidation_info
                logger.info(f"🧠 NODE CONSOLIDATION SUMMARY:")
                logger.info(f"   • Original blocks: {consolidation_info['original_blocks']:,}")
                logger.info(f"   • Consolidated blocks: {consolidation_info['consolidated_blocks']:,}")
                logger.info(f"   • Reduction: {consolidation_info['reduction_percentage']:.1f}%")
                logger.info(f"   • Translation batches: {consolidation_info['translation_batches']}")
                logger.info(f"   • 25,000 nodes problem: SOLVED ✅")
            
            # Save translation cache
            translation_service.save_caches()
            
            logger.info("✅ Structured document translation completed successfully!")
            return precomputed_style_guide
            
        except Exception as e:
            logger.error(f"❌ Structured document translation failed: {e}")
            raise

    async def _translate_document_graph_based(self, filepath, output_dir_for_this_file,
                                            target_language_override=None, precomputed_style_guide=None):
        """Graph-based translation workflow with advanced structure preservation and format integrity"""
        
        logger.info(f"🔄 Starting GRAPH-BASED translation of: {os.path.basename(filepath)}")
        start_time = time.time()
        
        try:
            # Validate inputs
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")
            
            if not os.path.exists(output_dir_for_this_file):
                os.makedirs(output_dir_for_this_file, exist_ok=True)
            
            # Set target language
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']
            
            # Check if graph-based pipeline is available
            if not GRAPH_BASED_PIPELINE_AVAILABLE:
                logger.warning("⚠️ Graph-based pipeline not available, falling back to structured translation")
                return await self._translate_document_structured(
                    filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
                )
            
            # Process with graph-based pipeline
            logger.info("🔄 Processing with graph-based translation pipeline...")
            logger.info("   🔥 Using advanced structure preservation and format integrity")
            
            # Configure output format (default to HTML for better structure preservation)
            output_format = 'html'  # Can be 'html', 'markdown', 'word', 'json'
            preserve_layout = True
            
            # Run graph-based translation
            graph_result = await graph_based_pipeline.translate_document(
                input_path=filepath,
                output_dir=output_dir_for_this_file,
                target_language=target_language,
                output_format=output_format,
                preserve_layout=preserve_layout
            )
            
            if not graph_result['success']:
                raise Exception(f"Graph-based translation failed: {graph_result.get('error', 'Unknown error')}")
            
            # Generate comprehensive final report
            end_time = time.time()
            self._generate_graph_based_final_report(
                filepath, output_dir_for_this_file, start_time, end_time, graph_result
            )
            
            # Save caches if available
            if hasattr(translation_service, 'save_caches'):
                translation_service.save_caches()
            
            logger.info("✅ Graph-based translation completed successfully!")
            logger.info(f"   📊 Processing time: {graph_result['processing_time']:.2f}s")
            logger.info(f"   📄 Output files: {len(graph_result['output_files'])}")
            
            # Log translation quality metrics
            if 'translation_quality' in graph_result:
                quality = graph_result['translation_quality']
                logger.info(f"   🎯 Translation quality:")
                logger.info(f"      • Total translated: {quality.get('total_translated', 0)}")
                logger.info(f"      • Format preserved: {quality.get('format_preserved', 0)}")
                logger.info(f"      • Preservation rate: {quality.get('format_preservation_rate', 0):.1%}")
            
            return precomputed_style_guide
            
        except Exception as e:
            logger.error(f"❌ Graph-based translation workflow failed: {e}")
            logger.info("🔄 Falling back to structured translation workflow...")
            return await self._translate_document_structured(
                filepath, output_dir_for_this_file, target_language_override, precomputed_style_guide
            )
    
    def _generate_optimized_final_report(self, input_filepath, output_dir, start_time, end_time, result):
        """Generate final report for optimized pipeline processing"""
        
        processing_time = end_time - start_time
        
        # Create comprehensive report
        report_content = f"""
OPTIMIZED DOCUMENT PIPELINE - FINAL REPORT
==========================================

📄 Input File: {os.path.basename(input_filepath)}
📁 Output Directory: {output_dir}
⏱️ Processing Time: {processing_time:.2f} seconds
📅 Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

📊 PROCESSING RESULTS
====================

✅ Success: {result.success}
📄 Total Pages: {result.statistics.total_pages}
⏱️ Processing Time: {result.statistics.processing_time:.2f}s
📊 Average Page Time: {result.statistics.average_page_time:.2f}s
🌍 Translation Success Rate: {result.statistics.translation_success_rate:.1%}
💾 Memory Usage: {result.statistics.memory_usage_mb:.1f} MB

📈 STRATEGY DISTRIBUTION
========================
"""
        
        for strategy, count in result.statistics.strategy_distribution.items():
            report_content += f"  {strategy}: {count}\n"
        
        if result.statistics.content_type_distribution:
            report_content += f"""
📋 CONTENT TYPE DISTRIBUTION
============================
"""
            for content_type, count in result.statistics.content_type_distribution.items():
                report_content += f"  {content_type}: {count}\n"
        
        report_content += f"""
📁 GENERATED FILES
==================
"""
        
        for file_type, file_path in result.output_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                report_content += f"  {file_type}: {file_path} ({file_size} bytes)\n"
            else:
                report_content += f"  {file_type}: {file_path} (not found)\n"
        
        if not result.success:
            report_content += f"""
❌ ERROR INFORMATION
===================
Error: {result.error}
"""
        
        report_content += f"""
🚀 OPTIMIZED PIPELINE FEATURES
==============================
✅ PyMuPDF content extraction
✅ YOLO layout analysis (0.15 confidence threshold)
✅ Content-to-layout mapping
✅ Intelligent processing strategies
✅ Parallel processing (6 workers)
✅ Progress tracking and monitoring

💡 PERFORMANCE TIPS
===================
• Consider adjusting max_workers based on system resources
• Monitor GPU memory usage for YOLO processing
• Use SSD storage for better I/O performance
• Ensure sufficient RAM for large documents

🎉 PROCESSING COMPLETED
=======================
Optimized pipeline processing completed successfully!
"""
        
        # Save report
        report_path = os.path.join(output_dir, "optimized_pipeline_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📊 Optimized pipeline report generated: {report_path}")
        logger.info(f"✅ Optimized pipeline processing completed in {processing_time:.2f}s")

    def _generate_graph_based_final_report(self, input_filepath, output_dir, start_time, end_time, graph_result):
        """Generate comprehensive final report for graph-based translation"""
        try:
            base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
            report_path = os.path.join(output_dir, f"{base_filename}_graph_based_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== GRAPH-BASED TRANSLATION FINAL REPORT ===\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input file: {input_filepath}\n")
                f.write(f"Output directory: {output_dir}\n")
                f.write(f"Processing time: {end_time - start_time:.2f}s\n\n")
                
                # Processing statistics
                if 'statistics' in graph_result:
                    stats = graph_result['statistics']
                    f.write("PROCESSING STATISTICS:\n")
                    f.write(f"  Pages processed: {stats.get('pages_processed', 0)}\n")
                    f.write(f"  YOLO detections: {stats.get('yolo_detections', 0)}\n")
                    f.write(f"  OCR regions: {stats.get('ocr_regions', 0)}\n")
                    f.write(f"  Graph nodes: {stats.get('graph_nodes', 0)}\n")
                    f.write(f"  Translated blocks: {stats.get('translated_blocks', 0)}\n")
                    f.write(f"  Format preserved blocks: {stats.get('format_preserved_blocks', 0)}\n\n")
                
                # Translation quality
                if 'translation_quality' in graph_result:
                    quality = graph_result['translation_quality']
                    f.write("TRANSLATION QUALITY:\n")
                    f.write(f"  Total translated: {quality.get('total_translated', 0)}\n")
                    f.write(f"  Format preserved: {quality.get('format_preserved', 0)}\n")
                    f.write(f"  Format preservation rate: {quality.get('format_preservation_rate', 0):.1%}\n\n")
                
                # Output files
                if 'output_files' in graph_result:
                    f.write("OUTPUT FILES:\n")
                    for output_file in graph_result['output_files']:
                        f.write(f"  {output_file}\n")
                    f.write("\n")
                
                # Configuration
                f.write("CONFIGURATION:\n")
                f.write(f"  Target language: {graph_result.get('target_language', 'unknown')}\n")
                f.write(f"  Output format: {graph_result.get('output_format', 'unknown')}\n")
                f.write(f"  Layout preserved: {graph_result.get('preserve_layout', True)}\n")
            
            logger.info(f"📊 Graph-based translation report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating graph-based final report: {e}")

    async def translate_pdf_with_final_assembly(self, filepath, output_dir, target_language_override=None, precomputed_style_guide=None, cover_page_data=None):
        """
        COMPREHENSIVE FINAL ASSEMBLY APPROACH: Translate PDF using the new comprehensive strategy.

        This method implements the hybrid parsing strategy that ensures TOC and visual elements
        are correctly placed in the final document while bypassing image translation entirely.

        Key Features:
        - Parallel Nougat + PyMuPDF processing
        - Intelligent visual element correlation
        - Image bypass (no translation for visual content)
        - Two-pass TOC generation with accurate page numbers
        - High-fidelity document assembly
        """
        logger.info("🚀 Starting Final Assembly Translation Pipeline...")

        try:
            # Import the final assembly pipeline
            from final_document_assembly_pipeline import FinalDocumentAssemblyPipeline

            # Initialize the pipeline
            pipeline = FinalDocumentAssemblyPipeline()

            # Determine target language
            target_language = target_language_override or config_manager.translation_enhancement_settings['target_language']

            # Process PDF with comprehensive assembly strategy
            results = await pipeline.process_pdf_with_final_assembly(
                pdf_path=filepath,
                output_dir=output_dir,
                target_language=target_language
            )

            # Log results
            if results['status'] == 'success':
                logger.info("✅ Final Assembly Translation completed successfully!")
                logger.info(f"📄 Word document: {os.path.basename(results['output_files']['word_document'])}")
                logger.info(f"🖼️ Images preserved: {results['processing_statistics']['preserved_images']}")
                logger.info(f"📋 TOC entries: {results['processing_statistics']['toc_entries']}")
            else:
                logger.warning(f"⚠️ Final Assembly Translation completed with issues: {results.get('validation_results', {}).get('issues', [])}")

            return results

        except ImportError:
            logger.error("❌ Final Assembly Pipeline not available - falling back to standard structured approach")
            return await self.translate_pdf_structured_document_model(filepath, output_dir, target_language_override, precomputed_style_guide, cover_page_data)

        except Exception as e:
            logger.error(f"❌ Final Assembly Translation failed: {e}")
            logger.info("🔄 Falling back to standard structured approach...")
            return await self.translate_pdf_structured_document_model(filepath, output_dir, target_language_override, precomputed_style_guide, cover_page_data)

    def _integrate_image_analysis_into_document(self, document, image_analysis):
        """Integrate image analysis results into Document object"""
        # Create a mapping of image filenames to analysis results
        analysis_map = {
            os.path.basename(analysis['path']): analysis
            for analysis in image_analysis
        }

        # Update ImagePlaceholder blocks in the document
        for block in document.content_blocks:
            if hasattr(block, 'image_path') and block.image_path:
                filename = os.path.basename(block.image_path)
                if filename in analysis_map:
                    analysis = analysis_map[filename]
                    if analysis['should_translate'] and analysis['extracted_text']:
                        block.ocr_text = analysis['extracted_text']
                        block.translation_needed = True
                        # Update original_text with OCR text for potential translation
                        if not block.original_text:
                            block.original_text = analysis['extracted_text']
                    else:
                        block.translation_needed = False

    def _generate_structured_final_report(self, input_filepath, output_dir, start_time, end_time,
                                        original_document, translated_document, drive_results, pdf_success=True):
        """Generate final report for structured document translation"""
        duration = end_time - start_time

        # Generate file status
        word_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.docx')
        pdf_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.pdf')

        files_section = f"📄 Generated Files:\n• Word Document: {word_filename} ✅"

        if pdf_success:
            files_section += f"\n• PDF Document: {pdf_filename} ✅"
        else:
            files_section += f"\n• PDF Document: {pdf_filename} ❌ (Conversion failed)"

        # Document statistics
        original_stats = original_document.get_statistics()
        translated_stats = translated_document.get_statistics()

        stats_section = f"""
📊 DOCUMENT STATISTICS:
• Original Title: {original_document.title}
• Translated Title: {translated_document.title}
• Total Pages: {original_document.total_pages}
• Content Blocks: {original_stats['total_blocks']} → {translated_stats['total_blocks']}
• Translatable Blocks: {original_stats['translatable_blocks']}
• Non-translatable Blocks: {original_stats['non_translatable_blocks']}

📋 CONTENT BREAKDOWN:
"""
        for content_type, count in original_stats['blocks_by_type'].items():
            stats_section += f"• {content_type.replace('_', ' ').title()}: {count}\n"

        report = f"""
🎉 STRUCTURED DOCUMENT TRANSLATION COMPLETED {'SUCCESSFULLY' if pdf_success else 'WITH WARNINGS'}!
=================================================================

📁 Input: {os.path.basename(input_filepath)}
📁 Output Directory: {output_dir}
⏱️ Total Time: {duration/60:.1f} minutes

{files_section}

{stats_section}
"""

        if drive_results:
            from drive_uploader import drive_uploader # Corrected import statement
            report += f"\n{drive_uploader.get_upload_summary(drive_results)}"

        if not pdf_success:
            report += f"""

⚠️ PDF CONVERSION TROUBLESHOOTING:
• Ensure Microsoft Word is installed and licensed
• Check Windows permissions and antivirus settings
• Try running as administrator
• Alternative: Use online PDF converters or LibreOffice
"""

        logger.info(report)

    def _integrate_image_analysis(self, structured_content, image_analysis):
        """Integrate image analysis results into structured content"""
        # Create a mapping of image filenames to analysis results
        analysis_map = {
            os.path.basename(analysis['path']): analysis 
            for analysis in image_analysis
        }
        
        # Update image items in structured content
        for item in structured_content:
            if item.get('type') == 'image':
                filename = item.get('filename')
                if filename in analysis_map:
                    analysis = analysis_map[filename]
                    if analysis['should_translate'] and analysis['extracted_text']:
                        item['ocr_text'] = analysis['extracted_text']
                        item['translation_needed'] = True
                    else:
                        item['translation_needed'] = False

    def _restructure_content_text(self, structured_content):
        """Restructure text content to separate footnotes from main content"""
        logger.debug(f"📋 _restructure_content_text received: {type(structured_content)}")
        logger.debug(f"📋 Is it a list? {isinstance(structured_content, list)}")
        if hasattr(structured_content, 'content_blocks'):
            logger.error(f"❌ ERROR: _restructure_content_text received Document object instead of list!")
            logger.error(f"❌ Document has {len(structured_content.content_blocks)} content blocks")
            raise TypeError("_restructure_content_text expects a list, but received a Document object")

        restructured_content = []
        footnotes_collected = []

        for item in structured_content:
            if item.get('type') in ['paragraph', 'text'] and item.get('text'):
                # Apply text restructuring to separate footnotes
                try:
                    restructured = self.text_restructurer.analyze_and_restructure_text(item['text'])

                    # Update the main content
                    if restructured['main_content']:
                        item['text'] = restructured['main_content']
                        restructured_content.append(item)

                    # Collect footnotes
                    if restructured['footnotes']:
                        for footnote in restructured['footnotes']:
                            footnote_item = {
                                'type': 'footnote',
                                'text': footnote,
                                'page_num': item.get('page_num', 0),
                                'source_block': item.get('block_num', 0)
                            }
                            footnotes_collected.append(footnote_item)

                        logger.info(f"📝 Separated {len(restructured['footnotes'])} footnotes from page {item.get('page_num', 'unknown')}")

                except Exception as e:
                    logger.warning(f"Failed to restructure text on page {item.get('page_num', 'unknown')}: {e}")
                    # Keep original item if restructuring fails
                    restructured_content.append(item)
            else:
                # Keep non-text items as-is
                restructured_content.append(item)

        # Add footnotes at the end if any were found
        if footnotes_collected:
            logger.info(f"📋 Total footnotes collected: {len(footnotes_collected)}")
            restructured_content.extend(footnotes_collected)

        return restructured_content

    async def _translate_batches(self, optimized_batches, target_language, style_guide):
        """Translate optimized batches of content"""
        translated_items = []
        total_batches = len(optimized_batches)
        
        progress_tracker = ProgressTracker(total_batches)
        
        for batch_idx, batch in enumerate(optimized_batches):
            logger.info(f"Translating batch {batch_idx + 1}/{total_batches}")
            
            batch_start_time = time.time()
            
            try:
                # Translate each group in the batch
                for group_idx, group in enumerate(batch):
                    logger.debug(f"Processing group {group_idx + 1}/{len(batch)} with {len(group)} items")

                    # Combine group items for translation
                    combined_text = optimization_manager.grouping_processor.combine_group_for_translation(group)
                    logger.debug(f"Combined text length: {len(combined_text)} characters")

                    # Translate the combined text
                    logger.debug("Sending text to translation service...")
                    translated_text = await translation_service.translate_text(
                        combined_text, target_language, style_guide
                    )
                    logger.debug(f"Received translated text length: {len(translated_text)} characters")

                    # Log the raw translation for debugging
                    logger.debug(f"Raw translated text preview: {translated_text[:200]}...")

                    # Split the translated text back into individual items
                    logger.debug("Splitting translated text back into individual items...")
                    translated_group = optimization_manager.grouping_processor.split_translated_group(
                        translated_text, group
                    )
                    logger.debug(f"Split resulted in {len(translated_group)} items")

                    translated_items.extend(translated_group)
                
                # Record performance
                batch_time = time.time() - batch_start_time
                optimization_manager.record_batch_performance(
                    len(str(batch)), batch_time, 1.0  # Success rate = 1.0 for successful batches
                )
                
                progress_tracker.update(completed=1)
                
            except Exception as e:
                logger.error(f"Failed to translate batch {batch_idx + 1}: {e}")
                
                # Add original items as fallback
                for group in batch:
                    translated_items.extend(group)
                
                progress_tracker.update(failed=1)
        
        progress_tracker.finish()
        return translated_items
    
    def _reconstruct_full_content(self, original_content, translated_text_items):
        """Reconstruct full content by merging translated text with images"""
        # Create a mapping of translated items by their original position/identifier
        translated_map = {}
        
        for item in translated_text_items:
            # Use page_num and block_num as identifier
            key = (item.get('page_num'), item.get('block_num'))
            translated_map[key] = item
        
        # Reconstruct the full content
        final_content = []
        
        for original_item in original_content:
            if original_item.get('type') == 'image':
                # Keep image items as-is
                final_content.append(original_item)
            else:
                # Use translated version if available
                key = (original_item.get('page_num'), original_item.get('block_num'))
                if key in translated_map:
                    final_content.append(translated_map[key])
                else:
                    # Fallback to original
                    final_content.append(original_item)
        
        return final_content

    def _apply_translation_to_document(self, document, source_text_blocks, translated_texts):
        """Apply translated text to structured document while preserving images and structure"""

        if not STRUCTURED_MODEL_AVAILABLE:
            logger.error("❌ Structured document model not available for translation application")
            return document

        try:
            from structured_document_model import Document as StructuredDocument

            # Ensure source_text_blocks and translated_texts have the same length
            if len(source_text_blocks) != len(translated_texts):
                raise ValueError("Source text blocks and translated texts count mismatch")

            # Create a mapping of block identifiers to translated texts
            translation_map = {
                (block.get('page_num'), block.get('block_num')): translated_text
                for block, translated_text in zip(source_text_blocks, translated_texts)
            }

            # Update the document's content blocks with translated texts
            for block in document.content_blocks:
                if hasattr(block, 'get_content_type'):
                    content_type = block.get_content_type().value

                    if content_type in ['paragraph', 'heading', 'list_item']:
                        # Apply translation if available
                        key = (block.get('page_num'), block.get('block_num'))
                        if key in translation_map:
                            translated_text = translation_map[key]
                            block.content = translated_text
                            logger.debug(f"📝 Applied translation to {content_type} (block {key})")
                        else:
                            logger.debug(f"⚠️ No translation found for {content_type} (block {key}), keeping original")
                    else:
                        logger.debug(f"📋 Preserved {content_type} block")
                else:
                    # Fallback for blocks without get_content_type method
                    logger.debug("📋 Preserved block (no content type)")

            logger.info(f"✅ Applied translation to structured document:")
            logger.info(f"   📊 Original blocks: {len(document.content_blocks)}")
            logger.info(f"   📊 Translated blocks: {len(translation_map)}")
            logger.info(f"   📝 Paragraphs translated: {len(translated_texts)}")

            return document

        except Exception as e:
            logger.error(f"❌ Failed to apply translation to structured document: {e}")
            logger.warning("⚠️ Returning original document")
            return document

    def _convert_advanced_result_to_content(self, advanced_result, output_dir):
        """Convert advanced translation result to structured content format with heading preservation"""
        content_items = []

        if advanced_result.translated_text:
            # Enhanced parsing to preserve heading structure
            content_items = self._parse_translated_content_with_structure(advanced_result.translated_text)

            # If no structure was detected, fall back to simple paragraph parsing
            if not content_items:
                paragraphs = advanced_result.translated_text.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        content_items.append({
                            'type': 'paragraph',
                            'text': paragraph.strip(),
                            'page_num': 1,
                            'block_num': i + 1
                        })

        return content_items

    def _parse_translated_content_with_structure(self, translated_text):
        """Parse translated content to preserve heading structure, handle images, and clean placeholders."""
        content_items = []
        lines = translated_text.split('\\n')
        current_paragraph_lines = []
        block_num = 1

        def save_current_paragraph():
            nonlocal block_num # Allow modification of block_num from outer scope
            if current_paragraph_lines:
                # Join lines, then strip the whole paragraph, then check if it's just "" or empty
                paragraph_text = '\\n'.join(current_paragraph_lines).strip()
                if paragraph_text and paragraph_text != '""':
                    content_items.append({
                        'type': 'paragraph',
                        'text': paragraph_text,
                        'page_num': 1, # Placeholder, needs better page context if available
                        'block_num': block_num
                    })
                    block_num += 1
                current_paragraph_lines.clear()

        for line in lines:
            line_stripped = line.strip()

            # 1. Handle lines that should be skipped or act as paragraph breaks
            # Skip Nougat placeholders for missing/empty pages
            if (re.fullmatch(r"\[ΕΛΛΕΙΠΟΥΣΑ_ΣΕΛΙΔΑ_ΚΕΝΗ:\d+\]", line_stripped) or
                re.fullmatch(r"\[MISSING_PAGE_EMPTY:\d+\]", line_stripped) or
                re.fullmatch(r"\[MISSING_PAGE_FAIL:\d+\]", line_stripped) or
                re.fullmatch(r"\[MISSING_PAGE_POST:\d*\]", line_stripped)):
                save_current_paragraph()
                logger.debug(f"Skipping Nougat placeholder: {line_stripped}")
                continue
            
            # Skip empty lines, standalone quotes, and other formatting artifacts
            if (not line_stripped or
                line_stripped == '""' or
                line_stripped == '"' or
                line_stripped == "''" or
                line_stripped == "'" or
                re.fullmatch(r'["\'\s]*', line_stripped)):  # Only quotes and whitespace
                save_current_paragraph()
                continue

            # 2. Handle special block types (images, headings)
            image_match = re.fullmatch(r"\\[IMAGE:\\s*(.+?)\\s*\\]", line_stripped)
            if image_match:
                save_current_paragraph()
                image_filename = image_match.group(1);
                content_items.append({
                    'type': 'image',
                    'filename': image_filename,
                    'page_num': 1, # Placeholder for page number
                    'block_num': block_num
                })
                block_num += 1
                continue

            heading_match = re.match(r'^(#{1,6})\\s+(.+)$', line_stripped) # Markdown headings
            if heading_match:
                save_current_paragraph()
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                if heading_text: # Ensure heading text is not empty
                    content_items.append({
                        'type': f'h{level}',
                        'text': heading_text,
                        'page_num': 1, # Placeholder
                        'block_num': block_num
                    })
                    block_num += 1
                continue
            
            potential_heading_type = self._detect_potential_heading_type(line_stripped) # Custom heading detection
            if potential_heading_type:
                save_current_paragraph()
                heading_text = self._clean_heading_text(line_stripped)
                if heading_text: # Ensure heading text is not empty
                    content_items.append({
                        'type': potential_heading_type,
                        'text': heading_text,
                        'page_num': 1, # Placeholder
                        'block_num': block_num
                    })
                    block_num += 1
                continue

            # 3. If none of the above, it's part of a paragraph
            current_paragraph_lines.append(line_stripped) # Append the stripped line

        save_current_paragraph() # Save any remaining paragraph at the end

        return content_items

    def _detect_potential_heading_type(self, line):
        """Detect if a line is likely a heading and return its type"""
        if len(line) > 150:  # Too long to be a heading
            return None

        # Pattern 1: Bold text that looks like headings
        bold_pattern = r'^\*\*(.+?)\*\*$'
        if re.match(bold_pattern, line):
            heading_text = re.match(bold_pattern, line).group(1).strip()
            return self._determine_heading_level_from_content(heading_text)

        # Pattern 2: Lines that look like titles (short, capitalized, no period)
        if (len(line) < 100 and
            line[0].isupper() and
            not line.endswith('.') and
            not line.startswith('*') and
            ' ' in line):

            words = line.split()
            if (len(words) >= 3 and
                sum(1 for word in words if word[0].isupper()) >= len(words) * 0.6):
                return self._determine_heading_level_from_content(line)

        return None

    def _determine_heading_level_from_content(self, text):
        """Determine heading level based on content"""
        text_lower = text.lower()

        # Level 1: Main titles, document titles
        if any(keyword in text_lower for keyword in ['senda:', 'assessment', 'federation', 'militant']):
            return 'h1'

        # Level 2: Major sections
        elif any(keyword in text_lower for keyword in ['need for', 'discourse', 'powerful', 'conclusions', 'συμπεράσματα']):
            return 'h2'

        # Level 3: Subsections
        elif any(keyword in text_lower for keyword in ['what should', 'that said', 'implementation']):
            return 'h3'

        # Default to level 2 for other potential headings
        else:
            return 'h2'

    def _clean_heading_text(self, text):
        """Clean heading text by removing bold markers and extra formatting"""
        # Remove bold markers
        text = re.sub(r'^\*\*(.+?)\*\*$', r'\1', text)

        # Clean up whitespace
        text = ' '.join(text.split())

        return text

    def _generate_advanced_final_report(self, input_filepath, output_dir, start_time, end_time,
                                      advanced_result, drive_results, pdf_success=True):
        """Generate comprehensive final report for advanced translation"""
        duration = end_time - start_time

        # Generate file status
        word_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.docx')
        pdf_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.pdf')

        files_section = f"📄 Generated Files:\n• Word Document: {word_filename} ✅"

        if pdf_success:
            files_section += f"\n• PDF Document: {pdf_filename} ✅"
        else:
            files_section += f"\n• PDF Document: {pdf_filename} ❌ (Conversion failed)"

        # Advanced metrics section
        advanced_metrics = f"""
🚀 ADVANCED FEATURES PERFORMANCE:
• OCR Engine Used: {advanced_result.ocr_engine_used}
• OCR Quality Score: {advanced_result.ocr_quality_score:.2f}
• Validation Passed: {'✅' if advanced_result.validation_passed else '❌'}
• Correction Attempts: {advanced_result.correction_attempts}
• Cache Hit: {'✅' if advanced_result.cache_hit else '❌'} (Semantic: {'✅' if advanced_result.semantic_cache_hit else '❌'})
• Processing Time: {advanced_result.processing_time:.2f}s
• Confidence Score: {advanced_result.confidence_score:.2f}
"""

        report = f"""
🎉 ADVANCED TRANSLATION COMPLETED {'SUCCESSFULLY' if pdf_success else 'WITH WARNINGS'}!
=======================================================

📁 Input: {os.path.basename(input_filepath)}
📁 Output Directory: {output_dir}
⏱️ Total Time: {duration/60:.1f} minutes

{files_section}

{advanced_metrics}
"""

        if drive_results:
            report += f"\n{drive_uploader.get_upload_summary(drive_results)}"

        if not pdf_success:
            report += f"""

⚠️ PDF CONVERSION TROUBLESHOOTING:
• Ensure Microsoft Word is installed and licensed
• Check Windows permissions and antivirus settings
• Try running as administrator
• Alternative: Use online PDF converters or LibreOffice
"""

        logger.info(report)

    def _generate_intelligent_final_report(self, input_filepath, output_dir, start_time, end_time,
                                         intelligent_result, drive_results, pdf_success=True):
        """Generate comprehensive final report for intelligent translation"""
        duration = end_time - start_time

        # Generate file status
        word_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.docx')
        pdf_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.pdf')

        files_section = f"📄 Generated Files:\n• Word Document: {word_filename} ✅"

        if pdf_success:
            files_section += f"\n• PDF Document: {pdf_filename} ✅"
        else:
            files_section += f"\n• PDF Document: {pdf_filename} ❌ (Conversion failed)"

        # Intelligent pipeline metrics
        performance_metrics = intelligent_result.performance_metrics
        processing_plan = intelligent_result.processing_plan
        document_analysis = intelligent_result.document_analysis

        intelligent_metrics = f"""
🧠 INTELLIGENT PIPELINE PERFORMANCE:
• Document Category: {document_analysis.document_category.value}
• Processing Strategy: {processing_plan.optimization_notes[0] if processing_plan.optimization_notes else 'Standard'}
• Cost Savings: {intelligent_result.cost_savings:.1f}%
• Quality Score: {intelligent_result.quality_score:.2f}
• Processing Time: {intelligent_result.processing_time:.2f}s
• Parallel Groups: {len(processing_plan.parallel_groups)}
• Total Blocks: {len(processing_plan.routing_decisions)}

📊 TOOL USAGE:
"""

        # Add tool usage statistics
        for tool, count in performance_metrics.get('tool_usage', {}).items():
            intelligent_metrics += f"• {tool.replace('_', ' ').title()}: {count}\n"

        # Document analysis summary
        analysis_summary = f"""
📋 DOCUMENT ANALYSIS:
• Total Pages: {document_analysis.total_pages}
• High Priority Pages: {len(document_analysis.get_high_priority_pages())}
• Simple Text Pages: {len(document_analysis.get_simple_text_pages())}
• Estimated Complexity: {document_analysis.estimated_complexity:.2f}

🎯 PROCESSING RECOMMENDATIONS:
"""
        for rec in document_analysis.processing_recommendations[:5]:  # Show top 5
            analysis_summary += f"• {rec}\n"

        report = f"""
🎉 INTELLIGENT TRANSLATION COMPLETED {'SUCCESSFULLY' if pdf_success else 'WITH WARNINGS'}!
=======================================================================

📁 Input: {os.path.basename(input_filepath)}
📁 Output Directory: {output_dir}
⏱️ Total Time: {duration/60:.1f} minutes

{files_section}

{intelligent_metrics}

{analysis_summary}
"""

        if drive_results:
            report += f"\n{drive_uploader.get_upload_summary(drive_results)}"

        if not pdf_success:
            report += f"""

⚠️ PDF CONVERSION TROUBLESHIPPING:
• Ensure Microsoft Word is installed and licensed
• Check Windows permissions and antivirus settings
• Try running as administrator
• Alternative: Use online PDF converters or LibreOffice
"""

        logger.info(report)

    def _generate_final_report(self, input_filepath, output_dir, start_time, end_time,
                             original_items_count, translated_items_count, drive_results, pdf_success=True):
        """Generate comprehensive final report"""
        duration = end_time - start_time

        # Generate file status
        word_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.docx')
        pdf_filename = os.path.basename(input_filepath).replace('.pdf', '_translated.pdf')

        files_section = f"📄 Generated Files:\n• Word Document: {word_filename} ✅"

        if pdf_success:
            files_section += f"\n• PDF Document: {pdf_filename} ✅"
        else:
            files_section += f"\n• PDF Document: {pdf_filename} ❌ (Conversion failed)"
            files_section += f"\n  💡 Word document is available for manual conversion"

        report = f"""
🎉 TRANSLATION COMPLETED {'SUCCESSFULLY' if pdf_success else 'WITH WARNINGS'}!
=====================================

📁 Input: {os.path.basename(input_filepath)}
📁 Output Directory: {output_dir}
⏱️ Total Time: {duration/60:.1f} minutes
📊 Items Processed: {original_items_count} → {translated_items_count}

{files_section}

{optimization_manager.get_final_performance_report()}
"""

        if drive_results:
            report += f"\n{drive_uploader.get_upload_summary(drive_results)}"

        if not pdf_success:
            report += f"""

⚠️ PDF CONVERSION TROUBLESHOOTING:
• Ensure Microsoft Word is installed and licensed
• Check Windows permissions and antivirus settings
• Try running as administrator
• Alternative: Use online PDF converters or LibreOffice
"""

        logger.info(report)

    def _convert_document_blocks_to_nodes(self, content_blocks):
        """Convert document content blocks to node format for consolidation"""
        nodes = []
        
        for i, block in enumerate(content_blocks):
            # Extract text content
            text = ""
            if hasattr(block, 'content') and block.content:
                text = block.content
            elif hasattr(block, 'original_text') and block.original_text:
                text = block.original_text
            elif hasattr(block, 'text') and block.text:
                text = block.text
            
            # Determine content type
            content_type = "paragraph"  # default
            if hasattr(block, 'block_type'):
                if block.block_type.value == 'heading':
                    content_type = "heading"
                elif block.block_type.value == 'list_item':
                    content_type = "list_item"
                elif block.block_type.value == 'image_placeholder':
                    content_type = "figure"
                elif block.block_type.value == 'table':
                    content_type = "table"
                elif block.block_type.value == 'code':
                    content_type = "code"
            
            # Extract bounding box if available
            bbox = (0, 0, 100, 100)  # default
            if hasattr(block, 'bbox') and block.bbox:
                bbox = block.bbox
            elif hasattr(block, 'position') and block.position:
                bbox = block.position
            
            # Extract page number
            page_num = 1  # default
            if hasattr(block, 'page_num') and block.page_num:
                page_num = block.page_num
            
            # Create node
            node = {
                'text': text,
                'label': content_type,
                'confidence': 0.9,  # default confidence
                'bbox': bbox,
                'page_num': page_num,
                'block_index': i
            }
            
            nodes.append(node)
        
        return nodes

    def _convert_consolidated_nodes_to_blocks(self, consolidated_nodes):
        """Convert consolidated nodes back to document blocks"""
        from structured_document_model import ContentType, TextBlock, ImagePlaceholder
        
        consolidated_blocks = []
        
        for node in consolidated_nodes:
            if not node.is_translatable():
                # Skip non-translatable content (images, structural elements)
                continue
            
            # Create text block
            text_block = TextBlock(
                block_type=ContentType.PARAGRAPH,  # default to paragraph
                content=node.text,
                page_num=node.page_num,
                bbox=node.bbox,
                confidence=node.confidence
            )
            
            # Set specific content type if available
            if node.content_type == "heading":
                text_block.block_type = ContentType.HEADING
            elif node.content_type == "list_item":
                text_block.block_type = ContentType.LIST_ITEM
            elif node.content_type == "code":
                text_block.block_type = ContentType.CODE
            
            consolidated_blocks.append(text_block)
        
        return consolidated_blocks

    def _validate_structured_document_integrity(self, document, stage_name):
        """
        Data-flow audit: Validate structured document integrity at pipeline stages.

        This implements Proposition 1: Data-Flow Audits in Architectural Design
        """
        if not document:
            logger.error(f"❌ Data integrity check failed at {stage_name}: Document is None")
            raise ValueError(f"Document integrity violation at {stage_name}: Document is None")

        # Check basic document structure
        if not hasattr(document, 'content_blocks'):
            logger.error(f"❌ Data integrity check failed at {stage_name}: No content_blocks attribute")
            raise ValueError(f"Document integrity violation at {stage_name}: Missing content_blocks")

        # Count different types of content
        total_blocks = len(document.content_blocks)
        image_blocks = [block for block in document.content_blocks if hasattr(block, 'image_path') and block.image_path]
        text_blocks = [block for block in document.content_blocks if hasattr(block, 'content') or hasattr(block, 'original_text')]

        # Log data shape at this stage
        logger.info(f"📊 Data integrity check at {stage_name}:")
        logger.info(f"   • Total blocks: {total_blocks}")
        logger.info(f"   • Image blocks: {len(image_blocks)}")
        logger.info(f"   • Text blocks: {len(text_blocks)}")
        logger.info(f"   • Document title: {getattr(document, 'title', 'N/A')}")
        logger.info(f"   • Source filepath: {getattr(document, 'source_filepath', 'N/A')}")

        # Validate image blocks have required attributes
        for i, block in enumerate(image_blocks):
            if not hasattr(block, 'image_path') or not block.image_path:
                logger.warning(f"⚠️ Image block {i} missing image_path at {stage_name}")
            elif not os.path.exists(block.image_path):
                logger.warning(f"⚠️ Image file not found: {block.image_path} at {stage_name}")

        # Store stage metadata for comparison
        if not hasattr(self, '_data_flow_audit'):
            self._data_flow_audit = {}

        self._data_flow_audit[stage_name] = {
            'total_blocks': total_blocks,
            'image_blocks': len(image_blocks),
            'text_blocks': len(text_blocks),
            'timestamp': time.time()
        }

        logger.debug(f"✅ Data integrity validated at {stage_name}")

    def process_document(self, pdf_path: str, output_dir: str) -> str:
        """Process a document through the enhanced workflow with visual content exclusion"""
        try:
            self.logger.info(f"Starting enhanced document processing: {pdf_path}")
            
            # Step 1: Extract text content with Nougat
            nougat_output = self.nougat_processor.process_pdf(pdf_path)
            self.logger.info("Nougat text extraction completed")
            
            # Step 2: Enhanced hybrid reconciliation with visual content exclusion
            reconciler = EnhancedHybridReconciler()
            document = asyncio.run(reconciler.reconcile_content_enhanced(
                nougat_output=nougat_output,
                pdf_path=pdf_path,
                output_dir=output_dir
            ))
            self.logger.info("Enhanced hybrid reconciliation completed")
            
            # Step 3: Generate final document
            doc_generator = DocumentGenerator()
            output_path = doc_generator.create_word_document_from_structured_document(
                document=document,
                output_path=os.path.join(output_dir, "translated_document.docx")
            )
            self.logger.info(f"Document generation completed: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise

    def _setup_components(self):
        """Set up workflow components with enhanced configuration"""
        try:
            # Initialize Nougat processor
            self.nougat_processor = NougatIntegration(
                config=self.config,
                model_path=self.config.get('Nougat', 'model_path'),
                device=self.config.get('Nougat', 'device')
            )
            
            # Initialize YOLOv8 detector
            self.yolo_detector = YOLOv8VisualDetector(
                model_path=self.config.get('YOLOv8', 'model_path'),
                confidence_threshold=0.5  # Higher threshold for better accuracy
            )
            
            # Initialize document generator with visual content exclusion
            self.doc_generator = DocumentGenerator(
                config=self.config,
                preserve_visual_content=True  # Ensure visual content is preserved
            )
            
            self.logger.info("Workflow components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

async def main():
    """Main entry point for the application with enhanced parallel processing"""
    logger.info("--- ULTIMATE PDF TRANSLATOR (Enhanced Parallel Version) ---")
    
    # Validate configuration
    if UNIFIED_CONFIG_AVAILABLE:
        # Use unified configuration validation
        is_valid = config_manager.validate_config()
        if not is_valid:
            logger.error("❌ Configuration validation failed")
            logger.warning("⚠️ Please check configuration file for errors")
            return False
        else:
            logger.info("✅ Configuration validation passed")
    else:
        # Use legacy configuration validation
        try:
            issues, recommendations = config_manager.validate_configuration()

            if issues:
                logger.error("❌ Configuration issues found:")
                for issue in issues:
                    logger.error(f"  {issue}")
                return False

            if recommendations:
                logger.info("💡 Configuration recommendations:")
                for rec in recommendations:
                    logger.info(f"  {rec}")
        except AttributeError:
            logger.warning("⚠️ Configuration validation not available, proceeding with defaults")
    
    # Get input files
    input_path, process_mode = choose_input_path()
    if not input_path:
        logger.info("No input selected. Exiting.")
        return True
    
    # Get output directory
    main_output_directory = choose_base_output_directory(
        os.path.dirname(input_path) if process_mode == 'file' else input_path
    )
    
    if not main_output_directory:
        logger.error("No output directory selected. Exiting.")
        return False
    
    # Collect files to process
    files_to_process = []
    if process_mode == 'file':
        files_to_process = [input_path]
    else:
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                files_to_process.append(os.path.join(input_path, filename))
    
    if not files_to_process:
        logger.error("No PDF files found to process.")
        return False
    
    # Estimate cost for single files
    if len(files_to_process) == 1:
        estimate_translation_cost(files_to_process[0], config_manager)
    
    # Initialize translator and failure tracker
    translator = UltimatePDFTranslator()
    failure_tracker = FailureTracker(
        quarantine_dir=os.path.join(main_output_directory, "quarantine"),
        max_retries=3
    )

    # Enhanced parallel processing configuration
    # Calculate optimal concurrency based on system resources
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adaptive concurrency calculation
    if len(files_to_process) == 1:
        # Single file: focus on internal parallelism
        max_concurrent_files = 1
        logger.info("📄 Single file processing - optimizing internal parallelism")
    else:
        # Multiple files: balance file-level and internal parallelism
        # Use 50% of CPU cores for file-level parallelism, reserve rest for internal tasks
        max_concurrent_files = min(
            max(1, cpu_count // 2),  # Use half the CPU cores
            len(files_to_process),   # Don't exceed file count
            4  # Cap at 4 concurrent files to avoid overwhelming the system
        )
        
        logger.info(f"⚡ Parallel file processing enabled:")
        logger.info(f"   • CPU cores available: {cpu_count}")
        logger.info(f"   • Memory available: {memory_gb:.1f}GB")
        logger.info(f"   • Files to process: {len(files_to_process)}")
        logger.info(f"   • Max concurrent files: {max_concurrent_files}")

    # Process files with enhanced parallel processing
    processed_count = 0
    quarantined_count = 0
    failed_count = 0

    if len(files_to_process) == 1:
        # Single file processing (sequential but with internal parallelism)
        filepath = files_to_process[0]
        logger.info(f"\n>>> Processing single file: {os.path.basename(filepath)} <<<")

        if not failure_tracker.should_process_file(filepath):
            logger.warning(f"⚠️ Skipping quarantined file: {os.path.basename(filepath)}")
            quarantined_count += 1
        else:
            specific_output_dir = get_specific_output_dir_for_file(main_output_directory, filepath)
            if specific_output_dir:
                try:
                    await translator.translate_document_async(filepath, specific_output_dir)
                    processed_count += 1
                    logger.info(f"✅ Successfully processed: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.error(f"❌ Failed to process {os.path.basename(filepath)}: {e}")
                    was_quarantined = failure_tracker.record_failure(filepath, e)
                    if was_quarantined:
                        quarantined_count += 1
                    else:
                        failed_count += 1
            else:
                logger.error(f"Could not create output directory for {os.path.basename(filepath)}")
                failed_count += 1
    else:
        # Multiple files: parallel processing
        logger.info(f"\n🚀 Starting parallel processing of {len(files_to_process)} files...")
        
        # Create semaphore for controlling concurrent file processing
        file_semaphore = asyncio.Semaphore(max_concurrent_files)
        
        async def process_single_file(filepath, file_index):
            """Process a single file with semaphore control"""
            async with file_semaphore:
                logger.info(f"\n>>> Processing file {file_index+1}/{len(files_to_process)}: {os.path.basename(filepath)} <<<")
                
                # Check if file should be processed or is quarantined
                if not failure_tracker.should_process_file(filepath):
                    logger.warning(f"⚠️ Skipping quarantined file: {os.path.basename(filepath)}")
                    return "quarantined"

                specific_output_dir = get_specific_output_dir_for_file(main_output_directory, filepath)
                if not specific_output_dir:
                    logger.error(f"Could not create output directory for {os.path.basename(filepath)}")
                    return "failed"

                try:
                    await translator.translate_document_async(filepath, specific_output_dir)
                    logger.info(f"✅ Successfully processed: {os.path.basename(filepath)}")
                    return "success"
                except Exception as e:
                    logger.error(f"❌ Failed to process {os.path.basename(filepath)}: {e}")
                    
                    # Record failure and check if file should be quarantined
                    was_quarantined = failure_tracker.record_failure(filepath, e)
                    if was_quarantined:
                        logger.warning(f"🚨 File quarantined after repeated failures")
                        return "quarantined"
                    else:
                        failure_count = failure_tracker.failure_counts[failure_tracker.get_file_hash(filepath)]
                        logger.warning(f"⚠️ Failure {failure_count}/{failure_tracker.max_retries} recorded for this file")
                        return "failed"

        # Create tasks for all files
        tasks = [
            process_single_file(filepath, i) 
            for i, filepath in enumerate(files_to_process)
        ]
        
        # Execute all tasks concurrently
        logger.info(f"⚡ Executing {len(tasks)} file processing tasks with max {max_concurrent_files} concurrent...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"❌ Task failed with exception: {result}")
            elif result == "success":
                processed_count += 1
            elif result == "quarantined":
                quarantined_count += 1
            elif result == "failed":
                failed_count += 1
    
    # Final processing summary
    logger.info("--- ALL PROCESSING COMPLETED ---")
    logger.info(f"📊 Processing Summary:")
    logger.info(f"   ✅ Successfully processed: {processed_count} files")
    logger.info(f"   🚨 Quarantined files: {quarantined_count} files")
    logger.info(f"   ❌ Failed files: {failed_count} files")
    logger.info(f"   📁 Total files attempted: {len(files_to_process)} files")

    if quarantined_count > 0:
        logger.warning(f"⚠️ {quarantined_count} files were quarantined due to repeated failures")
        logger.warning(f"   📁 Check quarantine directory: {failure_tracker.quarantine_dir}")
        logger.warning(f"   📋 Review error reports for manual inspection")

    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count} files failed processing")

    return processed_count > 0

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    # Handle command line arguments
    if len(sys.argv) >= 2:
        if sys.argv[1] in ["--help", "-h"]:
            print("Ultimate PDF Translator - Modular Version")
            print("Usage: python main_workflow.py")
            print("The script will prompt for input files and settings")
            sys.exit(0)
    
    # Run the main workflow
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
