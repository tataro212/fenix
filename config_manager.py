"""
Configuration Manager for Ultimate PDF Translator

Handles loading and accessing configuration settings from config.ini
Enhanced with comprehensive Pydantic settings models for type safety and validation.
"""

import os
import configparser
import logging
from typing import List, Optional, Literal
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Setup logging
logger = logging.getLogger(__name__)

class GeminiSettings(BaseSettings):
    """Pydantic model for Gemini API configuration with type safety and validation"""
    
    model_name: str = Field(default="gemini-1.5-pro-latest", description="Gemini model name")
    translation_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Translation temperature (0.0-2.0)")
    max_concurrent_api_calls: int = Field(default=5, ge=1, le=20, description="Maximum concurrent API calls")
    api_call_timeout_seconds: int = Field(default=600, ge=30, le=1800, description="API call timeout in seconds")
    api_key: Optional[str] = Field(default=None, description="Gemini API key")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name has proper prefix"""
        if not v.startswith("models/"):
            return f"models/{v}"
        return v
    
    class Config:
        env_prefix = "GEMINI_"

class PDFProcessingSettings(BaseSettings):
    """Pydantic model for PDF processing configuration"""
    
    # Content detection
    start_content_keywords: List[str] = Field(
        default_factory=lambda: [
            "introduction", "εισαγωγή", "πρόλογος", "foreword", "chapter 1", 
            "chapter i", "κεφάλαιο 1", "κεφάλαιο α", "μέρος πρώτο", "part one", 
            "summary", "περίληψη", "abstract"
        ],
        description="Keywords to detect content start"
    )
    bibliography_keywords: List[str] = Field(
        default_factory=lambda: [
            "bibliography", "references", "sources", "literature cited", 
            "works cited", "πηγές", "βιβλιογραφία", "αναφορές"
        ],
        description="Keywords to detect bibliography section"
    )
    toc_detection_keywords: List[str] = Field(
        default_factory=lambda: [
            "contents", "table of contents", "περιεχόμενα", "πίνακας περιεχομένων"
        ],
        description="Keywords to detect table of contents"
    )
    
    # Text processing
    max_chars_per_subchunk: int = Field(default=12000, ge=1000, le=50000, description="Maximum characters per text chunk")
    aggregate_small_chunks_target_size: int = Field(default=10000, ge=500, le=20000, description="Target size for aggregating small chunks")
    min_chars_for_standalone_chunk: int = Field(default=350, ge=50, le=2000, description="Minimum characters for standalone chunk")
    
    # Image extraction
    extract_images: bool = Field(default=True, description="Whether to extract images")
    min_image_width_px: int = Field(default=8, ge=1, le=100, description="Minimum image width in pixels")
    min_image_height_px: int = Field(default=8, ge=1, le=100, description="Minimum image height in pixels")
    
    # OCR settings
    perform_ocr: bool = Field(default=False, description="Whether to perform OCR on images")
    ocr_language: str = Field(default="eng", description="OCR language code")
    min_ocr_words_for_translation: int = Field(default=3, ge=1, le=20, description="Minimum OCR words to trigger translation")
    
    # Header detection
    heading_max_words: int = Field(default=13, ge=1, le=50, description="Maximum words in heading detection")
    
    # Table extraction
    extract_tables_as_images: bool = Field(default=True, description="Extract tables as images")
    min_table_columns: int = Field(default=2, ge=1, le=20, description="Minimum table columns")
    min_table_rows: int = Field(default=2, ge=1, le=100, description="Minimum table rows")
    min_table_width_points: int = Field(default=100, ge=10, le=1000, description="Minimum table width in points")
    min_table_height_points: int = Field(default=50, ge=10, le=1000, description="Minimum table height in points")
    
    # Equation extraction
    extract_equations_as_images: bool = Field(default=True, description="Extract equations as images")
    min_equation_width_points: int = Field(default=30, ge=5, le=500, description="Minimum equation width in points")
    min_equation_height_points: int = Field(default=15, ge=5, le=200, description="Minimum equation height in points")
    detect_math_symbols: bool = Field(default=True, description="Detect mathematical symbols")
    
    # Figure extraction
    extract_figures_by_caption: bool = Field(default=True, description="Extract figures by caption detection")
    min_figure_width_points: int = Field(default=50, ge=10, le=1000, description="Minimum figure width in points")
    min_figure_height_points: int = Field(default=50, ge=10, le=1000, description="Minimum figure height in points")
    max_caption_to_figure_distance_points: int = Field(default=100, ge=10, le=500, description="Maximum distance from caption to figure")
    
    class Config:
        env_prefix = "PDF_"

class WordOutputSettings(BaseSettings):
    """Pydantic model for Word document output configuration"""
    
    apply_styles_to_paragraphs: bool = Field(default=True, description="Apply styles to paragraphs")
    apply_styles_to_headings: bool = Field(default=True, description="Apply styles to headings")
    default_image_width_inches: float = Field(default=5.0, ge=0.1, le=20.0, description="Default image width in inches")
    generate_toc: bool = Field(default=True, description="Generate table of contents")
    toc_title: str = Field(default="Πίνακας Περιεχομένων", description="Title for table of contents")
    list_indent_per_level_inches: float = Field(default=0.25, ge=0.0, le=2.0, description="List indent per level in inches")
    heading_space_before_pt: int = Field(default=6, ge=0, le=50, description="Heading space before in points")
    paragraph_first_line_indent_inches: float = Field(default=0.0, ge=0.0, le=2.0, description="Paragraph first line indent in inches")
    paragraph_space_after_pt: int = Field(default=6, ge=0, le=50, description="Paragraph space after in points")
    
    class Config:
        env_prefix = "WORD_"

class TranslationSettings(BaseSettings):
    """Pydantic model for translation enhancement configuration"""
    
    target_language: str = Field(default="Ελληνικά", description="Target language for translation")
    use_glossary: bool = Field(default=False, description="Use glossary for translation")
    glossary_file_path: str = Field(default="glossary.json", description="Path to glossary file")
    use_translation_cache: bool = Field(default=True, description="Use translation cache")
    translation_cache_file_path: str = Field(default="translation_cache.json", description="Path to translation cache file")
    translation_style_tone: Literal["formal", "informal", "technical", "academic"] = Field(
        default="formal", description="Translation style and tone"
    )
    analyze_document_style_first: bool = Field(default=True, description="Analyze document style before translation")
    batch_style_analysis_reuse: bool = Field(default=True, description="Reuse style analysis for batch translation")
    perform_quality_assessment: bool = Field(default=True, description="Perform translation quality assessment")
    qa_strategy: Literal["full", "sample", "minimal"] = Field(default="full", description="Quality assessment strategy")
    qa_sample_percentage: float = Field(default=0.1, ge=0.0, le=1.0, description="QA sample percentage (0.0-1.0)")
    
    class Config:
        env_prefix = "TRANSLATION_"

class OptimizationSettings(BaseSettings):
    """Pydantic model for API optimization configuration"""
    
    enable_smart_grouping: bool = Field(default=True, description="Enable smart content grouping")
    max_group_size_chars: int = Field(default=12000, ge=1000, le=50000, description="Maximum group size in characters")
    max_items_per_group: int = Field(default=8, ge=1, le=50, description="Maximum items per group")
    enable_ocr_grouping: bool = Field(default=True, description="Enable OCR content grouping")
    aggressive_grouping_mode: bool = Field(default=True, description="Use aggressive grouping mode")
    smart_ocr_filtering: bool = Field(default=True, description="Use smart OCR filtering")
    min_ocr_words_for_translation_enhanced: int = Field(default=8, ge=1, le=50, description="Enhanced minimum OCR words for translation")
    
    # NEW: Performance optimization settings
    max_workers: int = Field(default=6, ge=1, le=20, description="Maximum worker processes")
    enable_concurrent_translation: bool = Field(default=True, description="Enable concurrent translation processing")
    translation_batch_size: int = Field(default=5, ge=1, le=20, description="Translation batch size for concurrent processing")
    
    class Config:
        env_prefix = "OPT_"

class GoogleDriveSettings(BaseSettings):
    """Pydantic model for Google Drive integration configuration"""
    
    target_folder_id: Optional[str] = Field(default=None, description="Google Drive target folder ID")
    credentials_file: str = Field(default="mycreds.txt", description="Path to Google Drive credentials file")
    
    @field_validator('target_folder_id')
    @classmethod
    def validate_folder_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize folder ID"""
        if v and v.lower() in ["none", "", "null"]:
            return None
        return v
    
    class Config:
        env_prefix = "GDRIVE_"

class YOLOv8Settings(BaseSettings):
    """Pydantic model for YOLOv8 configuration"""
    
    model_path: str = Field(default="yolov8m.pt", description="Path to YOLOv8 model file")
    fallback_model_path: str = Field(default="yolov8m.pt", description="Fallback model path if primary fails")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold (0.0-1.0)")
    iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="IoU threshold for NMS (0.0-1.0)")
    max_detections: int = Field(default=100, ge=1, le=1000, description="Maximum number of detections per image")
    image_size: int = Field(default=640, ge=320, le=1280, description="Input image size for YOLO model")
    device_preference: Literal["auto", "cuda", "cpu"] = Field(default="auto", description="Device preference for inference")
    
    class Config:
        env_prefix = "YOLO_"

class PyMuPDFSettings(BaseSettings):
    """Pydantic model for PyMuPDF processing configuration"""
    
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for PyMuPDF extractions")
    bbox_overlap_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Bounding box overlap threshold")
    hyphenation_reconstruction: bool = Field(default=True, description="Enable hyphenation reconstruction")
    text_density_calculation: bool = Field(default=True, description="Enable text density calculation")
    visual_density_calculation: bool = Field(default=True, description="Enable visual density calculation")
    
    class Config:
        env_prefix = "PYMUPDF_"

class AppSettings(BaseSettings):
    """
    Comprehensive Pydantic-based application settings.
    
    This replaces the manual configparser approach with type-safe,
    validated configuration management.
    """
    
    # Core settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")
    
    # Component settings
    gemini: GeminiSettings = Field(default_factory=GeminiSettings, description="Gemini API settings")
    pdf_processing: PDFProcessingSettings = Field(default_factory=PDFProcessingSettings, description="PDF processing settings")
    word_output: WordOutputSettings = Field(default_factory=WordOutputSettings, description="Word output settings")
    translation: TranslationSettings = Field(default_factory=TranslationSettings, description="Translation settings")
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings, description="Optimization settings")
    google_drive: GoogleDriveSettings = Field(default_factory=GoogleDriveSettings, description="Google Drive settings")
    # NEW: Model and processing settings
    yolov8: YOLOv8Settings = Field(default_factory=YOLOv8Settings, description="YOLOv8 model settings")
    pymupdf: PyMuPDFSettings = Field(default_factory=PyMuPDFSettings, description="PyMuPDF processing settings")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Legacy Config class for backward compatibility
class Config(BaseSettings):
    MAX_WORKERS: int = 6
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ConfigManager:
    """
    Enhanced Configuration Manager with Pydantic type safety and validation.
    
    This provides a hybrid approach: loads from config.ini for backward compatibility,
    but exposes type-safe Pydantic models for all settings access.
    """
    
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.user_config_read_successfully = False
        
        # Load configuration from file and environment
        self._load_config()
        self._load_environment()
        
        # NEW: Create Pydantic settings with loaded values
        self.settings = self._create_pydantic_settings()
        
        # Initialize API with validated settings
        self._initialize_api()
        
    def _load_config(self):
        """Load configuration from config.ini"""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
                self.user_config_read_successfully = True
                logger.info("Settings loaded from user's config.ini. Defaults will be applied for missing values.")
            except configparser.Error as e:
                logger.error(f"Error reading config.ini: {e}. Using full default configuration.")
        else:
            logger.warning("config.ini not found. Using full default configuration.")
    
    def _load_environment(self):
        """Load environment variables"""
        load_dotenv()
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Some functionality will be limited.")
    
    def _create_pydantic_settings(self) -> AppSettings:
        """
        Create Pydantic settings by merging config.ini values with environment variables.
        
        This provides the type safety and validation benefits of Pydantic while
        maintaining backward compatibility with existing config.ini files.
        """
        try:
            # Prepare environment variables from config.ini
            config_env = {}
            
            # Map Gemini settings
            if self.config.has_section('GeminiAPI'):
                if self.config.has_option('GeminiAPI', 'model_name'):
                    config_env['GEMINI_MODEL_NAME'] = self.config.get('GeminiAPI', 'model_name')
                if self.config.has_option('GeminiAPI', 'translation_temperature'):
                    config_env['GEMINI_TRANSLATION_TEMPERATURE'] = self.config.get('GeminiAPI', 'translation_temperature')
                if self.config.has_option('GeminiAPI', 'max_concurrent_api_calls'):
                    config_env['GEMINI_MAX_CONCURRENT_API_CALLS'] = self.config.get('GeminiAPI', 'max_concurrent_api_calls')
                if self.config.has_option('GeminiAPI', 'api_call_timeout_seconds'):
                    config_env['GEMINI_API_CALL_TIMEOUT_SECONDS'] = self.config.get('GeminiAPI', 'api_call_timeout_seconds')
            
            # Add API key from environment loading
            if self.api_key:
                config_env['GEMINI_API_KEY'] = self.api_key
            
            # Map PDF processing settings
            if self.config.has_section('PDFProcessing'):
                if self.config.has_option('PDFProcessing', 'max_chars_per_subchunk'):
                    config_env['PDF_MAX_CHARS_PER_SUBCHUNK'] = self.config.get('PDFProcessing', 'max_chars_per_subchunk')
                if self.config.has_option('PDFProcessing', 'extract_images'):
                    config_env['PDF_EXTRACT_IMAGES'] = self.config.get('PDFProcessing', 'extract_images')
                # Add more mappings as needed...
            
            # Map optimization settings
            if self.config.has_section('APIOptimization'):
                if self.config.has_option('APIOptimization', 'enable_smart_grouping'):
                    config_env['OPT_ENABLE_SMART_GROUPING'] = self.config.get('APIOptimization', 'enable_smart_grouping')
                if self.config.has_option('APIOptimization', 'max_group_size_chars'):
                    config_env['OPT_MAX_GROUP_SIZE_CHARS'] = self.config.get('APIOptimization', 'max_group_size_chars')
                # Add more mappings as needed...
            
            # Map YOLOv8 settings (NEW)
            if self.config.has_section('YOLOv8'):
                if self.config.has_option('YOLOv8', 'model_path'):
                    config_env['YOLO_MODEL_PATH'] = self.config.get('YOLOv8', 'model_path')
                if self.config.has_option('YOLOv8', 'confidence_threshold'):
                    config_env['YOLO_CONFIDENCE_THRESHOLD'] = self.config.get('YOLOv8', 'confidence_threshold')
                if self.config.has_option('YOLOv8', 'iou_threshold'):
                    config_env['YOLO_IOU_THRESHOLD'] = self.config.get('YOLOv8', 'iou_threshold')
                if self.config.has_option('YOLOv8', 'max_detections'):
                    config_env['YOLO_MAX_DETECTIONS'] = self.config.get('YOLOv8', 'max_detections')
                if self.config.has_option('YOLOv8', 'image_size'):
                    config_env['YOLO_IMAGE_SIZE'] = self.config.get('YOLOv8', 'image_size')
                if self.config.has_option('YOLOv8', 'device_preference'):
                    config_env['YOLO_DEVICE_PREFERENCE'] = self.config.get('YOLOv8', 'device_preference')
            
            # Map PyMuPDF settings (NEW)
            if self.config.has_section('PyMuPDFProcessing'):
                if self.config.has_option('PyMuPDFProcessing', 'extraction_confidence'):
                    config_env['PYMUPDF_EXTRACTION_CONFIDENCE'] = self.config.get('PyMuPDFProcessing', 'extraction_confidence')
                if self.config.has_option('PyMuPDFProcessing', 'bbox_overlap_threshold'):
                    config_env['PYMUPDF_BBOX_OVERLAP_THRESHOLD'] = self.config.get('PyMuPDFProcessing', 'bbox_overlap_threshold')
                if self.config.has_option('PyMuPDFProcessing', 'hyphenation_reconstruction'):
                    config_env['PYMUPDF_HYPHENATION_RECONSTRUCTION'] = self.config.get('PyMuPDFProcessing', 'hyphenation_reconstruction')
            
            # Temporarily set environment variables for Pydantic to read
            original_env = {}
            for key, value in config_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = str(value)
            
            try:
                # Create Pydantic settings (will read from environment)
                settings = AppSettings()
                logger.info("✅ Pydantic settings created successfully with type validation")
                return settings
            finally:
                # Restore original environment
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                        
        except Exception as e:
            logger.warning(f"⚠️ Failed to create Pydantic settings: {e}. Using defaults.")
            return AppSettings()  # Use all defaults
    
    def _initialize_api(self):
        """Initialize Gemini API using validated Pydantic settings"""
        api_key = self.settings.gemini.api_key
        if api_key:
            genai.configure(api_key=api_key)
            logger.info("✅ Gemini API configured successfully with validated settings")
        else:
            logger.warning("⚠️ No Gemini API key available")
    
    def get_config_value(self, section, key, default, type_func=str):
        """Get configuration value with type conversion and defaults"""
        if not self.user_config_read_successfully and not self.config.has_section(section):
            return default
            
        if self.config.has_section(section) and self.config.has_option(section, key):
            value = self.config.get(section, key)
            value = value.split('#')[0].split(';')[0].strip()
            
            try:
                if type_func == bool:
                    if isinstance(value, bool):
                        return value
                    if value.lower() in ('true', 'yes', 'on', '1'):
                        return True
                    if value.lower() in ('false', 'no', 'off', '0'):
                        return False
                    raise ValueError(f"Not a boolean: {value}")
                elif value is None and type_func == str:
                    return default
                return type_func(value if value is not None else default)
            except ValueError:
                return default
        return default
    
    @property
    def gemini_settings(self):
        """Get Gemini API settings (ENHANCED: Now using validated Pydantic model)"""
        return {
            'model_name': self.settings.gemini.model_name,
            'temperature': self.settings.gemini.translation_temperature,
            'max_concurrent_calls': self.settings.gemini.max_concurrent_api_calls,
            'timeout': self.settings.gemini.api_call_timeout_seconds,
            'api_key': self.settings.gemini.api_key
        }
    
    @property
    def pdf_processing_settings(self):
        """Get PDF processing settings"""
        start_keywords_str = self.get_config_value('PDFProcessing', 'start_content_keywords', 
            "introduction, εισαγωγή, πρόλογος, foreword, chapter 1, chapter i, κεφάλαιο 1, κεφάλαιο α, μέρος πρώτο, part one, summary, περίληψη, abstract")
        
        return {
            'start_content_keywords': [k.strip().lower() for k in start_keywords_str.split(',') if k.strip()],
            'bibliography_keywords': self._parse_keyword_list('PDFProcessing', 'bibliography_keywords', 
                "bibliography, references, sources, literature cited, works cited, πηγές, βιβλιογραφία, αναφορές"),
            'toc_detection_keywords': self._parse_keyword_list('PDFProcessing', 'toc_detection_keywords',
                "contents, table of contents, περιεχόμενα, πίνακας περιεχομένων"),
            'max_chars_per_subchunk': self.get_config_value('PDFProcessing', 'max_chars_per_subchunk', 12000, int),
            'aggregate_small_chunks_target_size': self.get_config_value('PDFProcessing', 'aggregate_small_chunks_target_size', 10000, int),
            'min_chars_for_standalone_chunk': self.get_config_value('PDFProcessing', 'min_chars_for_standalone_chunk', 350, int),
            'extract_images': self.get_config_value('PDFProcessing', 'extract_images', True, bool),
            'perform_ocr': self.get_config_value('PDFProcessing', 'perform_ocr_on_images', False, bool),
            'ocr_language': self.get_config_value('PDFProcessing', 'ocr_language', "eng"),
            'min_ocr_words_for_translation': self.get_config_value('PDFProcessing', 'min_ocr_words_for_translation', 3, int),

            # Image extraction settings
            'min_image_width_px': self.get_config_value('PDFProcessing', 'min_image_width_px', 8, int),
            'min_image_height_px': self.get_config_value('PDFProcessing', 'min_image_height_px', 8, int),

            # Header detection settings
            'heading_max_words': self.get_config_value('PDFProcessing', 'heading_max_words', 13, int),

            # Table extraction settings
            'extract_tables_as_images': self.get_config_value('PDFProcessing', 'extract_tables_as_images', True, bool),
            'min_table_columns': self.get_config_value('PDFProcessing', 'min_table_columns', 2, int),
            'min_table_rows': self.get_config_value('PDFProcessing', 'min_table_rows', 2, int),
            'min_table_width_points': self.get_config_value('PDFProcessing', 'min_table_width_points', 100, int),
            'min_table_height_points': self.get_config_value('PDFProcessing', 'min_table_height_points', 50, int),

            # Equation extraction settings
            'extract_equations_as_images': self.get_config_value('PDFProcessing', 'extract_equations_as_images', True, bool),
            'min_equation_width_points': self.get_config_value('PDFProcessing', 'min_equation_width_points', 30, int),
            'min_equation_height_points': self.get_config_value('PDFProcessing', 'min_equation_height_points', 15, int),
            'detect_math_symbols': self.get_config_value('PDFProcessing', 'detect_math_symbols', True, bool),

            # Figure extraction settings
            'extract_figures_by_caption': self.get_config_value('PDFProcessing', 'extract_figures_by_caption', True, bool),
            'min_figure_width_points': self.get_config_value('PDFProcessing', 'min_figure_width_points', 50, int),
            'min_figure_height_points': self.get_config_value('PDFProcessing', 'min_figure_height_points', 50, int),
            'max_caption_to_figure_distance_points': self.get_config_value('PDFProcessing', 'max_caption_to_figure_distance_points', 100, int)
        }
    
    @property
    def word_output_settings(self):
        """Get Word document output settings"""
        return {
            'apply_styles_to_paragraphs': self.get_config_value('WordOutput', 'apply_styles_to_paragraphs', True, bool),
            'apply_styles_to_headings': self.get_config_value('WordOutput', 'apply_styles_to_headings', True, bool),
            'default_image_width_inches': self.get_config_value('WordOutput', 'default_image_width_inches', 5.0, float),
            'generate_toc': True,  # Re-enabled with improved implementation
            'toc_title': self.get_config_value('WordOutput', 'toc_title', "Πίνακας Περιεχομένων"),
            'list_indent_per_level_inches': self.get_config_value('WordOutput', 'list_indent_per_level_inches', 0.25, float),
            'heading_space_before_pt': self.get_config_value('WordOutput', 'heading_space_before_pt', 6, int),
            'paragraph_first_line_indent_inches': self.get_config_value('WordOutput', 'paragraph_first_line_indent_inches', 0.0, float),
            'paragraph_space_after_pt': self.get_config_value('WordOutput', 'paragraph_space_after_pt', 6, int)
        }
    
    @property
    def translation_enhancement_settings(self):
        """Get translation enhancement settings"""
        return {
            'target_language': self.get_config_value('TranslationEnhancements', 'target_language', "Ελληνικά"),
            'use_glossary': self.get_config_value('TranslationEnhancements', 'use_glossary', False, bool),
            'glossary_file_path': self.get_config_value('TranslationEnhancements', 'glossary_file_path', "glossary.json"),
            'use_translation_cache': self.get_config_value('TranslationEnhancements', 'use_translation_cache', True, bool),
            'translation_cache_file_path': self.get_config_value('TranslationEnhancements', 'translation_cache_file_path', "translation_cache.json"),
            'translation_style_tone': self.get_config_value('TranslationEnhancements', 'translation_style_tone', "formal").strip().lower(),
            'analyze_document_style_first': self.get_config_value('TranslationEnhancements', 'analyze_document_style_first', True, bool),
            'batch_style_analysis_reuse': self.get_config_value('TranslationEnhancements', 'batch_style_analysis_reuse', True, bool),
            'perform_quality_assessment': self.get_config_value('TranslationEnhancements', 'perform_quality_assessment', True, bool),
            'qa_strategy': self.get_config_value('TranslationEnhancements', 'qa_strategy', 'full', str).lower(),
            'qa_sample_percentage': self.get_config_value('TranslationEnhancements', 'qa_sample_percentage', 0.1, float)
        }
    
    @property
    def optimization_settings(self):
        """Get API optimization settings (ENHANCED: Now using validated Pydantic model with new performance settings)"""
        return {
            'enable_smart_grouping': self.settings.optimization.enable_smart_grouping,
            'max_group_size_chars': self.settings.optimization.max_group_size_chars,
            'max_items_per_group': self.settings.optimization.max_items_per_group,
            'enable_ocr_grouping': self.settings.optimization.enable_ocr_grouping,
            'aggressive_grouping_mode': self.settings.optimization.aggressive_grouping_mode,
            'smart_ocr_filtering': self.settings.optimization.smart_ocr_filtering,
            'min_ocr_words_for_translation_enhanced': self.settings.optimization.min_ocr_words_for_translation_enhanced,
            # NEW: Performance optimization settings
            'max_workers': self.settings.optimization.max_workers,
            'enable_concurrent_translation': self.settings.optimization.enable_concurrent_translation,
            'translation_batch_size': self.settings.optimization.translation_batch_size
        }
    
    @property
    def yolov8_settings(self):
        """Get YOLOv8 model settings (NEW: Centralized configuration)"""
        return {
            'model_path': self.settings.yolov8.model_path,
            'fallback_model_path': self.settings.yolov8.fallback_model_path,
            'confidence_threshold': self.settings.yolov8.confidence_threshold,
            'iou_threshold': self.settings.yolov8.iou_threshold,
            'max_detections': self.settings.yolov8.max_detections,
            'image_size': self.settings.yolov8.image_size,
            'device_preference': self.settings.yolov8.device_preference
        }
    
    @property
    def pymupdf_settings(self):
        """Get PyMuPDF processing settings (NEW: Centralized configuration)"""
        return {
            'extraction_confidence': self.settings.pymupdf.extraction_confidence,
            'bbox_overlap_threshold': self.settings.pymupdf.bbox_overlap_threshold,
            'hyphenation_reconstruction': self.settings.pymupdf.hyphenation_reconstruction,
            'text_density_calculation': self.settings.pymupdf.text_density_calculation,
            'visual_density_calculation': self.settings.pymupdf.visual_density_calculation
        }
    
    @property
    def google_drive_settings(self):
        """Get Google Drive settings"""
        folder_id = self.get_config_value('GoogleDrive', 'gdrive_target_folder_id', "")
        return {
            'target_folder_id': folder_id if folder_id and folder_id.lower() != "none" else None,
            'credentials_file': "mycreds.txt"
        }

    @property
    def enhanced_word_settings(self):
        """Get enhanced Word document settings"""
        base_settings = self.word_output_settings
        enhanced_settings = {
            'max_image_width_inches': self.get_config_value('WordOutput', 'max_image_width_inches', 6.5, float),
            'max_image_height_inches': self.get_config_value('WordOutput', 'max_image_height_inches', 8.0, float),
            'maintain_image_aspect_ratio': self.get_config_value('WordOutput', 'maintain_image_aspect_ratio', True, bool),
            'toc_max_heading_length': self.get_config_value('WordOutput', 'toc_max_heading_length', 80, int),
        }
        return {**base_settings, **enhanced_settings}

    @property
    def translation_strategy_settings(self):
        """Get translation strategy settings"""
        return {
            'translation_priority': self.get_config_value('TranslationStrategy', 'translation_priority', 'balanced'),
            'enable_importance_analysis': self.get_config_value('TranslationStrategy', 'enable_importance_analysis', True, bool),
            'skip_boilerplate_text': self.get_config_value('TranslationStrategy', 'skip_boilerplate_text', True, bool),
            'skip_code_blocks': self.get_config_value('TranslationStrategy', 'skip_code_blocks', True, bool),
        }

    @property
    def advanced_caching_settings(self):
        """Get advanced caching settings"""
        return {
            'max_cache_entries': self.get_config_value('AdvancedCaching', 'max_cache_entries', 10000, int),
            'similarity_threshold': self.get_config_value('AdvancedCaching', 'similarity_threshold', 0.85, float),
            'context_window_chars': self.get_config_value('AdvancedCaching', 'context_window_chars', 200, int),
            'enable_fuzzy_matching': self.get_config_value('AdvancedCaching', 'enable_fuzzy_matching', True, bool),
        }

    @property
    def ocr_preprocessing_settings(self):
        """Get OCR preprocessing settings"""
        return {
            'enable_ocr_grayscale': self.get_config_value('OCRPreprocessing', 'enable_ocr_grayscale', True, bool),
            'enable_binarization': self.get_config_value('OCRPreprocessing', 'enable_binarization', True, bool),
            'binarization_threshold': self.get_config_value('OCRPreprocessing', 'binarization_threshold', 'auto'),
            'enable_noise_reduction': self.get_config_value('OCRPreprocessing', 'enable_noise_reduction', True, bool),
            'enable_deskewing': self.get_config_value('OCRPreprocessing', 'enable_deskewing', False, bool),
            'enhance_contrast': self.get_config_value('OCRPreprocessing', 'enhance_contrast', True, bool),
            'upscale_factor': self.get_config_value('OCRPreprocessing', 'upscale_factor', 2.0, float),
            'ocr_dpi': self.get_config_value('OCRPreprocessing', 'ocr_dpi', 300, int),
        }
    
    def _parse_keyword_list(self, section, key, default):
        """Parse comma-separated keyword list"""
        keywords_str = self.get_config_value(section, key, default)
        return [k.strip().lower() for k in keywords_str.split(',') if k.strip()]
    
    def validate_configuration(self):
        """Validate configuration and return issues/recommendations"""
        issues = []
        recommendations = []
        
        # Check API Key
        if not self.api_key:
            issues.append("❌ GEMINI_API_KEY not found in environment variables")
            recommendations.append("💡 Set GEMINI_API_KEY in your .env file or environment")
        
        # Check model configuration
        gemini_settings = self.gemini_settings
        if "2.5-pro" in gemini_settings['model_name']:
            recommendations.append("💰 Consider using 'gemini-1.5-flash-latest' for cost efficiency")
        
        # Check batch settings
        opt_settings = self.optimization_settings
        if opt_settings['max_group_size_chars'] > 15000:
            recommendations.append("⚠️ Large batch size may cause API timeouts - consider reducing to 12000")
        
        if gemini_settings['max_concurrent_calls'] > 10:
            recommendations.append("⚠️ High concurrent calls may trigger rate limits - consider reducing to 5")
        
        # Check smart grouping
        if not opt_settings['enable_smart_grouping']:
            recommendations.append("💡 Enable smart_grouping for significant API cost reduction")
        
        return issues, recommendations

# Global configuration instance
try:
    config_manager = ConfigManager()
except Exception as e:
    print(f"Error creating global config_manager: {e}")
    import traceback
    traceback.print_exc()
    raise
