"""
Configuration Manager for Ultimate PDF Translator

Handles loading and accessing configuration settings from config.ini
"""

import os
import configparser
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Setup logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.user_config_read_successfully = False
        self._load_config()
        self._load_environment()
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
    
    def _initialize_api(self):
        """Initialize Gemini API if key is available"""
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully")
    
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
        """Get Gemini API settings"""
        model_name_config = self.get_config_value('GeminiAPI', 'model_name', "gemini-1.5-pro-latest")
        model_name = model_name_config if model_name_config.startswith("models/") else f"models/{model_name_config}"
        
        return {
            'model_name': model_name,
            'temperature': self.get_config_value('GeminiAPI', 'translation_temperature', 0.1, float),
            'max_concurrent_calls': self.get_config_value('GeminiAPI', 'max_concurrent_api_calls', 5, int),
            'timeout': self.get_config_value('GeminiAPI', 'api_call_timeout_seconds', 600, int),
            'api_key': self.api_key
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
        """Get API optimization settings"""
        return {
            'enable_smart_grouping': self.get_config_value('APIOptimization', 'enable_smart_grouping', True, bool),
            'max_group_size_chars': self.get_config_value('APIOptimization', 'max_group_size_chars', 12000, int),
            'max_items_per_group': self.get_config_value('APIOptimization', 'max_items_per_group', 8, int),
            'enable_ocr_grouping': self.get_config_value('APIOptimization', 'enable_ocr_grouping', True, bool),
            'aggressive_grouping_mode': self.get_config_value('APIOptimization', 'aggressive_grouping_mode', True, bool),
            'smart_ocr_filtering': self.get_config_value('APIOptimization', 'smart_ocr_filtering', True, bool),
            'min_ocr_words_for_translation_enhanced': self.get_config_value('APIOptimization', 'min_ocr_words_for_translation', 8, int)
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
