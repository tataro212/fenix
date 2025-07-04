"""
Gemini Service Module for Ultimate PDF Translator

Handles integration with Google's Gemini API for translation and text processing.
"""

import os
import logging
import google.generativeai as genai
from config_manager import config_manager
import asyncio
import time

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google's Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini service with API key and model configuration"""
        self.settings = config_manager.gemini_settings
        
        # Configure API key
        api_key = self.settings.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("No Gemini API key found. Please set GOOGLE_API_KEY environment variable or configure in settings.")
        
        genai.configure(api_key=api_key)
        
        # Initialize model - Use 1.5 Flash for faster processing
        model_name = self.settings.get('model_name', 'models/gemini-1.5-flash')
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"🚀 Gemini service initialized with model: {model_name}")
    
    async def translate_text(self, text: str, target_language: str, timeout: float = None) -> str:
        """Translate text using Gemini API with enhanced word boundary preservation, adaptive timeout, and exponential backoff."""
        max_retries = 3
        try:
            # Get timeout from config or use adaptive timeout based on text length
            if timeout is None:
                config_timeout = self.settings.get('api_call_timeout_seconds', 600)
                text_length = len(text)
                if text_length <= 1000:
                    timeout = 30.0
                elif text_length <= 5000:
                    timeout = 120.0
                elif text_length <= 10000:
                    timeout = 300.0
                else:
                    timeout = min(config_timeout, 600.0)
                logger.debug(f"📊 Adaptive timeout: {timeout}s for {text_length} chars")
            
            # SMART BATCHING FIX: Check if text contains XML segments - if so, treat as single unit
            if '<seg id=' in text and '</seg>' in text:
                logger.debug("🎯 XML segments detected - processing as single unit for smart batching")
                # Process the entire XML payload as one unit
                prompt = (
                    f"Translate the following XML content to {target_language}. "
                    "Preserve the <seg> tags and their id attributes exactly. "
                    "Do not add any text or explanation outside of the <seg> tags. "
                    f"TEXT TO TRANSLATE:\n{text}"
                )
                
                attempt = 0
                while attempt <= max_retries:
                    try:
                        # --- BEGIN LOGGING ADDITIONS (Directive 1.3) ---
                        logger.info(f"Sending request to Gemini for prompt: {prompt[:200]}...")
                        logger.debug(f"Full prompt for Gemini: {prompt}")
                        response = await asyncio.wait_for(
                            self.model.generate_content_async(prompt),
                            timeout=timeout
                        )
                        logger.debug(f"Received raw response from Gemini: {response}")
                        # --- END LOGGING ADDITIONS ---
                        # Defensive: check for .text or .parts
                        if hasattr(response, 'text') and response.text:
                            translated_text = response.text.strip()
                        elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                            translated_text = response.parts[0].text.strip()
                        else:
                            logger.error("Gemini response is empty or invalid.")
                            return ""
                        logger.info("Successfully received and parsed response from Gemini.")
                        return translated_text
                    except asyncio.TimeoutError as e:
                        if attempt == 0:
                            logger.warning(f"⚠️ Translation timeout after {timeout}s, using original text. Payload: {repr(text)[:500]}")
                        else:
                            logger.warning(f"⚠️ Retry {attempt}: Translation timeout after {timeout}s. Payload: {repr(text)[:500]}")
                        if attempt >= max_retries:
                            logger.error(f"❌ Max retries reached for timeout. Returning original text. Payload: {repr(text)[:500]}")
                            return text
                        time.sleep(2 ** attempt)
                        attempt += 1
                    except Exception as e:
                        logger.critical(f"FATAL: An unhandled exception occurred during Gemini API call: {e}")
                        if attempt >= max_retries:
                            logger.error(f"❌ Max retries reached for error. Returning original text. Payload: {repr(text)[:500]}")
                            return ""
                        time.sleep(2 ** attempt)
                        attempt += 1
            else:
                # LEGACY: For non-XML content, use original sentence-splitting logic
                logger.debug("📝 Non-XML content detected - using sentence-splitting logic")
                sentences = self._split_into_sentences(text)
                translated_sentences = []
                for sentence in sentences:
                    if not sentence.strip():
                        translated_sentences.append(sentence)
                        continue
                    prompt = (
                        f"Translate the following XML content to {target_language}. "
                        "Preserve the <seg> tags and their id attributes exactly. "
                        "Do not add any text or explanation outside of the <seg> tags. "
                        f"TEXT TO TRANSLATE:\n{sentence}"
                    )
                    attempt = 0
                    while attempt <= max_retries:
                        try:
                            # --- BEGIN LOGGING ADDITIONS (Directive 1.3) ---
                            logger.info(f"Sending request to Gemini for prompt: {prompt[:200]}...")
                            logger.debug(f"Full prompt for Gemini: {prompt}")
                            response = await asyncio.wait_for(
                                self.model.generate_content_async(prompt),
                                timeout=timeout
                            )
                            logger.debug(f"Received raw response from Gemini: {response}")
                            # --- END LOGGING ADDITIONS ---
                            # Defensive: check for .text or .parts
                            if hasattr(response, 'text') and response.text:
                                translated_text = response.text.strip()
                            elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                                translated_text = response.parts[0].text.strip()
                            else:
                                logger.error("Gemini response is empty or invalid.")
                                return ""
                            logger.info("Successfully received and parsed response from Gemini.")
                            translated_sentences.append(translated_text)
                            break
                        except asyncio.TimeoutError as e:
                            if attempt == 0:
                                logger.warning(f"⚠️ Translation timeout after {timeout}s, using original text. Payload: {repr(sentence)[:500]}")
                            else:
                                logger.warning(f"⚠️ Retry {attempt}: Translation timeout after {timeout}s. Payload: {repr(sentence)[:500]}")
                            if attempt >= max_retries:
                                logger.error(f"❌ Max retries reached for timeout. Returning original text. Payload: {repr(sentence)[:500]}")
                                translated_sentences.append(sentence)
                                break
                            time.sleep(2 ** attempt)
                            attempt += 1
                        except Exception as e:
                            logger.critical(f"FATAL: An unhandled exception occurred during Gemini API call: {e}")
                            if attempt >= max_retries:
                                logger.error(f"❌ Max retries reached for error. Returning original text. Payload: {repr(sentence)[:500]}")
                                translated_sentences.append("")
                                break
                            time.sleep(2 ** attempt)
                            attempt += 1
                return self._recombine_sentences(translated_sentences)
        except Exception as e:
            logger.error(f"Error translating text with Gemini: {e}. Full payload: {repr(text)[:2000]}")
            return ""
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences while preserving formatting"""
        # Split on sentence boundaries but keep the delimiters
        import re
        sentence_pattern = r'([.!?]+\s+)'
        sentences = re.split(sentence_pattern, text)
        
        # Recombine delimiters with their sentences
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        
        return result
    
    def _recombine_sentences(self, sentences: list) -> str:
        """Recombine sentences with proper spacing"""
        return ' '.join(sentences)
    
    def translate_text_sync(self, text: str, target_language: str) -> str:
        """Synchronous wrapper for translate_text"""
        try:
            return asyncio.run(self.translate_text(text, target_language))
        except RuntimeError:
            # If already in an event loop, use alternative
            return asyncio.get_event_loop().run_until_complete(
                self.translate_text(text, target_language)
            ) 