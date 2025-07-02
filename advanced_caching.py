"""
Advanced Contextual Caching System for Ultimate PDF Translator

Implements fuzzy matching, context-aware caching, and similarity-based retrieval
to maximize cache hit rates and reduce API calls.
"""

import json
import os
import hashlib
import logging
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config_manager import config_manager

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with context and metadata"""
    original_text: str
    translated_text: str
    target_language: str
    model_name: str
    context_hash: str
    similarity_hash: str
    timestamp: float
    usage_count: int = 0
    quality_score: float = 1.0

class ContextualCacheManager:
    """
    Advanced caching system with context awareness and fuzzy matching
    for improved cache hit rates and translation consistency.
    """
    
    def __init__(self):
        self.settings = config_manager.translation_enhancement_settings
        self.cache_file = self.settings['translation_cache_file_path']
        self.enabled = self.settings['use_translation_cache']
        
        # Cache configuration
        try:
            if hasattr(config_manager, 'get_value'):
                self.max_cache_size = config_manager.get_value('advanced_caching', 'max_cache_entries', 10000)
                self.similarity_threshold = config_manager.get_value('advanced_caching', 'similarity_threshold', 0.85)
                self.context_window_size = config_manager.get_value('advanced_caching', 'context_window_chars', 200)
                self.enable_fuzzy_matching = config_manager.get_value('advanced_caching', 'enable_fuzzy_matching', True)
            elif hasattr(config_manager, 'get_config_value'):
                self.max_cache_size = config_manager.get_config_value('AdvancedCaching', 'max_cache_entries', 10000, int)
                self.similarity_threshold = config_manager.get_config_value('AdvancedCaching', 'similarity_threshold', 0.85, float)
                self.context_window_size = config_manager.get_config_value('AdvancedCaching', 'context_window_chars', 200, int)
                self.enable_fuzzy_matching = config_manager.get_config_value('AdvancedCaching', 'enable_fuzzy_matching', True, bool)
            else:
                self.max_cache_size = 10000
                self.similarity_threshold = 0.85
                self.context_window_size = 200
                self.enable_fuzzy_matching = True
        except Exception as e:
            logger.warning(f"Could not get advanced caching config: {e}, using defaults")
            self.max_cache_size = 10000
            self.similarity_threshold = 0.85
            self.context_window_size = 200
            self.enable_fuzzy_matching = True
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_index: Dict[str, List[str]] = {}  # similarity_hash -> list of cache_keys
        
        if self.enabled:
            self.load_cache()
    
    def load_cache(self):
        """Load cache from file with enhanced structure"""
        if not self.enabled or not self.cache_file:
            return
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Convert to enhanced cache entries
                for key, data in cache_data.items():
                    if isinstance(data, str):
                        # Legacy format - convert
                        entry = CacheEntry(
                            original_text="",
                            translated_text=data,
                            target_language="unknown",
                            model_name="unknown",
                            context_hash="",
                            similarity_hash=self._generate_similarity_hash(""),
                            timestamp=0.0
                        )
                    else:
                        # New format
                        entry = CacheEntry(**data)
                    
                    self.cache[key] = entry
                    self._update_similarity_index(key, entry)
                
                logger.info(f"Loaded {len(self.cache)} cached translations with advanced features")
                
            except Exception as e:
                logger.error(f"Error loading advanced cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to file with enhanced structure"""
        if not self.enabled or not self.cache_file:
            return
        
        try:
            # Convert cache entries to serializable format
            cache_data = {}
            for key, entry in self.cache.items():
                cache_data[key] = {
                    'original_text': entry.original_text,
                    'translated_text': entry.translated_text,
                    'target_language': entry.target_language,
                    'model_name': entry.model_name,
                    'context_hash': entry.context_hash,
                    'similarity_hash': entry.similarity_hash,
                    'timestamp': entry.timestamp,
                    'usage_count': entry.usage_count,
                    'quality_score': entry.quality_score
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.cache)} cached translations with metadata")
            
        except Exception as e:
            logger.error(f"Error saving advanced cache: {e}")
    
    def get_cached_translation(self, text: str, target_language: str, model_name: str, 
                             prev_context: str = "", next_context: str = "") -> Optional[str]:
        """Get cached translation with context awareness and fuzzy matching"""
        if not self.enabled:
            return None
        
        # Generate cache key with context
        cache_key = self._generate_contextual_cache_key(text, target_language, model_name, prev_context, next_context)
        
        # Direct cache hit
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.usage_count += 1
            logger.debug(f"Direct cache hit for text: {text[:50]}...")
            return entry.translated_text
        
        # Fuzzy matching if enabled
        if self.enable_fuzzy_matching:
            fuzzy_result = self._find_fuzzy_match(text, target_language, model_name, prev_context, next_context)
            if fuzzy_result:
                logger.debug(f"Fuzzy cache hit for text: {text[:50]}...")
                return fuzzy_result
        
        return None
    
    def cache_translation(self, text: str, target_language: str, model_name: str, 
                         translation: str, prev_context: str = "", next_context: str = "",
                         quality_score: float = 1.0):
        """Cache translation with enhanced metadata"""
        if not self.enabled:
            return
        
        import time
        
        # Generate cache key and entry
        cache_key = self._generate_contextual_cache_key(text, target_language, model_name, prev_context, next_context)
        context_hash = self._generate_context_hash(prev_context, next_context)
        similarity_hash = self._generate_similarity_hash(text)
        
        entry = CacheEntry(
            original_text=text,
            translated_text=translation,
            target_language=target_language,
            model_name=model_name,
            context_hash=context_hash,
            similarity_hash=similarity_hash,
            timestamp=time.time(),
            usage_count=1,
            quality_score=quality_score
        )
        
        # Add to cache
        self.cache[cache_key] = entry
        self._update_similarity_index(cache_key, entry)
        
        # Manage cache size
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
    
    def _generate_contextual_cache_key(self, text: str, target_language: str, model_name: str,
                                     prev_context: str, next_context: str) -> str:
        """Generate cache key that includes context information"""
        # Create context snippet
        context_snippet = (prev_context[-self.context_window_size:] + 
                          next_context[:self.context_window_size])
        
        # Generate hash
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        hasher.update(target_language.encode('utf-8'))
        hasher.update(model_name.encode('utf-8'))
        hasher.update(context_snippet.encode('utf-8'))
        
        return hasher.hexdigest()
    
    def _generate_context_hash(self, prev_context: str, next_context: str) -> str:
        """Generate hash for context information"""
        context_snippet = (prev_context[-self.context_window_size:] + 
                          next_context[:self.context_window_size])
        
        hasher = hashlib.md5()
        hasher.update(context_snippet.encode('utf-8'))
        return hasher.hexdigest()
    
    def _generate_similarity_hash(self, text: str) -> str:
        """Generate hash for similarity matching"""
        # Normalize text for similarity matching
        normalized = ' '.join(text.lower().split())
        
        hasher = hashlib.md5()
        hasher.update(normalized.encode('utf-8'))
        return hasher.hexdigest()
    
    def _update_similarity_index(self, cache_key: str, entry: CacheEntry):
        """Update similarity index for fuzzy matching"""
        similarity_hash = entry.similarity_hash
        
        if similarity_hash not in self.similarity_index:
            self.similarity_index[similarity_hash] = []
        
        if cache_key not in self.similarity_index[similarity_hash]:
            self.similarity_index[similarity_hash].append(cache_key)
    
    def _find_fuzzy_match(self, text: str, target_language: str, model_name: str,
                         prev_context: str, next_context: str) -> Optional[str]:
        """Find fuzzy match using similarity algorithms"""
        
        # Generate similarity hash for the input
        input_similarity_hash = self._generate_similarity_hash(text)
        
        # Check for exact similarity hash match first
        if input_similarity_hash in self.similarity_index:
            for cache_key in self.similarity_index[input_similarity_hash]:
                entry = self.cache[cache_key]
                if (entry.target_language == target_language and 
                    entry.model_name == model_name):
                    entry.usage_count += 1
                    return entry.translated_text
        
        # Fuzzy text matching
        best_match = None
        best_similarity = 0.0
        
        normalized_input = ' '.join(text.lower().split())
        
        for cache_key, entry in self.cache.items():
            if (entry.target_language != target_language or 
                entry.model_name != model_name):
                continue
            
            # Calculate text similarity
            normalized_cached = ' '.join(entry.original_text.lower().split())
            similarity = difflib.SequenceMatcher(None, normalized_input, normalized_cached).ratio()
            
            # Context similarity bonus
            if prev_context or next_context:
                input_context_hash = self._generate_context_hash(prev_context, next_context)
                if input_context_hash == entry.context_hash:
                    similarity += 0.1  # Bonus for matching context
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.usage_count += 1
            logger.debug(f"Fuzzy match found with similarity: {best_similarity:.2f}")
            return best_match.translated_text
        
        return None
    
    def _cleanup_cache(self):
        """Clean up cache by removing least used entries"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort by usage count and timestamp (least used and oldest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].usage_count, x[1].timestamp)
        )
        
        # Remove 20% of entries
        entries_to_remove = int(len(self.cache) * 0.2)
        
        for i in range(entries_to_remove):
            cache_key, entry = sorted_entries[i]
            
            # Remove from main cache
            del self.cache[cache_key]
            
            # Remove from similarity index
            similarity_hash = entry.similarity_hash
            if similarity_hash in self.similarity_index:
                if cache_key in self.similarity_index[similarity_hash]:
                    self.similarity_index[similarity_hash].remove(cache_key)
                
                # Clean up empty similarity hash entries
                if not self.similarity_index[similarity_hash]:
                    del self.similarity_index[similarity_hash]
        
        logger.info(f"Cache cleanup: removed {entries_to_remove} entries, {len(self.cache)} remaining")
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""
        if not self.cache:
            return {'total_entries': 0}
        
        total_usage = sum(entry.usage_count for entry in self.cache.values())
        avg_quality = sum(entry.quality_score for entry in self.cache.values()) / len(self.cache)
        
        # Language distribution
        language_dist = {}
        for entry in self.cache.values():
            lang = entry.target_language
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        return {
            'total_entries': len(self.cache),
            'total_usage': total_usage,
            'average_quality_score': avg_quality,
            'similarity_index_size': len(self.similarity_index),
            'language_distribution': language_dist,
            'fuzzy_matching_enabled': self.enable_fuzzy_matching,
            'similarity_threshold': self.similarity_threshold
        }

# Global advanced cache manager instance
advanced_cache_manager = ContextualCacheManager()
