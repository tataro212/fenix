"""
Semantic Caching using Vector Embeddings

This module provides semantic caching capabilities that can find and reuse
translations for text chunks that are semantically similar, even if not identical.
"""

import os
import json
import logging
import hashlib
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SemanticCacheEntry:
    """Entry in the semantic cache"""
    original_text: str
    translated_text: str
    target_language: str
    model_name: str
    embedding: np.ndarray
    timestamp: float
    usage_count: int
    similarity_threshold: float
    context_hash: str
    quality_score: float = 1.0

class SemanticCache:
    """
    Semantic cache that uses vector embeddings to find similar translations.
    Extends traditional caching with similarity-based retrieval.
    """
    
    def __init__(self, cache_dir: str = "semantic_cache", 
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 10000,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory to store cache files
            similarity_threshold: Minimum similarity score for cache hits
            max_cache_size: Maximum number of entries to keep in cache
            embedding_model: Sentence transformer model for embeddings
        """
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.embedding_model_name = embedding_model
        
        # Initialize cache storage
        self.cache: Dict[str, SemanticCacheEntry] = {}
        self.embeddings_index: np.ndarray = None
        self.cache_keys: List[str] = []
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'cache_size': 0,
            'avg_similarity_score': 0.0
        }
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"🧠 Semantic cache initialized")
        logger.info(f"   Cache directory: {cache_dir}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Embedding model: {embedding_model}")
        logger.info(f"   Loaded entries: {len(self.cache)}")
    
    def _init_embedding_model(self):
        """Initialize the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_available = True
            logger.info(f"✅ Embedding model loaded: {self.embedding_model_name}")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, semantic caching disabled")
            self.embedding_model = None
            self.embedding_available = False
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
            self.embedding_available = False
    
    def get_cached_translation(self, text: str, target_language: str, 
                             model_name: str, context: str = "") -> Optional[str]:
        """
        Get cached translation using semantic similarity.
        
        Args:
            text: Text to translate
            target_language: Target language
            model_name: Translation model name
            context: Additional context for caching
            
        Returns:
            Cached translation if found, None otherwise
        """
        self.stats['total_queries'] += 1
        
        if not self.embedding_available or not text.strip():
            return None
        
        # Generate cache key for exact match
        cache_key = self._generate_cache_key(text, target_language, model_name, context)
        
        # Check for exact match first
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.usage_count += 1
            self.stats['exact_hits'] += 1
            logger.debug(f"🎯 Exact cache hit for: {text[:50]}...")
            return entry.translated_text
        
        # Check for semantic similarity
        similar_entry = self._find_similar_entry(text, target_language, model_name, context)
        if similar_entry:
            similar_entry.usage_count += 1
            self.stats['semantic_hits'] += 1
            logger.debug(f"🔍 Semantic cache hit for: {text[:50]}...")
            return similar_entry.translated_text
        
        self.stats['misses'] += 1
        return None
    
    def cache_translation(self, text: str, target_language: str, model_name: str,
                         translation: str, context: str = "", quality_score: float = 1.0):
        """
        Cache a translation with semantic indexing.
        
        Args:
            text: Original text
            target_language: Target language
            model_name: Translation model name
            translation: Translated text
            context: Additional context
            quality_score: Quality score of the translation
        """
        if not self.embedding_available or not text.strip():
            return
        
        # Generate embedding for the text
        try:
            embedding = self.embedding_model.encode([text])[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return
        
        # Create cache entry
        cache_key = self._generate_cache_key(text, target_language, model_name, context)
        context_hash = self._generate_context_hash(context)
        
        entry = SemanticCacheEntry(
            original_text=text,
            translated_text=translation,
            target_language=target_language,
            model_name=model_name,
            embedding=embedding,
            timestamp=time.time(),
            usage_count=1,
            similarity_threshold=self.similarity_threshold,
            context_hash=context_hash,
            quality_score=quality_score
        )
        
        # Add to cache
        self.cache[cache_key] = entry
        self.cache_keys.append(cache_key)
        
        # Update embeddings index
        self._update_embeddings_index()
        
        # Manage cache size
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
        
        self.stats['cache_size'] = len(self.cache)
        
        # Periodically save cache
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def _find_similar_entry(self, text: str, target_language: str, 
                          model_name: str, context: str) -> Optional[SemanticCacheEntry]:
        """Find semantically similar cache entry"""
        if not self.cache or self.embeddings_index is None:
            return None
        
        try:
            # Generate embedding for query text
            query_embedding = self.embedding_model.encode([text])[0]
            
            # Calculate similarities with all cached embeddings
            similarities = np.dot(self.embeddings_index, query_embedding)
            
            # Find best matches above threshold
            best_indices = np.where(similarities >= self.similarity_threshold)[0]
            
            if len(best_indices) == 0:
                return None
            
            # Sort by similarity score
            sorted_indices = best_indices[np.argsort(similarities[best_indices])[::-1]]
            
            # Find best match with same language and model
            context_hash = self._generate_context_hash(context)
            
            for idx in sorted_indices:
                cache_key = self.cache_keys[idx]
                entry = self.cache[cache_key]
                
                # Check language and model compatibility
                if (entry.target_language == target_language and 
                    entry.model_name == model_name):
                    
                    # Update average similarity score for stats
                    self.stats['avg_similarity_score'] = (
                        (self.stats['avg_similarity_score'] * self.stats['semantic_hits'] + 
                         similarities[idx]) / (self.stats['semantic_hits'] + 1)
                    )
                    
                    logger.debug(f"Found similar entry with similarity: {similarities[idx]:.3f}")
                    return entry
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar entry: {e}")
            return None
    
    def _update_embeddings_index(self):
        """Update the embeddings index for similarity search"""
        if not self.cache:
            self.embeddings_index = None
            return
        
        try:
            embeddings = []
            self.cache_keys = []
            
            for cache_key, entry in self.cache.items():
                embeddings.append(entry.embedding)
                self.cache_keys.append(cache_key)
            
            self.embeddings_index = np.array(embeddings)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.embeddings_index, axis=1, keepdims=True)
            self.embeddings_index = self.embeddings_index / norms
            
        except Exception as e:
            logger.error(f"Error updating embeddings index: {e}")
            self.embeddings_index = None
    
    def _generate_cache_key(self, text: str, target_language: str, 
                          model_name: str, context: str) -> str:
        """Generate cache key for exact matching"""
        content = f"{text}|{target_language}|{model_name}|{context}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _generate_context_hash(self, context: str) -> str:
        """Generate hash for context"""
        return hashlib.md5(context.encode('utf-8')).hexdigest()[:8]
    
    def _cleanup_cache(self):
        """Remove least recently used entries to manage cache size"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort entries by usage count and timestamp
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].usage_count, x[1].timestamp)
        )
        
        # Remove oldest, least used entries
        entries_to_remove = len(self.cache) - self.max_cache_size + 100  # Remove extra for buffer
        
        for i in range(entries_to_remove):
            cache_key, _ = sorted_entries[i]
            del self.cache[cache_key]
        
        # Update embeddings index
        self._update_embeddings_index()
        
        logger.info(f"🧹 Cache cleanup: removed {entries_to_remove} entries")
    
    def _save_cache(self):
        """Save cache to disk"""
        if not self.cache:
            return
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Save cache entries (without embeddings for JSON compatibility)
            cache_data = {}
            for key, entry in self.cache.items():
                cache_data[key] = {
                    'original_text': entry.original_text,
                    'translated_text': entry.translated_text,
                    'target_language': entry.target_language,
                    'model_name': entry.model_name,
                    'timestamp': entry.timestamp,
                    'usage_count': entry.usage_count,
                    'similarity_threshold': entry.similarity_threshold,
                    'context_hash': entry.context_hash,
                    'quality_score': entry.quality_score
                }
            
            cache_file = os.path.join(self.cache_dir, 'semantic_cache.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Save embeddings separately
            embeddings_file = os.path.join(self.cache_dir, 'embeddings.pkl')
            embeddings_data = {
                'embeddings': {key: entry.embedding for key, entry in self.cache.items()},
                'cache_keys': self.cache_keys,
                'embeddings_index': self.embeddings_index
            }
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            logger.debug(f"💾 Semantic cache saved: {len(self.cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save semantic cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, 'semantic_cache.json')
            embeddings_file = os.path.join(self.cache_dir, 'embeddings.pkl')
            
            if not (os.path.exists(cache_file) and os.path.exists(embeddings_file)):
                return
            
            # Load cache data
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Load embeddings
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            # Reconstruct cache entries
            for key, data in cache_data.items():
                if key in embeddings_data['embeddings']:
                    entry = SemanticCacheEntry(
                        original_text=data['original_text'],
                        translated_text=data['translated_text'],
                        target_language=data['target_language'],
                        model_name=data['model_name'],
                        embedding=embeddings_data['embeddings'][key],
                        timestamp=data['timestamp'],
                        usage_count=data['usage_count'],
                        similarity_threshold=data['similarity_threshold'],
                        context_hash=data['context_hash'],
                        quality_score=data.get('quality_score', 1.0)
                    )
                    self.cache[key] = entry
            
            # Restore index
            self.cache_keys = embeddings_data.get('cache_keys', [])
            self.embeddings_index = embeddings_data.get('embeddings_index')
            
            self.stats['cache_size'] = len(self.cache)
            
            logger.info(f"📂 Semantic cache loaded: {len(self.cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load semantic cache: {e}")
            self.cache = {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_queries = self.stats['total_queries']
        if total_queries == 0:
            hit_rate = 0.0
        else:
            hit_rate = (self.stats['exact_hits'] + self.stats['semantic_hits']) / total_queries
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'semantic_hit_rate': self.stats['semantic_hits'] / total_queries if total_queries > 0 else 0.0,
            'embedding_model': self.embedding_model_name,
            'similarity_threshold': self.similarity_threshold,
            'max_cache_size': self.max_cache_size
        }
    
    def clear_cache(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.embeddings_index = None
        self.cache_keys = []
        self.stats = {
            'total_queries': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'cache_size': 0,
            'avg_similarity_score': 0.0
        }
        logger.info("🗑️ Semantic cache cleared")
    
    def __del__(self):
        """Save cache when object is destroyed"""
        if hasattr(self, 'cache') and self.cache:
            self._save_cache()
