#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Intelligent Content Batcher for PDF Translation Pipeline

This module implements content-flow-based batching that:
1. Removes page boundaries from batching logic
2. Groups content by type (text, lists, titles, etc.)
3. Implements 12,000 character limit with content type awareness
4. Provides semantic grouping for optimal translation
5. Enables parallel processing of batches
6. Advanced content type classification with pattern matching
7. Semantic coherence calculation
8. Translation priority assignment
9. Comprehensive performance tracking and reporting
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class ContentType(Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    UNKNOWN = "unknown"

@dataclass
class ContentItem:
    id: str
    text: str
    content_type: ContentType
    bbox: Tuple[int, int, int, int]
    confidence: float
    page_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        
    def is_translatable(self) -> bool:
        return self.content_type in [
            ContentType.PARAGRAPH, ContentType.HEADING, ContentType.LIST_ITEM, 
            ContentType.CAPTION, ContentType.FOOTNOTE
        ]

@dataclass
class ContentBatch:
    batch_id: str
    items: List[ContentItem]
    content_types: Set[ContentType]
    total_chars: int
    semantic_coherence: float
    translation_priority: int = 1
    
    def get_combined_text(self) -> str:
        texts = [item.text.strip() for item in self.items if item.is_translatable()]
        return "\n\n".join(texts)
    
    def get_item_ids(self) -> List[str]:
        return [item.id for item in self.items]

class ContentTypeClassifier:
    """Advanced content type classifier using pattern matching and heuristics"""
    
    def __init__(self):
        self.content_patterns = {
            ContentType.HEADING: [
                r'^[A-Z][A-Z\s]+$',  # ALL CAPS
                r'^\d+\.\s+[A-Z]',   # 1. Title
                r'^Chapter\s+\d+',   # Chapter 1
                r'^Section\s+\d+',   # Section 1
                r'^[IVX]+\.\s+',     # I. II. III.
                r'^\d+\.\d+\s+',     # 1.1 1.2
                r'^[A-Z][a-z]+\s*:', # Title:
            ],
            ContentType.LIST_ITEM: [
                r'^[\•\-\*]\s+',     # • - *
                r'^\d+\.\s+',        # 1. 2.
                r'^[a-z]\)\s+',      # a) b)
                r'^[A-Z]\)\s+',      # A) B)
                r'^[ivx]+\)\s+',     # i) ii)
            ],
            ContentType.CAPTION: [
                r'^Figure\s+\d+',    # Figure 1
                r'^Fig\.\s+\d+',     # Fig. 1
                r'^Table\s+\d+',     # Table 1
                r'^Image\s+\d+',     # Image 1
                r'^Photo\s+\d+',     # Photo 1
            ],
            ContentType.FOOTNOTE: [
                r'^\d+\s+',          # 1 
                r'^\*+\s+',          # ***
                r'^[a-z]+\s+',       # a b c
            ],
            ContentType.CODE: [
                r'[{}()\[\]]',       # Code brackets
                r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(',  # function()
                r'[a-zA-Z_][a-zA-Z0-9_]*\s*=',   # variable =
                r'import\s+',        # import
                r'def\s+',           # def
                r'class\s+',         # class
                r'if\s+',            # if
                r'for\s+',           # for
                r'while\s+',         # while
            ]
        }
        
        # Patterns for non-translatable content
        self.non_translatable_patterns = [
            r'^[0-9\s\.\-]+$',      # Numbers only
            r'^[A-Z\s]+$',          # ALL CAPS only
            r'^[a-z]+\.[a-z]+$',    # file.ext
            r'^[A-Z]{2,}$',         # ACRONYMS
            r'^\d+$',               # Just numbers
            r'^[^\w\s]+$',          # Only symbols
            r'^[a-zA-Z0-9_]+$',     # Code identifiers
        ]
    
    def classify_content(self, text: str, label: str = "text") -> ContentType:
        """Classify content based on text patterns and label"""
        text = text.strip()
        
        # Check for non-translatable content first
        for pattern in self.non_translatable_patterns:
            if re.match(pattern, text):
                return ContentType.UNKNOWN
        
        # Check content patterns
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return content_type
        
        # Fallback to label-based classification
        label_mapping = {
            'title': ContentType.HEADING,
            'heading': ContentType.HEADING,
            'paragraph': ContentType.PARAGRAPH,
            'list': ContentType.LIST_ITEM,
            'caption': ContentType.CAPTION,
            'footnote': ContentType.FOOTNOTE,
            'table': ContentType.TABLE,
            'image': ContentType.IMAGE,
            'code': ContentType.CODE,
        }
        
        return label_mapping.get(label.lower(), ContentType.PARAGRAPH)

class IntelligentContentBatcher:
    """
    Enhanced intelligent content batcher with semantic coherence,
    content type compatibility, and comprehensive reporting.
    """
    
    def __init__(self, max_batch_chars: int = 12000, min_batch_chars: int = 100):
        self.max_batch_chars = max_batch_chars
        self.min_batch_chars = min_batch_chars
        self.classifier = ContentTypeClassifier()
        
        # Performance tracking
        self.batching_stats = {
            'total_items': 0,
            'translatable_items': 0,
            'batches_created': 0,
            'average_batch_size': 0,
            'content_type_distribution': defaultdict(int),
            'batching_time': 0,
            'semantic_coherence_avg': 0.0,
            'character_utilization_avg': 0.0
        }
        
        logger.info(f"🚀 Enhanced IntelligentContentBatcher initialized with {max_batch_chars} char limit")
    
    def create_content_items(self, mapped_content: Dict[str, Any]) -> List[ContentItem]:
        """Create content items from PyMuPDF-YOLO mapping with advanced classification"""
        content_items = []
        
        for area_id, area_data in mapped_content.items():
            # Extract data from different possible formats
            if hasattr(area_data, 'combined_text'):
                text = area_data.combined_text
                label = getattr(area_data.layout_info, 'label', 'text')
                bbox = getattr(area_data.layout_info, 'bbox', (0, 0, 0, 0))
                confidence = getattr(area_data.layout_info, 'confidence', 0.5)
                page_num = getattr(area_data, 'page_num', 0)
            else:
                text = area_data.get('combined_text', '')
                label = area_data.get('layout_info', {}).get('label', 'text')
                bbox = area_data.get('layout_info', {}).get('bbox', (0, 0, 0, 0))
                confidence = area_data.get('layout_info', {}).get('confidence', 0.5)
                page_num = area_data.get('page_num', 0)
            
            if not text.strip():
                continue
            
            # Use advanced classification
            content_type = self.classifier.classify_content(text, label)
            
            item = ContentItem(
                id=area_id,
                text=text,
                content_type=content_type,
                bbox=bbox,
                confidence=confidence,
                page_num=page_num,
                metadata={
                    'original_label': label, 
                    'classification_confidence': confidence,
                    'classification_method': 'pattern_based'
                }
            )
            
            content_items.append(item)
            self.batching_stats['content_type_distribution'][content_type.value] += 1
        
        self.batching_stats['total_items'] = len(content_items)
        self.batching_stats['translatable_items'] = len([item for item in content_items if item.is_translatable()])
        
        logger.info(f"📝 Created {len(content_items)} content items")
        logger.info(f"   Translatable: {self.batching_stats['translatable_items']}")
        logger.info(f"   Content types: {dict(self.batching_stats['content_type_distribution'])}")
        
        return content_items
    
    def create_intelligent_batches(self, content_items: List[ContentItem]) -> List[ContentBatch]:
        """Create intelligent batches with semantic coherence and content type compatibility"""
        start_time = time.time()
        
        translatable_items = [item for item in content_items if item.is_translatable()]
        
        if not translatable_items:
            logger.warning("⚠️ No translatable items found")
            return []
        
        # Sort by reading order (page, y-position, x-position)
        sorted_items = sorted(translatable_items, key=lambda item: (item.page_num, item.bbox[1], item.bbox[0]))
        
        batches = []
        current_batch = []
        current_chars = 0
        current_types = set()
        
        for item in sorted_items:
            item_chars = item.char_count
            
            # Check if adding this item would exceed the limit
            if current_chars + item_chars > self.max_batch_chars and current_batch:
                batch = self._create_batch(current_batch, len(batches))
                batches.append(batch)
                
                current_batch = [item]
                current_chars = item_chars
                current_types = {item.content_type}
            else:
                # Check content type compatibility
                if self._is_content_type_compatible(current_types, item.content_type):
                    current_batch.append(item)
                    current_chars += item_chars
                    current_types.add(item.content_type)
                else:
                    # Start new batch for incompatible content type
                    if current_batch:
                        batch = self._create_batch(current_batch, len(batches))
                        batches.append(batch)
                    
                    current_batch = [item]
                    current_chars = item_chars
                    current_types = {item.content_type}
        
        # Create final batch
        if current_batch:
            batch = self._create_batch(current_batch, len(batches))
            batches.append(batch)
        
        # Update statistics
        self._update_batching_statistics(batches, time.time() - start_time)
        
        logger.info(f"📦 Created {len(batches)} intelligent batches:")
        for i, batch in enumerate(batches):
            logger.info(f"   Batch {i+1}: {len(batch.items)} items, {batch.total_chars} chars, "
                       f"types: {[t.value for t in batch.content_types]}, "
                       f"coherence: {batch.semantic_coherence:.3f}")
        
        return batches
    
    def _is_content_type_compatible(self, current_types: Set[ContentType], new_type: ContentType) -> bool:
        """Check if new content type is compatible with current batch types"""
        if not current_types:
            return True
        
        # Define compatibility rules
        compatibility_rules = {
            ContentType.PARAGRAPH: {ContentType.PARAGRAPH, ContentType.HEADING, ContentType.FOOTNOTE},
            ContentType.HEADING: {ContentType.PARAGRAPH, ContentType.HEADING},
            ContentType.LIST_ITEM: {ContentType.LIST_ITEM, ContentType.HEADING},
            ContentType.CAPTION: {ContentType.CAPTION},
            ContentType.FOOTNOTE: {ContentType.FOOTNOTE, ContentType.PARAGRAPH},
        }
        
        # Check compatibility with all current types
        for current_type in current_types:
            if current_type not in compatibility_rules:
                continue
            if new_type not in compatibility_rules[current_type]:
                return False
        
        return True
    
    def _create_batch(self, items: List[ContentItem], batch_index: int) -> ContentBatch:
        """Create a content batch with semantic coherence and priority calculation"""
        total_chars = sum(item.char_count for item in items)
        content_types = {item.content_type for item in items}
        
        # Calculate semantic coherence
        coherence = self._calculate_semantic_coherence(items)
        
        # Calculate translation priority
        priority = self._calculate_translation_priority(items)
        
        return ContentBatch(
            batch_id=f"batch_{batch_index}",
            items=items,
            content_types=content_types,
            total_chars=total_chars,
            semantic_coherence=coherence,
            translation_priority=priority
        )
    
    def _calculate_semantic_coherence(self, items: List[ContentItem]) -> float:
        """Calculate semantic coherence of items in a batch"""
        if len(items) <= 1:
            return 1.0
        
        # Content type coherence (fewer types = higher coherence)
        content_types = {item.content_type for item in items}
        type_coherence = 1.0 / len(content_types)
        
        # Spatial coherence (closer items = higher coherence)
        spatial_coherence = self._calculate_spatial_coherence(items)
        
        # Text length coherence (similar lengths = higher coherence)
        lengths = [item.char_count for item in items]
        length_variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
        length_coherence = max(0, 1 - (length_variance / 10000))  # Normalize
        
        # Weighted combination
        coherence = (type_coherence * 0.4 + spatial_coherence * 0.4 + length_coherence * 0.2)
        
        return coherence
    
    def _calculate_spatial_coherence(self, items: List[ContentItem]) -> float:
        """Calculate spatial coherence based on item positions"""
        if len(items) <= 1:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(items) - 1):
            item1 = items[i]
            item2 = items[i + 1]
            
            # Calculate center points
            x1, y1, x2, y2 = item1.bbox
            x3, y3, x4, y4 = item2.bbox
            
            center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
            
            # Euclidean distance
            distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            total_distance += distance
            count += 1
        
        if count == 0:
            return 1.0
        
        avg_distance = total_distance / count
        
        # Normalize: closer items = higher coherence
        max_reasonable_distance = 500  # pixels
        coherence = max(0, 1 - (avg_distance / max_reasonable_distance))
        
        return coherence
    
    def _calculate_translation_priority(self, items: List[ContentItem]) -> int:
        """Calculate translation priority for a batch (1=highest, 3=lowest)"""
        priority_scores = {
            ContentType.HEADING: 3,      # High priority
            ContentType.PARAGRAPH: 2,    # Medium priority
            ContentType.LIST_ITEM: 2,    # Medium priority
            ContentType.CAPTION: 1,      # Low priority
            ContentType.FOOTNOTE: 1,     # Low priority
        }
        
        total_score = sum(priority_scores.get(item.content_type, 1) for item in items)
        avg_score = total_score / len(items)
        
        if avg_score >= 2.5:
            return 1  # High priority
        elif avg_score >= 1.5:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    
    def _update_batching_statistics(self, batches: List[ContentBatch], batching_time: float):
        """Update comprehensive batching statistics"""
        if not batches:
            return
        
        total_chars = sum(batch.total_chars for batch in batches)
        total_coherence = sum(batch.semantic_coherence for batch in batches)
        
        self.batching_stats.update({
            'batches_created': len(batches),
            'average_batch_size': sum(len(batch.items) for batch in batches) / len(batches),
            'batching_time': batching_time,
            'semantic_coherence_avg': total_coherence / len(batches),
            'character_utilization_avg': (total_chars / len(batches)) / self.max_batch_chars
        })
    
    def get_batching_report(self) -> Dict[str, Any]:
        """Generate comprehensive batching report"""
        return {
            'statistics': self.batching_stats.copy(),
            'efficiency_metrics': {
                'items_per_batch': (
                    self.batching_stats['translatable_items'] 
                    / max(self.batching_stats['batches_created'], 1)
                ),
                'character_utilization': (
                    self.batching_stats['character_utilization_avg'] * 100
                ),
                'content_type_efficiency': dict(self.batching_stats['content_type_distribution']),
                'semantic_coherence_avg': self.batching_stats['semantic_coherence_avg']
            },
            'performance': {
                'batching_time_seconds': self.batching_stats['batching_time'],
                'items_per_second': (
                    self.batching_stats['total_items'] 
                    / max(self.batching_stats['batching_time'], 0.001)
                )
            },
            'quality_metrics': {
                'average_semantic_coherence': self.batching_stats['semantic_coherence_avg'],
                'content_type_distribution': dict(self.batching_stats['content_type_distribution']),
                'translatable_item_ratio': (
                    self.batching_stats['translatable_items'] 
                    / max(self.batching_stats['total_items'], 1)
                )
            }
        }

def create_intelligent_content_batcher(max_batch_chars: int = 12000) -> IntelligentContentBatcher:
    """Factory function to create an intelligent content batcher"""
    return IntelligentContentBatcher(max_batch_chars=max_batch_chars) 