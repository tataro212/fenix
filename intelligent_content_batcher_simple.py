#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Intelligent Content Batcher for PDF Translation Pipeline
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class ContentType(Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    UNKNOWN = "unknown"

@dataclass
class ContentItem:
    id: str
    text: str
    content_type: ContentType
    bbox: Tuple[int, int, int, int]
    confidence: float
    page_num: int
    
    def __post_init__(self):
        self.char_count = len(self.text)
        
    def is_translatable(self) -> bool:
        return self.content_type in [
            ContentType.PARAGRAPH, ContentType.HEADING, ContentType.LIST_ITEM, 
            ContentType.CAPTION
        ]

@dataclass
class ContentBatch:
    batch_id: str
    items: List[ContentItem]
    content_types: Set[ContentType]
    total_chars: int
    
    def get_combined_text(self) -> str:
        texts = [item.text.strip() for item in self.items if item.is_translatable()]
        return "\n\n".join(texts)

class IntelligentContentBatcher:
    def __init__(self, max_batch_chars: int = 12000):
        self.max_batch_chars = max_batch_chars
        logger.info(f"IntelligentContentBatcher initialized with {max_batch_chars} char limit")
    
    def create_content_items(self, mapped_content: Dict[str, Any]) -> List[ContentItem]:
        content_items = []
        
        for area_id, area_data in mapped_content.items():
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
            
            # Simple content type classification
            if label.lower() in ['title', 'heading']:
                content_type = ContentType.HEADING
            elif label.lower() in ['list']:
                content_type = ContentType.LIST_ITEM
            elif label.lower() in ['caption']:
                content_type = ContentType.CAPTION
            else:
                content_type = ContentType.PARAGRAPH
            
            item = ContentItem(
                id=area_id,
                text=text,
                content_type=content_type,
                bbox=bbox,
                confidence=confidence,
                page_num=page_num
            )
            
            content_items.append(item)
        
        logger.info(f"Created {len(content_items)} content items")
        return content_items
    
    def create_intelligent_batches(self, content_items: List[ContentItem]) -> List[ContentBatch]:
        translatable_items = [item for item in content_items if item.is_translatable()]
        
        if not translatable_items:
            logger.warning("No translatable items found")
            return []
        
        batches = []
        current_batch = []
        current_chars = 0
        
        for item in translatable_items:
            item_chars = item.char_count
            
            if current_chars + item_chars > self.max_batch_chars and current_batch:
                batch = self._create_batch(current_batch, len(batches))
                batches.append(batch)
                
                current_batch = [item]
                current_chars = item_chars
            else:
                current_batch.append(item)
                current_chars += item_chars
        
        if current_batch:
            batch = self._create_batch(current_batch, len(batches))
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} intelligent batches")
        return batches
    
    def _create_batch(self, items: List[ContentItem], batch_index: int) -> ContentBatch:
        total_chars = sum(item.char_count for item in items)
        content_types = {item.content_type for item in items}
        
        return ContentBatch(
            batch_id=f"batch_{batch_index}",
            items=items,
            content_types=content_types,
            total_chars=total_chars
        )
    
    def get_batching_report(self) -> Dict[str, Any]:
        return {
            'statistics': {'total_items': 0, 'batches_created': 0},
            'efficiency_metrics': {
                'items_per_batch': 0.0,
                'character_utilization': 0.0
            }
        } 