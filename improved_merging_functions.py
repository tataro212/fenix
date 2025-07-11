
def improved_is_merge_candidate_dt(prev_block, next_block):
    """
    Improved merging logic with better block type detection and spatial checks
    """
    import re
    
    # Expanded mergeable types to include all text-like content
    mergeable_types = {
        'paragraph', 'text', 'caption', 'list_item', 'bibliography', 
        'footnote', 'heading'  # Include headings that might be part of paragraphs
    }
    
    # Get block types and normalize
    prev_type = str(getattr(prev_block, 'block_type', '')).lower().replace('blocktype.', '')
    next_type = str(getattr(next_block, 'block_type', '')).lower().replace('blocktype.', '')
    
    self.logger.info(f"[MERGE CHECK] {getattr(prev_block, 'block_id', 'unknown')} ({prev_type}) + {getattr(next_block, 'block_id', 'unknown')} ({next_type})")
    
    # Check if both blocks are mergeable types
    if prev_type not in mergeable_types or next_type not in mergeable_types:
        self.logger.debug(f"[NOT MERGEABLE] {prev_type} → {next_type}")
        return False
    
    # Get text content with fallbacks
    def get_block_text(block):
        if hasattr(block, 'translated_text') and block.translated_text:
            return block.translated_text
        elif hasattr(block, 'original_text') and block.original_text:
            return block.original_text
        elif hasattr(block, 'get_display_text'):
            return str(block.get_display_text())
        return ''
    
    prev_text = get_block_text(prev_block).rstrip()
    next_text = get_block_text(next_block).lstrip()
    
    # Skip if either text is empty
    if not prev_text or not next_text:
        self.logger.debug(f"[NO MERGE] One of the texts is empty.")
        return False
    
    # Check for translation artifacts and skip them
    if 'μετάφραση εγγράφου' in prev_text or 'μετάφραση εγγράφου' in next_text:
        self.logger.debug(f"[NO MERGE] Translation artifact detected.")
        return False
    
    self.logger.info(f"[TEXTS] '{prev_text[-40:]}' → '{next_text[:40]}'")
    
    # SPATIAL PROXIMITY CHECK
    if hasattr(prev_block, 'bbox') and hasattr(next_block, 'bbox'):
        if prev_block.bbox and next_block.bbox:
            # Calculate vertical distance
            vertical_distance = abs(prev_block.bbox[1] - next_block.bbox[1])
            # Calculate horizontal alignment
            horizontal_diff = abs(prev_block.bbox[0] - next_block.bbox[0])
            
            # If blocks are too far apart vertically, don't merge
            if vertical_distance > 50:  # Increased threshold
                self.logger.debug(f"[NO MERGE] Vertical distance too large: {vertical_distance}")
                return False
            
            # If blocks are significantly misaligned horizontally, don't merge
            if horizontal_diff > 100:  # Allow some indentation
                self.logger.debug(f"[NO MERGE] Horizontal misalignment: {horizontal_diff}")
                return False
    
    # TEXT CONTENT MERGING RULES
    # 1. Hyphen at end of previous text
    if prev_text.endswith('-'):
        self.logger.info(f"[MERGE] {getattr(prev_block, 'block_id', 'unknown')} ends with hyphen")
        return True
    
    # 2. No strong punctuation + lowercase start
    if not re.search(r'[.!?;:…]$', prev_text) and next_text and next_text[0].islower():
        self.logger.info(f"[MERGE] {getattr(prev_block, 'block_id', 'unknown')} + {getattr(next_block, 'block_id', 'unknown')} - no punctuation + lowercase")
        return True
    
    # 3. Previous text ends with comma and next starts with lowercase
    if prev_text.endswith(',') and next_text and next_text[0].islower():
        self.logger.info(f"[MERGE] {getattr(prev_block, 'block_id', 'unknown')} + {getattr(next_block, 'block_id', 'unknown')} - comma + lowercase")
        return True
    
    # 4. Previous text ends with opening parenthesis and next starts with lowercase
    if prev_text.endswith('(') and next_text and next_text[0].islower():
        self.logger.info(f"[MERGE] {getattr(prev_block, 'block_id', 'unknown')} + {getattr(next_block, 'block_id', 'unknown')} - parenthesis + lowercase")
        return True
    
    # 5. Previous text ends with quote and next starts with lowercase
    if prev_text.endswith('"') and next_text and next_text[0].islower():
        self.logger.info(f"[MERGE] {getattr(prev_block, 'block_id', 'unknown')} + {getattr(next_block, 'block_id', 'unknown')} - quote + lowercase")
        return True
    
    self.logger.info(f"[NO MERGE] Pattern doesn't match for '{prev_text[-20:]}' + '{next_text[:20]}'")
    return False

def improved_merge_paragraph_fragments_dt(blocks):
    """
    Improved merging function with better error handling and logging
    """
    if not blocks:
        return []
    
    merged_blocks = []
    i = 0
    
    while i < len(blocks):
        current = blocks[i]
        j = i + 1
        
        # Try to merge consecutive blocks
        while j < len(blocks):
            self.logger.info(f"[LOOP] i={i}, j={j}, current={getattr(current, 'block_id', 'unknown')}, next={getattr(blocks[j], 'block_id', 'unknown')}")
            
            if improved_is_merge_candidate_dt(current, blocks[j]):
                next_block = blocks[j]
                self.logger.info(f"[MERGING] {getattr(current, 'block_id', 'unknown')} + {getattr(next_block, 'block_id', 'unknown')}")
                
                # Get text content
                def get_block_text(block):
                    if hasattr(block, 'translated_text') and block.translated_text:
                        return block.translated_text
                    elif hasattr(block, 'original_text') and block.original_text:
                        return block.original_text
                    elif hasattr(block, 'get_display_text'):
                        return str(block.get_display_text())
                    return ''
                
                current_text = get_block_text(current)
                next_text = get_block_text(next_block)
                
                # Merge text based on ending
                if current_text.rstrip().endswith('-'):
                    merged_text = current_text.rstrip()[:-1] + next_text.lstrip()
                else:
                    merged_text = current_text.rstrip() + ' ' + next_text.lstrip()
                
                # Update text fields
                if hasattr(current, 'translated_text'):
                    current.translated_text = merged_text
                if hasattr(current, 'original_text'):
                    current.original_text = merged_text
                
                # Expand bounding box
                if hasattr(current, 'bbox') and hasattr(next_block, 'bbox'):
                    if current.bbox and next_block.bbox:
                        current.bbox = [
                            min(current.bbox[0], next_block.bbox[0]),
                            min(current.bbox[1], next_block.bbox[1]),
                            max(current.bbox[2], next_block.bbox[2]),
                            max(current.bbox[3], next_block.bbox[3])
                        ]
                
                j += 1
            else:
                break
        
        merged_blocks.append(current)
        i = j
    
    return merged_blocks

def fix_block_type_classification(blocks):
    """
    Fix block type classification issues before merging
    """
    for block in blocks:
        if hasattr(block, 'block_type'):
            block_type = str(block.block_type).lower().replace('blocktype.', '')
            
            # Get text content
            text = ''
            if hasattr(block, 'translated_text') and block.translated_text:
                text = block.translated_text
            elif hasattr(block, 'original_text') and block.original_text:
                text = block.original_text
            elif hasattr(block, 'get_display_text'):
                text = str(block.get_display_text())
            
            # Fix empty title blocks
            if block_type == 'title' and not text.strip():
                # Check if this should be a different type
                if hasattr(block, 'bbox'):
                    # If it's an image placeholder, change to image
                    if hasattr(block, 'image_path') or 'image' in str(block).lower():
                        block.block_type = 'image'
                    else:
                        # Remove empty title blocks by setting text to empty
                        if hasattr(block, 'translated_text'):
                            block.translated_text = ''
                        if hasattr(block, 'original_text'):
                            block.original_text = ''
            
            # Fix paragraph blocks that look like headings
            if block_type == 'paragraph' and text.strip():
                if (text.strip().endswith('?') or text.strip().endswith(':')) and len(text.strip()) < 100:
                    # This looks like a heading
                    block.block_type = 'heading'
            
            # Fix text blocks that should be paragraphs
            if block_type == 'text' and text.strip() and len(text.strip()) > 50:
                # Long text blocks should be paragraphs
                block.block_type = 'paragraph'
    
    return blocks

def fix_translation_issues(blocks):
    """
    Fix translation-related issues
    """
    for block in blocks:
        original_text = ''
        translated_text = ''
        
        if hasattr(block, 'original_text'):
            original_text = block.original_text or ''
        if hasattr(block, 'translated_text'):
            translated_text = block.translated_text or ''
        
        # Remove translation artifacts
        if translated_text and 'μετάφραση εγγράφου' in translated_text:
            block.translated_text = ''
        
        # Use original text as fallback if translation is empty
        if not translated_text and original_text:
            block.translated_text = original_text
        
        # Clean up excessive whitespace
        if translated_text:
            block.translated_text = ' '.join(translated_text.split())
        if original_text:
            block.original_text = ' '.join(original_text.split())
    
    return blocks

def fix_spatial_ordering(blocks):
    """
    Fix spatial ordering of blocks for proper reading order
    """
    # Sort blocks by vertical position (y-coordinate) first
    blocks_with_bbox = [b for b in blocks if hasattr(b, 'bbox') and b.bbox]
    blocks_without_bbox = [b for b in blocks if not hasattr(b, 'bbox') or not b.bbox]
    
    # Sort by vertical position, then horizontal position
    blocks_with_bbox.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    
    # Combine sorted blocks with unsorted blocks
    return blocks_with_bbox + blocks_without_bbox
