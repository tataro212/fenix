
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
    
    # Check if both blocks are mergeable types
    if prev_type not in mergeable_types or next_type not in mergeable_types:
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
        return False
    
    # Check for translation artifacts and skip them
    if 'μετάφραση εγγράφου' in prev_text or 'μετάφραση εγγράφου' in next_text:
        return False
    
    # SPATIAL PROXIMITY CHECK
    if hasattr(prev_block, 'bbox') and hasattr(next_block, 'bbox'):
        if prev_block.bbox and next_block.bbox:
            # Calculate vertical distance
            vertical_distance = abs(prev_block.bbox[1] - next_block.bbox[1])
            # Calculate horizontal alignment
            horizontal_diff = abs(prev_block.bbox[0] - next_block.bbox[0])
            
            # If blocks are too far apart vertically, don't merge
            if vertical_distance > 50:  # Increased threshold
                return False
            
            # If blocks are significantly misaligned horizontally, don't merge
            if horizontal_diff > 100:  # Allow some indentation
                return False
    
    # TEXT CONTENT MERGING RULES
    # 1. Hyphen at end of previous text
    if prev_text.endswith('-'):
        return True
    
    # 2. No strong punctuation + lowercase start
    if not re.search(r'[.!?;:…]$', prev_text) and next_text and next_text[0].islower():
        return True
    
    # 3. Previous text ends with comma and next starts with lowercase
    if prev_text.endswith(',') and next_text and next_text[0].islower():
        return True
    
    # 4. Previous text ends with opening parenthesis and next starts with lowercase
    if prev_text.endswith('(') and next_text and next_text[0].islower():
        return True
    
    # 5. Previous text ends with quote and next starts with lowercase
    if prev_text.endswith('"') and next_text and next_text[0].islower():
        return True
    
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
            if improved_is_merge_candidate_dt(current, blocks[j]):
                next_block = blocks[j]
                
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
