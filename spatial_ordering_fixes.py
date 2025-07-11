
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

def detect_reading_order(blocks):
    """
    Detect proper reading order for complex layouts
    """
    if not blocks:
        return blocks
    
    # Group blocks by approximate vertical position (within 20 pixels)
    vertical_groups = []
    current_group = []
    current_y = None
    
    for block in blocks:
        if hasattr(block, 'bbox') and block.bbox:
            y_pos = block.bbox[1]
            
            if current_y is None or abs(y_pos - current_y) <= 20:
                current_group.append(block)
                current_y = y_pos
            else:
                if current_group:
                    vertical_groups.append(current_group)
                current_group = [block]
                current_y = y_pos
        else:
            current_group.append(block)
    
    if current_group:
        vertical_groups.append(current_group)
    
    # Sort each group by horizontal position (left to right)
    for group in vertical_groups:
        group.sort(key=lambda b: b.bbox[0] if hasattr(b, 'bbox') and b.bbox else 0)
    
    # Flatten groups back to list
    return [block for group in vertical_groups for block in group]
