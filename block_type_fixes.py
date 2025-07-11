
def fix_block_type_classification(blocks):
    """
    Fix block type classification issues
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
                        # Remove empty title blocks
                        continue
            
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
