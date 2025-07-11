
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

def validate_translations(blocks):
    """
    Validate translation quality and completeness
    """
    issues = []
    
    for block in blocks:
        original_text = getattr(block, 'original_text', '')
        translated_text = getattr(block, 'translated_text', '')
        
        # Check for empty translations
        if original_text and not translated_text:
            issues.append({
                'type': 'empty_translation',
                'block_id': getattr(block, 'block_id', 'unknown'),
                'original': original_text[:100]
            })
        
        # Check for translation artifacts
        if translated_text and 'μετάφραση εγγράφου' in translated_text:
            issues.append({
                'type': 'translation_artifact',
                'block_id': getattr(block, 'block_id', 'unknown'),
                'translated': translated_text[:100]
            })
    
    return issues
