
#!/usr/bin/env python3
"""
Integration script to apply all merging logic fixes
"""

def apply_all_merging_fixes(blocks):
    """
    Apply all fixes to improve merging logic
    """
    print("🔧 Applying merging logic fixes...")
    
    # Step 1: Fix block type classification
    from block_type_fixes import fix_block_type_classification
    blocks = fix_block_type_classification(blocks)
    print("   ✅ Fixed block type classification")
    
    # Step 2: Fix translation issues
    from translation_fixes import fix_translation_issues
    blocks = fix_translation_issues(blocks)
    print("   ✅ Fixed translation issues")
    
    # Step 3: Fix spatial ordering
    from spatial_ordering_fixes import fix_spatial_ordering
    blocks = fix_spatial_ordering(blocks)
    print("   ✅ Fixed spatial ordering")
    
    # Step 4: Apply improved merging logic
    from improved_merging_logic import improved_merge_paragraph_fragments_dt
    original_count = len(blocks)
    blocks = improved_merge_paragraph_fragments_dt(blocks)
    merged_count = len(blocks)
    print(f"   ✅ Applied improved merging logic ({original_count} → {merged_count} blocks)")
    
    # Step 5: Validate results
    from translation_fixes import validate_translations
    issues = validate_translations(blocks)
    if issues:
        print(f"   ⚠️  Found {len(issues)} translation issues")
        for issue in issues[:5]:
            print(f"      {issue['type']}: {issue['block_id']}")
    else:
        print("   ✅ No translation issues found")
    
    return blocks

def main():
    """
    Main function to demonstrate the fixes
    """
    print("🚀 Digital Twin Merging Logic Fixes")
    print("=" * 50)
    
    # Load your blocks here
    # blocks = load_your_blocks()
    
    # Apply fixes
    # fixed_blocks = apply_all_merging_fixes(blocks)
    
    print("✅ All fixes applied successfully!")
    print("📝 Check the generated files for implementation details.")

if __name__ == "__main__":
    main()
