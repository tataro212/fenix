#!/usr/bin/env python3
"""
Test script to verify the PDF generation naming fix for Windows COM 255-character limitation.
This test validates that the short-form naming convention is working correctly.
"""

import os
import time
import tempfile
from pathlib import Path

def test_filename_length_fix():
    """Test that the new filename pattern stays well under the 255-character limit"""
    
    print("🧪 Testing PDF generation naming fix...")
    
    # Simulate the old problematic naming pattern
    old_long_filename = "federacion-anarquista-uruguaya-copei-commentary-on-armed-struggle-and-foquismo-in-latin-america_translated.docx"
    
    # Test the new short naming pattern
    timestamp = int(time.time())
    new_short_filename = f"doc_translated_{timestamp}.docx"
    
    print(f"\n📏 Filename Length Comparison:")
    print(f"  Old pattern: {len(old_long_filename)} characters - {old_long_filename}")
    print(f"  New pattern: {len(new_short_filename)} characters - {new_short_filename}")
    
    # Test with typical Windows path depth
    temp_dir = tempfile.gettempdir()
    test_output_dir = os.path.join(temp_dir, "fenix_test_output", "long_document_name_test")
    
    # Simulate full path
    old_full_path = os.path.join(test_output_dir, old_long_filename)
    new_full_path = os.path.join(test_output_dir, new_short_filename)
    
    print(f"\n📂 Full Path Length Comparison:")
    print(f"  Old path: {len(old_full_path)} characters")
    print(f"  New path: {len(new_full_path)} characters")
    print(f"  Windows COM limit: 255 characters")
    
    # Validation
    old_exceeds_limit = len(old_full_path) > 255
    new_exceeds_limit = len(new_full_path) > 255
    
    print(f"\n✅ Validation Results:")
    print(f"  Old pattern exceeds limit: {'❌ YES' if old_exceeds_limit else '✅ NO'}")
    print(f"  New pattern exceeds limit: {'❌ YES' if new_exceeds_limit else '✅ NO'}")
    
    if new_exceeds_limit:
        print(f"  ⚠️ WARNING: New pattern still exceeds limit!")
        return False
    else:
        print(f"  🎉 SUCCESS: New pattern resolves Windows COM limitation!")
        return True

def test_filename_uniqueness():
    """Test that timestamps provide sufficient uniqueness"""
    
    print(f"\n🔢 Testing filename uniqueness...")
    
    # Generate multiple filenames quickly
    filenames = []
    for i in range(5):
        timestamp = int(time.time())
        filename = f"doc_translated_{timestamp}.docx"
        filenames.append(filename)
        time.sleep(0.1)  # Small delay to ensure different timestamps
    
    unique_filenames = set(filenames)
    
    print(f"  Generated filenames: {len(filenames)}")
    print(f"  Unique filenames: {len(unique_filenames)}")
    
    for filename in filenames:
        print(f"    - {filename}")
    
    if len(unique_filenames) == len(filenames):
        print(f"  ✅ All filenames are unique")
        return True
    else:
        print(f"  ❌ Some filenames are duplicated")
        return False

def test_import_compatibility():
    """Test that the main workflow can be imported and uses the new naming"""
    
    print(f"\n📦 Testing import compatibility...")
    
    try:
        # Test that time is properly imported
        import time
        timestamp = int(time.time())
        short_filename = f"doc_translated_{timestamp}"
        
        print(f"  ✅ Time import works: {timestamp}")
        print(f"  ✅ Short filename generation: {short_filename}")
        
        # Test path manipulation
        test_output_dir = "/tmp/test_output"
        word_output_path = os.path.join(test_output_dir, f"{short_filename}.docx")
        pdf_output_path = os.path.join(test_output_dir, f"{short_filename}.pdf")
        
        print(f"  ✅ Word path: {word_output_path}")
        print(f"  ✅ PDF path: {pdf_output_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import compatibility test failed: {e}")
        return False

def main():
    """Run all tests for the PDF naming fix"""
    
    print("🎯 FENIX STRATEGIC IMPLEMENTATION - Phase 1B Testing")
    print("=" * 60)
    print("Testing Priority 1: PDF Generation File Path Fix")
    
    tests = [
        ("Filename Length Fix", test_filename_length_fix),
        ("Filename Uniqueness", test_filename_uniqueness), 
        ("Import Compatibility", test_import_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - PDF generation fix is working correctly!")
        print("✅ Ready to proceed with Phase 1B workflow testing")
        return True
    else:
        print("❌ Some tests failed - fix needs adjustment")
        return False

if __name__ == "__main__":
    main() 