#!/usr/bin/env python3
"""
Critical Fixes Validation Test

Validates the three critical fixes implemented:
1. Gemini 1.5 Flash model configuration  
2. Robust Greek language translation enforcement
3. Fixed TOC title translation parsing

This test runs quickly to verify fixes before full pipeline execution.
"""

import asyncio
import logging
import sys

# Configure logging for validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_critical_fixes():
    """Test all three critical fixes in isolation."""
    
    print("🔧 CRITICAL FIXES VALIDATION")
    print("=" * 50)
    
    # Test 1: Gemini 1.5 Flash Configuration
    print("\n1️⃣ Testing Gemini 1.5 Flash Configuration...")
    try:
        from config_manager import ConfigManager
        config = ConfigManager()
        
        model_name = config.get_gemini_model_name()
        print(f"   ✅ Model configured: {model_name}")
        
        if "1.5-flash" in model_name:
            print("   ✅ Gemini 1.5 Flash correctly configured")
            test1_passed = True
        else:
            print("   ❌ Still using wrong model version")
            test1_passed = False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        test1_passed = False
    
    # Test 2: Greek Translation Enhancement
    print("\n2️⃣ Testing Enhanced Greek Translation...")
    try:
        from gemini_service import GeminiService
        
        service = GeminiService()
        
        # Test the language normalization
        normalized = service._normalize_language_code("el")
        print(f"   ✅ Language normalization: 'el' → '{normalized}'")
        
        # Check if enhanced translation method exists
        has_enhanced_method = hasattr(service, 'translate_text_with_context')
        print(f"   ✅ Enhanced translation method available: {has_enhanced_method}")
        
        if normalized == "Greek" and has_enhanced_method:
            print("   ✅ Greek translation enhancement ready")
            test2_passed = True
        else:
            print("   ❌ Greek translation enhancement incomplete")
            test2_passed = False
            
    except Exception as e:
        print(f"   ❌ Greek translation test failed: {e}")
        test2_passed = False
    
    # Test 3: TOC Translation Parsing Fix
    print("\n3️⃣ Testing TOC Translation Parsing Fix...")
    try:
        from pymupdf_yolo_processor import PyMuPDFYOLOProcessor
        
        processor = PyMuPDFYOLOProcessor()
        
        # Check if the robust parsing method exists
        has_robust_parsing = hasattr(processor, '_parse_translated_titles_robust')
        print(f"   ✅ Robust TOC parsing method available: {has_robust_parsing}")
        
        # Test the parsing logic with sample data
        if has_robust_parsing:
            test_translation = "1. Εισαγωγή στην Τεχνητή Νοημοσύνη\n2. Μεθοδολογία Έρευνας\n3. Αποτελέσματα"
            test_originals = ["Introduction to AI", "Research Methodology", "Results"]
            
            parsed = processor._parse_translated_titles_robust(test_translation, test_originals)
            print(f"   ✅ Sample parsing test: {len(parsed)} titles parsed")
            
            if len(parsed) == len(test_originals):
                print("   ✅ TOC translation parsing fix ready")
                test3_passed = True
            else:
                print("   ❌ TOC parsing logic needs adjustment")
                test3_passed = False
        else:
            print("   ❌ Robust parsing method not found")
            test3_passed = False
            
    except Exception as e:
        print(f"   ❌ TOC translation test failed: {e}")
        test3_passed = False
    
    # Final Results
    print("\n🎯 VALIDATION RESULTS")
    print("=" * 50)
    
    tests_passed = sum([test1_passed, test2_passed, test3_passed])
    
    print(f"✅ Gemini 1.5 Flash: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✅ Greek Translation: {'PASS' if test2_passed else 'FAIL'}")  
    print(f"✅ TOC Parsing Fix: {'PASS' if test3_passed else 'FAIL'}")
    
    print(f"\n🏆 OVERALL: {tests_passed}/3 tests passed")
    
    if tests_passed == 3:
        print("🎉 All critical fixes validated successfully!")
        print("💡 Ready to run the full Digital Twin pipeline")
        return True
    else:
        print("⚠️ Some fixes need attention before pipeline execution")
        return False

if __name__ == "__main__":
    asyncio.run(test_critical_fixes()) 