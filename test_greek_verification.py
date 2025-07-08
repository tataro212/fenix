#!/usr/bin/env python3
"""
Greek Translation Verification Test

This test directly checks if the translation system produces actual Greek text
instead of Arabic, Korean, or other languages.

The user reported seeing Arabic and Korean text in their document output.
This test will verify our fixes work.
"""

import asyncio
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_direct_greek_translation():
    """Test direct Greek translation with actual text samples."""
    
    print("🇬🇷 GREEK TRANSLATION VERIFICATION TEST")
    print("=" * 60)
    
    # Test samples from academic context
    test_texts = [
        "Introduction to Artificial Intelligence",
        "Research Methodology and Data Analysis", 
        "Conclusion and Future Work",
        "The emergence of machine learning in modern society has transformed the way we approach complex problems.",
        "Table of Contents"
    ]
    
    print("\n🔍 Testing Enhanced Gemini Service...")
    
    try:
        from gemini_service import GeminiService
        
        # Test the enhanced service directly
        gemini_service = GeminiService()
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Testing: '{text}'")
            
            # Use the enhanced method
            if hasattr(gemini_service, 'translate_text_with_context'):
                translation = await gemini_service.translate_text_with_context(
                    text, 'Greek', 
                    context="Academic document", 
                    translation_style="academic"
                )
            else:
                translation = await gemini_service.translate_text(text, 'Greek')
            
            print(f"   Translation: {translation}")
            
            # Check if it contains Greek characters
            greek_chars = re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', translation)
            
            if greek_chars:
                print(f"   ✅ CONTAINS GREEK CHARACTERS: {len(greek_chars)} found")
            else:
                print(f"   ❌ NO GREEK CHARACTERS FOUND")
            
            # Check for non-Greek scripts
            arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F]', translation)
            korean_chars = re.findall(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]', translation)
            
            if arabic_chars:
                print(f"   ⚠️ CONTAINS ARABIC CHARACTERS: {len(arabic_chars)} found")
            if korean_chars:
                print(f"   ⚠️ CONTAINS KOREAN CHARACTERS: {len(korean_chars)} found")
        
        print("\n🔍 Testing Async Translation Service...")
        
        # Test the async translation service that the pipeline uses
        from async_translation_service import AsyncTranslationService
        
        async_service = AsyncTranslationService()
        
        test_text = "Artificial Intelligence and Machine Learning"
        print(f"\nTesting AsyncTranslationService with: '{test_text}'")
        
        translation = await async_service.translate_text(test_text, 'Greek')
        print(f"Translation: {translation}")
        
        # Check characters
        greek_chars = re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', translation)
        arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F]', translation)
        korean_chars = re.findall(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]', translation)
        
        if greek_chars:
            print(f"✅ AsyncTranslationService produces Greek: {len(greek_chars)} chars")
        else:
            print(f"❌ AsyncTranslationService NOT producing Greek")
            
        if arabic_chars or korean_chars:
            print(f"⚠️ AsyncTranslationService producing mixed languages: {len(arabic_chars)} Arabic, {len(korean_chars)} Korean")
        
        # Overall assessment
        print("\n🎯 FINAL ASSESSMENT")
        print("-" * 30)
        
        if greek_chars and not (arabic_chars or korean_chars):
            print("✅ TRANSLATION FIXED: Producing proper Greek")
            return True
        else:
            print("❌ TRANSLATION STILL BROKEN: Not producing proper Greek")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_greek_translation())
    
    if success:
        print("\n🎉 Greek translation is working correctly!")
        print("💡 The pipeline should now produce proper Greek output.")
    else:
        print("\n⚠️ Greek translation still needs fixing.")
        print("💡 Check the translation service configuration.") 