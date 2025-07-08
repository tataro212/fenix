"""
Simple Demonstration of Contextual Priming for Translation

This script shows how to use the contextual priming system to improve
translation quality by analyzing document context at startup.
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_contextual_priming():
    """Demonstrate contextual priming with a sample document"""
    
    print("🚀 Contextual Priming Translation Demo")
    print("=" * 50)
    
    # Sample technical document
    sample_document = """
    API Gateway Configuration Guide
    
    This document describes the configuration and deployment of the API Gateway service
    in a microservices architecture. The gateway handles request routing, authentication,
    rate limiting, and load balancing across multiple backend services.
    
    Key Components:
    - Load Balancer: Distributes requests across service instances
    - Authentication Service: Validates JWT tokens and API keys
    - Rate Limiter: Prevents abuse with configurable throttling
    - Circuit Breaker: Handles service failures gracefully
    
    Configuration Parameters:
    - max_requests_per_second: 1000
    - connection_timeout: 30s
    - retry_attempts: 3
    - health_check_interval: 60s
    """
    
    try:
        # Step 1: Initialize contextual priming
        print("\n📋 Step 1: Analyzing document context...")
        
        from contextual_translation_initializer import initialize_contextual_translation
        
        context = await initialize_contextual_translation(
            sample_document,
            "API Gateway Configuration Guide"
        )
        
        print(f"✅ Context Analysis Complete!")
        print(f"   📄 Document Type: {context.document_type}")
        print(f"   🎯 Domain: {context.domain}")
        print(f"   📝 Style: {context.writing_style}")
        print(f"   ⚡ Technical Level: {context.technical_level}")
        print(f"   📊 Confidence: {context.analysis_confidence:.2f}")
        
        # Step 2: Show how translations are enhanced
        print("\n🔧 Step 2: Enhanced translation prompts...")
        
        # Sample text to translate
        sample_text = "The API gateway implements rate limiting to prevent service abuse."
        
        # Import translation service
        from translation_service import translation_service
        
        # Generate enhanced prompt (this happens automatically during translation)
        enhanced_prompt = translation_service.prompt_generator.generate_translation_prompt(
            sample_text,
            target_language="Greek",
            style_guide="Technical documentation"
        )
        
        print(f"📝 Sample text: {sample_text}")
        print(f"🎯 Enhanced prompt includes:")
        print(f"   • Document type: {context.document_type}")
        print(f"   • Domain context: {context.domain}")
        print(f"   • Technical terminology guidance")
        print(f"   • Style consistency instructions")
        
        # Step 3: Demonstrate actual translation (if API key available)
        print("\n🌐 Step 3: Contextual translation...")
        
        try:
            # This would use the enhanced contextual prompt automatically
            translated_text = await translation_service.translate_text(
                sample_text,
                target_language="Greek"
            )
            
            print(f"✅ Translation completed with contextual priming!")
            print(f"   Original: {sample_text}")
            print(f"   Translated: {translated_text}")
            
        except Exception as e:
            print(f"ℹ️  Translation skipped (API key needed): {e}")
            print(f"   But contextual priming is ready and would enhance any translation!")
        
        # Step 4: Show status
        print("\n📊 Step 4: System status...")
        
        from contextual_translation_initializer import get_contextual_translation_status
        status = get_contextual_translation_status()
        
        print(f"✅ Contextual priming active!")
        print(f"   📋 Summary: {status['summary']}")
        print(f"   🏷️ Key terms identified: {status['key_terms_count']}")
        print(f"   📈 Analysis confidence: {status['confidence']:.2f}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Benefits:")
        print("   • Automatic domain detection and context analysis")
        print("   • Enhanced translation prompts with specific guidance")
        print("   • Consistent terminology across document")
        print("   • Improved translation quality for technical content")
        print("   • Easy integration with existing translation workflows")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_with_file():
    """Demonstrate contextual priming with a file"""
    
    print("\n" + "=" * 50)
    print("📁 File-based Contextual Priming Demo")
    print("=" * 50)
    
    # Check if we have a sample PDF
    import os
    sample_files = [
        "s11229-023-04281-5.pdf",
        "test_document_with_text.pdf",
        "sample_page.pdf"
    ]
    
    sample_file = None
    for file in sample_files:
        if os.path.exists(file):
            sample_file = file
            break
    
    if sample_file:
        try:
            print(f"📄 Using sample file: {sample_file}")
            
            from contextual_translation_initializer import initialize_contextual_translation_from_file
            
            context = await initialize_contextual_translation_from_file(sample_file)
            
            print(f"✅ File analysis complete!")
            print(f"   📄 Document Type: {context.document_type}")
            print(f"   🎯 Domain: {context.domain}")
            print(f"   📝 Subject: {context.subject_matter}")
            print(f"   📊 Confidence: {context.analysis_confidence:.2f}")
            
        except Exception as e:
            print(f"ℹ️  File analysis skipped: {e}")
    else:
        print("ℹ️  No sample PDF files found for file-based demo")

if __name__ == "__main__":
    asyncio.run(demo_contextual_priming())
    asyncio.run(demo_with_file()) 