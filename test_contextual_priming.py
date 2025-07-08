"""
Test Script for Contextual Priming Translation System

This script demonstrates how the contextual priming system improves translation
quality by analyzing document context and providing domain-specific guidance.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our contextual priming system
from contextual_translation_initializer import (
    initialize_contextual_translation,
    get_contextual_translation_status,
    contextual_initializer
)

# Test documents representing different domains
TEST_DOCUMENTS = {
    'academic': {
        'title': 'Machine Learning Research Paper',
        'text': """
        Abstract

        This paper presents a novel approach to deep learning optimization using adaptive gradient descent algorithms. 
        Our methodology incorporates dynamic learning rate adjustment based on loss function convergence patterns. 
        The experimental results demonstrate significant improvements in training efficiency and model accuracy 
        compared to traditional optimization techniques.

        1. Introduction

        Machine learning has revolutionized numerous fields through its ability to extract patterns from complex datasets. 
        However, the optimization of neural network parameters remains a challenging problem, particularly when dealing 
        with high-dimensional feature spaces and non-convex loss landscapes. Traditional gradient descent methods often 
        suffer from slow convergence rates and susceptibility to local minima.

        Recent advances in adaptive optimization algorithms, such as Adam and RMSprop, have shown promising results 
        in accelerating convergence. Nevertheless, these methods still rely on fixed hyperparameters that may not 
        be optimal across different stages of the training process. Our research addresses this limitation by 
        proposing a dynamic adaptation mechanism that adjusts optimization parameters based on real-time analysis 
        of the loss function behavior.

        2. Methodology

        Our approach builds upon the foundation of stochastic gradient descent (SGD) while incorporating several 
        key innovations. First, we implement a momentum-based parameter update scheme that maintains exponential 
        moving averages of both gradients and their second moments. This dual-averaging approach provides more 
        stable parameter updates and reduces the impact of noisy gradients.
        """
    },
    'technical': {
        'title': 'Software Architecture Documentation',
        'text': """
        System Architecture Overview

        This document describes the microservices architecture for the distributed e-commerce platform. 
        The system is built using containerized services orchestrated with Kubernetes, implementing 
        event-driven communication patterns through Apache Kafka message queues.

        Core Components:

        1. API Gateway Service
        - Handles client requests and routing
        - Implements rate limiting and authentication
        - Built with Node.js and Express framework
        - Deployed on AWS EKS cluster

        2. User Management Service
        - RESTful API for user operations (CRUD)
        - PostgreSQL database with connection pooling
        - JWT token-based authentication
        - Redis cache for session management

        3. Product Catalog Service
        - Elasticsearch for product search functionality
        - MongoDB for product data storage
        - Image processing pipeline using AWS Lambda
        - CDN integration for static asset delivery

        4. Order Processing Service
        - Event-driven architecture with Kafka producers/consumers
        - Saga pattern for distributed transaction management
        - Integration with payment gateway APIs
        - Inventory management with real-time stock updates

        Database Schema:

        The system uses a polyglot persistence approach with different databases optimized for specific use cases:
        - PostgreSQL for transactional data (users, orders)
        - MongoDB for document-based storage (products, reviews)
        - Redis for caching and session storage
        - Elasticsearch for full-text search capabilities

        API Endpoints:

        GET /api/v1/users/{id} - Retrieve user profile
        POST /api/v1/users - Create new user account
        PUT /api/v1/users/{id} - Update user information
        DELETE /api/v1/users/{id} - Deactivate user account
        """
    },
    'medical': {
        'title': 'Clinical Treatment Protocol',
        'text': """
        Treatment Protocol for Acute Myocardial Infarction

        Patient Assessment and Initial Management

        Upon presentation to the emergency department, patients with suspected acute myocardial infarction (AMI) 
        require immediate evaluation and intervention. The initial assessment should include a comprehensive history, 
        physical examination, 12-lead electrocardiogram (ECG), and cardiac biomarker testing.

        Diagnostic Criteria:

        1. Clinical Presentation
        - Chest pain or discomfort lasting >20 minutes
        - Radiation to left arm, jaw, or back
        - Associated symptoms: dyspnea, nausea, diaphoresis
        - Risk factors: diabetes, hypertension, smoking, family history

        2. Electrocardiographic Changes
        - ST-segment elevation ≥1mm in two contiguous leads
        - New left bundle branch block
        - Reciprocal ST-segment depression
        - Q-wave formation in affected leads

        3. Cardiac Biomarkers
        - Troponin I or T elevation above 99th percentile
        - CK-MB elevation (if troponin unavailable)
        - Peak levels typically occur 12-24 hours post-onset

        Treatment Protocol:

        Immediate Management (0-10 minutes):
        - Aspirin 325mg chewed, then 81mg daily
        - Clopidogrel 600mg loading dose, then 75mg daily
        - Atorvastatin 80mg daily
        - Metoprolol 25mg twice daily (if no contraindications)

        Reperfusion Therapy:
        - Primary PCI preferred if available within 90 minutes
        - Fibrinolytic therapy if PCI not available within 120 minutes
        - Contraindications to fibrinolysis: recent surgery, bleeding disorders, stroke

        Post-Intervention Monitoring:
        - Continuous cardiac monitoring for 24-48 hours
        - Serial ECGs every 6 hours for first 24 hours
        - Troponin levels at 6 and 12 hours post-admission
        - Echocardiogram within 24 hours to assess wall motion
        """
    },
    'business': {
        'title': 'Quarterly Financial Report',
        'text': """
        Q3 2024 Financial Performance Summary

        Executive Summary

        The company delivered strong financial performance in Q3 2024, with revenue growth of 15% year-over-year 
        and improved operational efficiency across all business segments. Our strategic initiatives in digital 
        transformation and market expansion continue to drive sustainable growth and enhanced shareholder value.

        Financial Highlights:

        Revenue Performance:
        - Total revenue: $145.2M (vs. $126.3M Q3 2023)
        - Organic growth: 12% excluding acquisitions
        - Recurring revenue: 78% of total revenue
        - Customer retention rate: 94%

        Profitability Metrics:
        - Gross margin: 68.5% (vs. 65.2% Q3 2023)
        - EBITDA: $42.8M (29.5% margin)
        - Net income: $28.1M (19.4% margin)
        - Earnings per share: $1.23 (diluted)

        Balance Sheet Strength:
        - Cash and equivalents: $89.4M
        - Total debt: $156.7M
        - Debt-to-equity ratio: 0.35
        - Working capital: $67.2M

        Key Performance Indicators:

        Customer Metrics:
        - Customer acquisition cost (CAC): $1,247
        - Customer lifetime value (CLV): $18,650
        - Monthly recurring revenue (MRR): $9.8M
        - Net promoter score (NPS): 72

        Operational Efficiency:
        - Employee productivity: +8% vs. Q3 2023
        - Cost per acquisition: -12% improvement
        - Time to market for new products: 15% reduction
        - Customer support response time: <2 hours average

        Market Position:
        - Market share in core segment: 23%
        - Geographic expansion: 3 new markets
        - Product portfolio: 15 new feature releases
        - Strategic partnerships: 7 new agreements

        Forward-Looking Statements:
        Based on current market conditions and our strategic roadmap, we anticipate continued growth in Q4 2024 
        with projected revenue of $152-158M and EBITDA margin of 30-32%. Our investment in R&D and market expansion 
        positions us well for sustained long-term growth.
        """
    }
}

async def test_contextual_priming_system():
    """Test the contextual priming system with different document types"""
    
    logger.info("🚀 Starting Contextual Priming System Test")
    logger.info("=" * 60)
    
    results = {}
    
    for doc_type, doc_data in TEST_DOCUMENTS.items():
        logger.info(f"\n📄 Testing {doc_type.upper()} Document:")
        logger.info(f"Title: {doc_data['title']}")
        logger.info("-" * 40)
        
        try:
            # Initialize contextual priming for this document
            start_time = time.time()
            context = await initialize_contextual_translation(
                doc_data['text'],
                doc_data['title'],
                force_reanalysis=True  # Force new analysis for testing
            )
            analysis_time = time.time() - start_time
            
            # Get status
            status = get_contextual_translation_status()
            
            # Store results
            results[doc_type] = {
                'context': context,
                'status': status,
                'analysis_time': analysis_time
            }
            
            # Display results
            logger.info(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            logger.info(f"📊 Document Type: {context.document_type}")
            logger.info(f"🎯 Domain: {context.domain}")
            logger.info(f"📝 Subject: {context.subject_matter}")
            logger.info(f"🎨 Style: {context.writing_style}")
            logger.info(f"🔊 Tone: {context.tone}")
            logger.info(f"👥 Audience: {context.audience}")
            logger.info(f"⚡ Technical Level: {context.technical_level}")
            logger.info(f"📈 Confidence: {context.analysis_confidence:.2f}")
            logger.info(f"🏷️ Key Terms: {len(context.key_terminology)}")
            
            # Show top key terms
            if context.key_terminology:
                top_terms = sorted(context.key_terminology.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"🔑 Top Terms: {', '.join([term for term, _ in top_terms])}")
            
            # Show domain-specific instructions
            logger.info(f"📋 Domain Instructions: {context.domain_specific_instructions}")
            
        except Exception as e:
            logger.error(f"❌ Error testing {doc_type}: {e}")
            results[doc_type] = {'error': str(e)}
    
    return results

async def test_translation_enhancement():
    """Test how contextual priming enhances translation prompts"""
    
    logger.info("\n" + "=" * 60)
    logger.info("🔬 Testing Translation Enhancement")
    logger.info("=" * 60)
    
    # Use the technical document for this test
    doc_data = TEST_DOCUMENTS['technical']
    
    # Initialize contextual priming
    await initialize_contextual_translation(
        doc_data['text'],
        doc_data['title']
    )
    
    # Import translation service to test prompt enhancement
    try:
        from translation_service import translation_service
        
        # Test text to translate
        test_text = "The API gateway implements rate limiting and authentication mechanisms."
        
        # Get enhanced prompt (this would normally be called internally)
        enhanced_prompt = translation_service.prompt_generator.generate_translation_prompt(
            test_text,
            target_language="Greek",
            style_guide="Technical documentation",
            item_type="technical description"
        )
        
        logger.info(f"🎯 Test Text: {test_text}")
        logger.info(f"📝 Enhanced Prompt Preview:")
        logger.info("-" * 40)
        
        # Show first part of the enhanced prompt
        prompt_lines = enhanced_prompt.split('\n')
        for i, line in enumerate(prompt_lines[:20]):  # Show first 20 lines
            logger.info(f"{line}")
        
        if len(prompt_lines) > 20:
            logger.info(f"... ({len(prompt_lines) - 20} more lines)")
        
        logger.info("-" * 40)
        logger.info("✅ Enhanced prompt generated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error testing translation enhancement: {e}")

async def test_caching_system():
    """Test the contextual priming caching system"""
    
    logger.info("\n" + "=" * 60)
    logger.info("💾 Testing Caching System")
    logger.info("=" * 60)
    
    # Test document
    doc_data = TEST_DOCUMENTS['academic']
    
    # First analysis (should be fresh)
    logger.info("🔍 First analysis (fresh)...")
    start_time = time.time()
    context1 = await initialize_contextual_translation(
        doc_data['text'],
        doc_data['title'],
        force_reanalysis=True
    )
    first_analysis_time = time.time() - start_time
    
    # Second analysis (should use cache)
    logger.info("📋 Second analysis (cached)...")
    start_time = time.time()
    context2 = await initialize_contextual_translation(
        doc_data['text'],
        doc_data['title'],
        force_reanalysis=False
    )
    second_analysis_time = time.time() - start_time
    
    # Compare results
    logger.info(f"⏱️ First analysis time: {first_analysis_time:.2f} seconds")
    logger.info(f"⏱️ Second analysis time: {second_analysis_time:.2f} seconds")
    logger.info(f"🚀 Speed improvement: {first_analysis_time/second_analysis_time:.1f}x faster")
    
    # Verify contexts are equivalent
    if context1.document_type == context2.document_type and context1.domain == context2.domain:
        logger.info("✅ Cached context matches original analysis")
    else:
        logger.warning("⚠️ Cached context differs from original")

async def demonstrate_integration():
    """Demonstrate how to integrate contextual priming into existing workflows"""
    
    logger.info("\n" + "=" * 60)
    logger.info("🔧 Integration Demonstration")
    logger.info("=" * 60)
    
    # Show how to integrate with existing translation functions
    logger.info("📝 Integration Examples:")
    logger.info("-" * 40)
    
    integration_examples = [
        {
            'name': 'Simple Text Translation',
            'code': '''
# At the start of your translation script
from contextual_translation_initializer import initialize_contextual_translation

# Initialize contextual priming
context = await initialize_contextual_translation(
    document_text[:8000],  # First 8000 characters
    "Technical Manual"
)

# Now all translations will use contextual priming automatically
result = await translation_service.translate_text(text, "Greek")
'''
        },
        {
            'name': 'File-based Translation',
            'code': '''
# Initialize from file
from contextual_translation_initializer import initialize_contextual_translation_from_file

context = await initialize_contextual_translation_from_file("research_paper.pdf")

# Process the document with contextual priming active
translated_doc = await translation_service.translate_document(document, "Greek")
'''
        },
        {
            'name': 'Status Monitoring',
            'code': '''
# Check contextual priming status
from contextual_translation_initializer import get_contextual_translation_status

status = get_contextual_translation_status()
print(f"Context: {status['summary']}")
print(f"Confidence: {status['confidence']}")
print(f"Key Terms: {status['key_terms_count']}")
'''
        }
    ]
    
    for example in integration_examples:
        logger.info(f"\n🔹 {example['name']}:")
        for line in example['code'].strip().split('\n'):
            logger.info(f"  {line}")

async def main():
    """Main test function"""
    
    try:
        # Test the contextual priming system
        results = await test_contextual_priming_system()
        
        # Test translation enhancement
        await test_translation_enhancement()
        
        # Test caching system
        await test_caching_system()
        
        # Demonstrate integration
        await demonstrate_integration()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 Test Summary")
        logger.info("=" * 60)
        
        successful_tests = sum(1 for result in results.values() if 'error' not in result)
        total_tests = len(results)
        
        logger.info(f"✅ Successful tests: {successful_tests}/{total_tests}")
        
        if successful_tests == total_tests:
            logger.info("🎉 All tests passed! Contextual priming system is working correctly.")
        else:
            logger.warning(f"⚠️ {total_tests - successful_tests} test(s) failed.")
            
        # Show benefits
        logger.info("\n🌟 Benefits of Contextual Priming:")
        logger.info("• Domain-specific translation guidance")
        logger.info("• Consistent terminology across documents")
        logger.info("• Improved translation quality for technical content")
        logger.info("• Automatic style and tone adaptation")
        logger.info("• Efficient caching for repeated analyses")
        logger.info("• Easy integration with existing workflows")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 