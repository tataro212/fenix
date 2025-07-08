"""
Intelligent PDF Translation Pipeline - Main Entry Point

This script demonstrates the complete intelligent document processing system
with complexity analysis, routing, and optimized processing strategies.
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our intelligent routing system
from intelligent_document_router import (
    IntelligentDocumentRouter,
    ProcessingStrategy,
    intelligent_document_router
)
from gemini_service import GeminiService
from utils import select_pdf_file, select_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('intelligent_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

async def run_intelligent_pipeline():
    """
    Run the intelligent PDF translation pipeline with user interaction.
    
    This function demonstrates:
    1. Document complexity analysis
    2. Intelligent routing decisions
    3. Optimized processing strategies
    4. Performance tracking
    """
    try:
        print("🎯 Intelligent PDF Translation Pipeline")
        print("=" * 50)
        print()
        
        # Step 1: Initialize services
        print("🚀 Initializing services...")
        gemini_service = GeminiService()
        router = IntelligentDocumentRouter(gemini_service)
        
        # Step 2: Select input file
        print("\n📁 Select PDF file to translate...")
        pdf_path = select_pdf_file()
        if not pdf_path:
            print("❌ No file selected. Exiting.")
            return
        
        print(f"✅ Selected: {os.path.basename(pdf_path)}")
        
        # Step 3: Select output directory
        print("\n📂 Select output directory...")
        output_dir = select_output_directory()
        if not output_dir:
            print("❌ No directory selected. Exiting.")
            return
        
        # Create output filename
        pdf_name = Path(pdf_path).stem
        # STRATEGIC FIX: Use short filename to avoid Windows COM 255-character limitation
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"doc_translated_{timestamp}.docx")
        
        print(f"✅ Output will be saved to: {os.path.basename(output_path)}")
        
        # Step 4: Get target language
        print("\n🌍 Select target language:")
        print("1. Greek (el)")
        print("2. Spanish (es)")
        print("3. French (fr)")
        print("4. German (de)")
        print("5. Italian (it)")
        
        try:
            choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
            if not choice:
                choice = "1"
                
            language_map = {
                "1": "el",
                "2": "es", 
                "3": "fr",
                "4": "de",
                "5": "it"
            }
            target_language = language_map.get(choice, "el")
            
            language_names = {
                "el": "Greek",
                "es": "Spanish",
                "fr": "French", 
                "de": "German",
                "it": "Italian"
            }
            print(f"✅ Target language: {language_names[target_language]}")
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.")
            return
        
        # Step 5: Optional strategy override
        print("\n⚙️ Processing strategy:")
        print("1. Automatic (recommended)")
        print("2. Force Fast Track")
        print("3. Force Hybrid Processing") 
        print("4. Force Full Digital Twin")
        
        try:
            strategy_choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
            if not strategy_choice:
                strategy_choice = "1"
                
            force_strategy = None
            if strategy_choice == "2":
                force_strategy = ProcessingStrategy.FAST_TRACK_SIMPLE
                print("✅ Strategy: Fast Track (forced)")
            elif strategy_choice == "3":
                force_strategy = ProcessingStrategy.HYBRID_PROCESSING
                print("✅ Strategy: Hybrid Processing (forced)")
            elif strategy_choice == "4":
                force_strategy = ProcessingStrategy.FULL_DIGITAL_TWIN
                print("✅ Strategy: Full Digital Twin (forced)")
            else:
                print("✅ Strategy: Automatic routing")
                
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.")
            return
        
        # Step 6: Execute intelligent processing
        print("\n" + "=" * 50)
        print("🎯 Starting Intelligent Document Processing")
        print("=" * 50)
        
        start_time = time.time()
        
        result = await router.route_and_process_document(
            pdf_path=pdf_path,
            output_path=output_path,
            target_language=target_language,
            force_strategy=force_strategy
        )
        
        end_time = time.time()
        
        # Step 7: Display results
        print("\n" + "=" * 50)
        print("📊 Processing Results")
        print("=" * 50)
        
        if result.success:
            print("✅ Processing completed successfully!")
            print(f"   📄 Output file: {result.output_path}")
            print(f"   ⏱️ Total time: {result.processing_time:.2f} seconds")
            print(f"   🎯 Strategy used: {result.strategy_used.value.upper()}")
            
            if result.complexity_analysis:
                analysis = result.complexity_analysis
                print(f"\n📊 Document Analysis:")
                print(f"   Complexity: {analysis.complexity_level.value.upper()}")
                print(f"   Confidence: {analysis.confidence_score:.2f}")
                print(f"   Pages: {analysis.page_count}")
                print(f"   Has images: {'Yes' if analysis.has_images else 'No'}")
                print(f"   Has tables: {'Yes' if analysis.has_tables else 'No'}")
                print(f"   Has TOC: {'Yes' if analysis.has_toc else 'No'}")
                print(f"   Font diversity: {analysis.font_diversity}")
            
            if result.routing_decision:
                decision = result.routing_decision
                print(f"\n🚦 Routing Decision:")
                print(f"   Strategy: {decision.strategy.value.upper()}")
                print(f"   Confidence: {decision.confidence:.2f}")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Estimated time: {decision.estimated_time_minutes:.1f} minutes")
                print(f"   Actual time: {result.processing_time/60:.1f} minutes")
            
            if result.performance_metrics:
                metrics = result.performance_metrics
                print(f"\n⚡ Performance Metrics:")
                for key, value in metrics.items():
                    print(f"   {key}: {value}")
            
        else:
            print("❌ Processing failed!")
            if result.error_message:
                print(f"   Error: {result.error_message}")
        
        # Step 8: Display performance report
        performance_report = router.get_performance_report()
        if performance_report.get('total_documents_processed', 0) > 0:
            print(f"\n📈 Session Performance Report:")
            print(f"   Documents processed: {performance_report['total_documents_processed']}")
            print(f"   Success rate: {performance_report['success_rate']:.1f}%")
            
            distribution = performance_report['strategy_distribution']
            print(f"   Strategy distribution:")
            for strategy, stats in distribution.items():
                print(f"     {strategy}: {stats['count']} docs ({stats['percentage']:.1f}%)")
        
        print("\n" + "=" * 50)
        print("🎉 Intelligent Pipeline Complete")
        print("=" * 50)
        
        # Cleanup
        if hasattr(gemini_service, 'cleanup'):
            await gemini_service.cleanup()
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
    finally:
        print("\nThank you for using the Intelligent PDF Translation Pipeline!")

async def demo_batch_processing():
    """
    Demonstrate batch processing with different document types.
    This function shows how the intelligent router handles various document complexities.
    """
    print("🎯 Intelligent Pipeline - Batch Processing Demo")
    print("=" * 50)
    
    # Initialize router
    gemini_service = GeminiService()
    router = IntelligentDocumentRouter(gemini_service)
    
    # Demo documents (you would replace these with actual file paths)
    demo_documents = [
        {
            'path': 'demo_simple.pdf',
            'description': 'Simple text document (< 10 pages, no images)',
            'expected_strategy': ProcessingStrategy.FAST_TRACK_SIMPLE
        },
        {
            'path': 'demo_moderate.pdf', 
            'description': 'Moderate complexity (tables, some formatting)',
            'expected_strategy': ProcessingStrategy.HYBRID_PROCESSING
        },
        {
            'path': 'demo_complex.pdf',
            'description': 'Complex document (images, TOC, equations)',
            'expected_strategy': ProcessingStrategy.FULL_DIGITAL_TWIN
        }
    ]
    
    print("This demo would process the following document types:")
    for i, doc in enumerate(demo_documents, 1):
        print(f"{i}. {doc['description']}")
        print(f"   Expected strategy: {doc['expected_strategy'].value}")
        print()
    
    print("Note: Replace demo file paths with actual PDF files to run this demo.")
    
    # Cleanup
    if hasattr(gemini_service, 'cleanup'):
        await gemini_service.cleanup()

def main():
    """Main entry point"""
    print("🎯 Intelligent PDF Translation Pipeline")
    print("Choose mode:")
    print("1. Interactive Processing (recommended)")
    print("2. Batch Processing Demo")
    
    try:
        choice = input("\nEnter choice (1-2) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            asyncio.run(run_intelligent_pipeline())
        elif choice == "2":
            asyncio.run(demo_batch_processing())
        else:
            print("Invalid choice. Running interactive mode.")
            asyncio.run(run_intelligent_pipeline())
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 