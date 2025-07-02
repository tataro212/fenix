#!/usr/bin/env python3
"""
Test Parallel Processing with Multiple Pages
"""

import os
import sys
import asyncio
import logging
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_parallel_processing():
    """Test parallel processing with multiple pages"""
    
    print("Testing Parallel Processing with Multiple Pages")
    print("=" * 60)
    
    # Create multiple test files to simulate multi-page processing
    test_files = []
    for i in range(3):
        content = f"""
Page {i+1} Title

This is page {i+1} of our test document. It contains sample content to test parallel processing capabilities.

This page has multiple paragraphs to simulate real document content.

List item 1 for page {i+1}
List item 2 for page {i+1}
List item 3 for page {i+1}
        """
        
        filename = f"test_page_{i+1}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        test_files.append(filename)
    
    print(f"Created {len(test_files)} test files for parallel processing")
    
    try:
        from optimized_document_pipeline import process_pdf_optimized
        
        # Test parallel processing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using output directory: {temp_dir}")
            
            start_time = time.time()
            
            # Process each file in parallel
            tasks = []
            for test_file in test_files:
                task = process_pdf_optimized(
                    test_file, 
                    temp_dir, 
                    target_language='fr',
                    max_workers=6
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            print(f"\nParallel Processing Results:")
            print(f"  Total files processed: {len(test_files)}")
            print(f"  Successful: {len(successful_results)}")
            print(f"  Failed: {len(failed_results)}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average time per file: {total_time/len(test_files):.2f}s")
            
            if successful_results:
                avg_processing_time = sum(r.statistics.processing_time for r in successful_results) / len(successful_results)
                print(f"  Average pipeline time: {avg_processing_time:.2f}s")
                print(f"  Parallel efficiency: {avg_processing_time/total_time:.1%}")
            
            # Show strategy distribution
            strategy_counts = {}
            for result in successful_results:
                for strategy, count in result.statistics.strategy_distribution.items():
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + count
            
            print(f"  Strategy distribution:")
            for strategy, count in strategy_counts.items():
                print(f"    {strategy}: {count}")
            
            # Show any errors
            if failed_results:
                print(f"  Errors:")
                for i, result in enumerate(failed_results):
                    print(f"    File {i+1}: {result.error}")
        
    except Exception as e:
        print(f"Parallel processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\nCleanup")
    for test_file in test_files:
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"Removed: {test_file}")
        except Exception as e:
            print(f"Cleanup warning for {test_file}: {e}")
    
    print("\nParallel processing test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_parallel_processing())
    
    if success:
        print("\nParallel processing test passed!")
        sys.exit(0)
    else:
        print("\nParallel processing test failed.")
        sys.exit(1) 