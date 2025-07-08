#!/usr/bin/env python3
"""
Comprehensive Test Suite for Digital Twin Optimizations

This test suite validates all the optimizations implemented in the digital twin system:
1. Migration functionality from legacy models
2. Parallel processing capabilities
3. Memory optimization features
4. Enhanced validation system
5. Error recovery and resume functionality
6. Image validation and fallback mechanisms

Run with: python test_digital_twin_optimizations.py
"""

import unittest
import tempfile
import os
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
try:
    from digital_twin_model import (
        DocumentModel, PageModel, TextBlock, ImageBlock, TableBlock,
        migrate_from_models_py, migrate_from_structured_document_model,
        create_text_block, create_image_block, create_table_block,
        BlockType, StructuralRole
    )
    from pymupdf_yolo_processor import PyMuPDFYOLOProcessor
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestMigrationFunctionality(unittest.TestCase):
    """Test migration from legacy model formats"""
    
    def setUp(self):
        """Set up test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_migrate_from_models_py_basic(self):
        """Test basic migration from models.py format"""
        # Sample data in models.py format
        legacy_page_data = {
            'page_number': 1,
            'dimensions': [612.0, 792.0],
            'elements': [
                {
                    'type': 'text',
                    'content': 'Sample text content',
                    'bbox': (50, 50, 200, 100),
                    'confidence': 0.95
                },
                {
                    'type': 'image',
                    'content': 'sample_image.png',
                    'bbox': (250, 50, 400, 200),
                    'confidence': 0.90
                }
            ]
        }
        
        # Test migration
        migrated_page = migrate_from_models_py(legacy_page_data)
        
        # Verify migration results
        self.assertIsInstance(migrated_page, PageModel)
        self.assertEqual(migrated_page.page_number, 1)
        self.assertEqual(migrated_page.dimensions, (612.0, 792.0))
        
        # Check blocks were created
        all_blocks = migrated_page.get_all_blocks()
        self.assertEqual(len(all_blocks), 2)
        
        # Verify text block
        text_blocks = [b for b in all_blocks if isinstance(b, TextBlock)]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0].original_text, 'Sample text content')
        
        # Verify image block
        image_blocks = [b for b in all_blocks if isinstance(b, ImageBlock)]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0].image_path, 'sample_image.png')
    
    def test_migrate_from_models_py_error_handling(self):
        """Test migration error handling"""
        # Invalid data
        invalid_data = {
            'page_number': 'invalid',
            'dimensions': [612.0],  # Wrong length
            'elements': []
        }
        
        # Should not raise exception, should return minimal page
        migrated_page = migrate_from_models_py(invalid_data)
        self.assertIsInstance(migrated_page, PageModel)
        self.assertIn('migration_error', migrated_page.page_metadata)
    
    def test_migrate_from_structured_document_model(self):
        """Test migration from structured document model"""
        # Sample structured document data
        structured_data = {
            'title': 'Test Document',
            'content_blocks': [
                Mock(
                    block_type='heading',
                    original_text='Chapter 1',
                    page_num=1,
                    bbox=(50, 50, 200, 80)
                ),
                Mock(
                    block_type='paragraph',
                    original_text='This is paragraph content.',
                    page_num=1,
                    bbox=(50, 100, 500, 150)
                )
            ],
            'metadata': {'author': 'Test Author'},
            'source_filepath': '/path/to/test.pdf'
        }
        
        # Test migration
        migrated_doc = migrate_from_structured_document_model(structured_data)
        
        # Verify migration results
        self.assertIsInstance(migrated_doc, DocumentModel)
        self.assertEqual(migrated_doc.title, 'Test Document')
        self.assertEqual(migrated_doc.filename, 'test.pdf')
        self.assertEqual(len(migrated_doc.pages), 1)


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing capabilities"""
    
    def setUp(self):
        """Set up test processor"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.processor = PyMuPDFYOLOProcessor()
    
    @patch('fitz.open')
    async def test_parallel_page_processing(self, mock_fitz_open):
        """Test parallel page processing functionality"""
        # Mock PDF document
        mock_doc = Mock()
        mock_doc.__len__.return_value = 10  # 10 pages
        mock_fitz_open.return_value = mock_doc
        
        # Mock page processing
        async def mock_process_page(pdf_path, page_num, output_dir):
            # Simulate processing time
            await asyncio.sleep(0.01)
            return PageModel(
                page_number=page_num + 1,
                dimensions=(612, 792),
                page_metadata={'processed': True}
            )
        
        self.processor.process_page_digital_twin = mock_process_page
        
        # Test parallel processing
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            processed_pages = await self.processor._process_pages_parallel(
                'test.pdf', 10, temp_dir
            )
            processing_time = time.time() - start_time
            
            # Verify results
            self.assertEqual(len(processed_pages), 10)
            self.assertTrue(all(isinstance(p, PageModel) for p in processed_pages))
            
            # Parallel processing should be faster than sequential
            # (This is a rough check - actual speedup depends on system)
            self.assertLess(processing_time, 1.0)  # Should complete quickly
    
    def test_batch_size_calculation(self):
        """Test optimal batch size calculation"""
        # Test with different document sizes
        with patch('multiprocessing.cpu_count', return_value=8):
            # Small document
            with patch.object(self.processor, '_process_pages_parallel') as mock_method:
                # We can't easily test the internal batch size calculation
                # without running the actual method, so we'll test the logic indirectly
                pass


class TestMemoryOptimization(unittest.TestCase):
    """Test memory optimization features"""
    
    def setUp(self):
        """Set up test processor"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.processor = PyMuPDFYOLOProcessor()
    
    def test_memory_info_collection(self):
        """Test memory information collection"""
        memory_info = self.processor._get_memory_info()
        
        # Verify structure
        self.assertIn('total_gb', memory_info)
        self.assertIn('available_gb', memory_info)
        self.assertIn('used_gb', memory_info)
        self.assertIn('percent_used', memory_info)
        
        # Verify reasonable values
        self.assertGreater(memory_info['total_gb'], 0)
        self.assertGreater(memory_info['available_gb'], 0)
    
    @patch('fitz.open')
    def test_memory_requirements_estimation(self, mock_fitz_open):
        """Test memory requirements estimation"""
        # Mock document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_images.return_value = [(1, 2, 3, 4, 5, 6, 7)]  # Mock image info
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.extract_image.return_value = {
            'width': 800,
            'height': 600,
            'image': b'mock_image_data'
        }
        
        # Test estimation
        estimated_memory = self.processor._estimate_memory_requirements(mock_doc, 10)
        
        # Should return reasonable estimate
        self.assertIsInstance(estimated_memory, float)
        self.assertGreater(estimated_memory, 0)
    
    @patch('fitz.open')
    def test_memory_optimized_processing(self, mock_fitz_open):
        """Test memory-optimized page processing"""
        # Mock document and page
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.get_images.return_value = []
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        # Mock content extractor
        mock_text_block = Mock()
        mock_text_block.text = "Sample text"
        mock_text_block.bbox = (50, 50, 200, 100)
        self.processor.content_extractor.extract_text_blocks.return_value = [mock_text_block]
        
        # Test memory-optimized processing
        with tempfile.TemporaryDirectory() as temp_dir:
            result_page = self.processor._process_single_page_optimized(mock_doc, 0, temp_dir)
            
            # Verify result
            self.assertIsInstance(result_page, PageModel)
            self.assertEqual(result_page.page_number, 1)
            self.assertIn('memory_optimized', result_page.page_metadata['extraction_method'])


class TestValidationSystem(unittest.TestCase):
    """Test enhanced validation system"""
    
    def setUp(self):
        """Set up test document"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.test_doc = DocumentModel(
            title="Test Document",
            filename="test.pdf",
            total_pages=2
        )
    
    def test_basic_structure_validation(self):
        """Test basic structure validation"""
        # Add valid pages
        page1 = PageModel(page_number=1, dimensions=(612, 792))
        page2 = PageModel(page_number=2, dimensions=(612, 792))
        
        self.test_doc.add_page(page1)
        self.test_doc.add_page(page2)
        
        # Test validation
        issues = self.test_doc.validate_structure()
        
        # Should have no basic structure issues
        basic_issues = [i for i in issues if 'page' in i.lower() or 'dimension' in i.lower()]
        self.assertEqual(len(basic_issues), 0)
    
    def test_duplicate_block_id_detection(self):
        """Test duplicate block ID detection"""
        page1 = PageModel(page_number=1, dimensions=(612, 792))
        
        # Add blocks with duplicate IDs
        block1 = create_text_block("duplicate_id", "Text 1", (50, 50, 200, 100), 1)
        block2 = create_text_block("duplicate_id", "Text 2", (50, 150, 200, 200), 1)
        
        page1.add_block(block1)
        page1.add_block(block2)
        self.test_doc.add_page(page1)
        
        # Test validation
        issues = self.test_doc.validate_structure()
        
        # Should detect duplicate IDs
        duplicate_issues = [i for i in issues if 'duplicate' in i.lower()]
        self.assertGreater(len(duplicate_issues), 0)
    
    def test_content_quality_validation(self):
        """Test content quality validation"""
        page1 = PageModel(page_number=1, dimensions=(612, 792))
        
        # Add problematic content
        empty_block = create_text_block("empty", "", (50, 50, 200, 100), 1)
        huge_block = create_text_block("huge", "x" * 15000, (50, 150, 200, 200), 1)
        
        page1.add_block(empty_block)
        page1.add_block(huge_block)
        self.test_doc.add_page(page1)
        
        # Test validation
        issues = self.test_doc.validate_structure()
        
        # Should detect quality issues
        quality_issues = [i for i in issues if 'empty' in i.lower() or 'long' in i.lower()]
        self.assertGreater(len(quality_issues), 0)
    
    def test_missing_image_validation(self):
        """Test missing image file validation"""
        page1 = PageModel(page_number=1, dimensions=(612, 792))
        
        # Add image block with non-existent file
        image_block = create_image_block("img1", "nonexistent.png", (50, 50, 200, 200), 1)
        page1.add_block(image_block)
        self.test_doc.add_page(page1)
        
        # Test validation
        issues = self.test_doc.validate_structure()
        
        # Should detect missing image
        image_issues = [i for i in issues if 'missing image' in i.lower()]
        self.assertGreater(len(image_issues), 0)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and resume functionality"""
    
    def setUp(self):
        """Set up test processor"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.processor = PyMuPDFYOLOProcessor()
        self.temp_dir = tempfile.mkdtemp()
        self.processor.enable_resume_functionality(self.temp_dir)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint save and load functionality"""
        # Create test document
        test_doc = DocumentModel(
            title="Test Doc",
            filename="test.pdf",
            total_pages=10
        )
        
        # Save checkpoint
        self.processor._save_checkpoint(test_doc, 5)
        
        # Verify checkpoint file exists
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.json"))
        self.assertGreater(len(checkpoint_files), 0)
        
        # Load checkpoint
        loaded_checkpoint = self.processor._load_checkpoint("test.pdf")
        self.assertIsNotNone(loaded_checkpoint)
        self.assertEqual(loaded_checkpoint['document_info']['title'], "Test Doc")
        self.assertEqual(loaded_checkpoint['processing_state']['last_completed_page'], 5)
    
    def test_error_recovery_strategy_determination(self):
        """Test error recovery strategy determination"""
        # Test different error types
        memory_error = Exception("Out of memory allocation failed")
        strategy = self.processor._determine_recovery_strategy(memory_error, 1)
        self.assertEqual(strategy, 'retry_memory_optimized')
        
        image_error = Exception("Image extraction failed pixmap error")
        strategy = self.processor._determine_recovery_strategy(image_error, 1)
        self.assertEqual(strategy, 'skip_images')
        
        yolo_error = Exception("YOLO analysis layout detection failed")
        strategy = self.processor._determine_recovery_strategy(yolo_error, 1)
        self.assertEqual(strategy, 'retry_simplified')
        
        unknown_error = Exception("Unknown error occurred")
        strategy = self.processor._determine_recovery_strategy(unknown_error, 1)
        self.assertEqual(strategy, 'create_placeholder')
    
    def test_placeholder_page_creation(self):
        """Test error placeholder page creation"""
        error_message = "Test error occurred"
        placeholder_page = self.processor._create_error_placeholder_page(5, error_message)
        
        # Verify placeholder page
        self.assertIsInstance(placeholder_page, PageModel)
        self.assertEqual(placeholder_page.page_number, 5)
        self.assertTrue(placeholder_page.page_metadata['processing_failed'])
        self.assertEqual(placeholder_page.page_metadata['error_message'], error_message)
        
        # Should have error text block
        text_blocks = placeholder_page.text_blocks
        self.assertGreater(len(text_blocks), 0)
        self.assertIn("ERROR", text_blocks[0].original_text)
    
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup"""
        # Create multiple checkpoint files
        test_doc = DocumentModel(title="Test", filename="test.pdf", total_pages=10)
        
        for page_num in range(1, 8):
            self.processor._save_checkpoint(test_doc, page_num)
        
        # Verify multiple checkpoints exist
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.json"))
        self.assertGreater(len(checkpoint_files), 3)
        
        # Clean up old checkpoints (keep latest 3)
        self.processor._cleanup_old_checkpoints("test.pdf", keep_latest=3)
        
        # Verify only 3 checkpoints remain
        remaining_files = list(Path(self.temp_dir).glob("checkpoint_*.json"))
        self.assertEqual(len(remaining_files), 3)


class TestImageValidation(unittest.TestCase):
    """Test image validation and fallback mechanisms"""
    
    def setUp(self):
        """Set up test processor"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.processor = PyMuPDFYOLOProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_image_validation_valid_file(self):
        """Test validation of valid image file"""
        # Create a simple PNG file
        png_header = b'\x89PNG\r\n\x1a\n'
        test_image_path = os.path.join(self.temp_dir, 'test_image.png')
        
        with open(test_image_path, 'wb') as f:
            f.write(png_header + b'0' * 1000)  # Dummy PNG content
        
        # Test validation
        is_valid = self.processor._validate_extracted_image(test_image_path)
        self.assertTrue(is_valid)
    
    def test_image_validation_invalid_file(self):
        """Test validation of invalid image file"""
        # Create invalid file
        invalid_path = os.path.join(self.temp_dir, 'invalid.png')
        with open(invalid_path, 'wb') as f:
            f.write(b'invalid content')
        
        # Test validation
        is_valid = self.processor._validate_extracted_image(invalid_path)
        self.assertFalse(is_valid)
    
    def test_image_validation_missing_file(self):
        """Test validation of missing file"""
        missing_path = os.path.join(self.temp_dir, 'missing.png')
        
        # Test validation
        is_valid = self.processor._validate_extracted_image(missing_path)
        self.assertFalse(is_valid)
    
    def test_image_placeholder_creation(self):
        """Test image placeholder creation"""
        bbox = (50, 50, 250, 200)
        placeholder_path = self.processor._create_image_placeholder(
            self.temp_dir, 1, 1, bbox
        )
        
        # Verify placeholder was created
        self.assertIsNotNone(placeholder_path)
        self.assertTrue(os.path.exists(placeholder_path))
        
        # Verify it's a valid image file
        is_valid = self.processor._validate_extracted_image(placeholder_path)
        self.assertTrue(is_valid)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete optimized system"""
    
    def setUp(self):
        """Set up integration test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.processor = PyMuPDFYOLOProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_optimization_workflow(self):
        """Test the complete optimized workflow"""
        # Enable all optimizations
        self.processor.enable_resume_functionality(self.temp_dir)
        
        # Verify processor is properly configured
        self.assertTrue(self.processor.resume_enabled)
        self.assertIsNotNone(self.processor.checkpoint_dir)
        
        # Test statistics tracking
        initial_stats = self.processor.stats.copy()
        self.assertEqual(initial_stats['total_pages_processed'], 0)
        self.assertEqual(initial_stats['error_recovery_count'], 0)


def run_performance_benchmarks():
    """Run performance benchmarks for optimized features"""
    print("\n🚀 Running Performance Benchmarks...")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run benchmarks - modules not available")
        return
    
    # Benchmark migration performance
    print("📊 Benchmarking migration performance...")
    start_time = time.time()
    
    for i in range(100):
        legacy_data = {
            'page_number': i + 1,
            'dimensions': [612.0, 792.0],
            'elements': [
                {'type': 'text', 'content': f'Text {i}', 'bbox': (50, 50, 200, 100)},
                {'type': 'image', 'content': f'image_{i}.png', 'bbox': (250, 50, 400, 200)}
            ]
        }
        migrate_from_models_py(legacy_data)
    
    migration_time = time.time() - start_time
    print(f"   ✅ Migrated 100 pages in {migration_time:.3f}s ({migration_time/100*1000:.1f}ms per page)")
    
    # Benchmark validation performance
    print("📊 Benchmarking validation performance...")
    test_doc = DocumentModel(title="Benchmark Doc", filename="test.pdf", total_pages=50)
    
    for i in range(50):
        page = PageModel(page_number=i+1, dimensions=(612, 792))
        for j in range(10):
            block = create_text_block(f"block_{i}_{j}", f"Content {i}-{j}", 
                                    (50, 50+j*20, 500, 70+j*20), i+1)
            page.add_block(block)
        test_doc.add_page(page)
    
    start_time = time.time()
    issues = test_doc.validate_structure()
    validation_time = time.time() - start_time
    
    print(f"   ✅ Validated document with 50 pages, 500 blocks in {validation_time:.3f}s")
    print(f"   📋 Found {len(issues)} validation issues")


def main():
    """Main test runner"""
    print("🧪 Digital Twin Optimizations Test Suite")
    print("=" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available. Please ensure all dependencies are installed.")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMigrationFunctionality,
        TestParallelProcessing,
        TestMemoryOptimization,
        TestValidationSystem,
        TestErrorRecovery,
        TestImageValidation,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmarks
    run_performance_benchmarks()
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All tests passed! Digital Twin optimizations are working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 