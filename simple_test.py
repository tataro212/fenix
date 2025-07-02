#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for Phase 1 implementation
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_import():
    """Test basic import functionality"""
    try:
        from intelligent_content_batcher import IntelligentContentBatcher
        logger.info("✅ IntelligentContentBatcher imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_batcher_creation():
    """Test batcher creation"""
    try:
        from intelligent_content_batcher import IntelligentContentBatcher
        batcher = IntelligentContentBatcher(max_batch_chars=12000)
        logger.info("✅ IntelligentContentBatcher created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Batcher creation failed: {e}")
        return False

def main():
    """Run basic tests"""
    logger.info("Starting basic tests...")
    
    # Test 1: Basic import
    if not test_basic_import():
        return False
    
    # Test 2: Batcher creation
    if not test_batcher_creation():
        return False
    
    logger.info("✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)