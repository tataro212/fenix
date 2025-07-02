#!/usr/bin/env python3
"""
Script to upload Main Workflow Enhanced files to tatarofenix GitHub repository
This script will:
1. Empty the existing repository
2. Upload only the core files from the main workflow enhanced system
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

# List of essential files to upload (as per the enhanced module map)
ESSENTIAL_FILES = [
    'main_workflow_enhanced.py',
    'pdf_processor.py',
    'text_translator.py',
    'main.py',
    'document_generator.py',
    'optimized_document_pipeline.py',
    'processing_strategies.py',
    'pymupdf_yolo_processor.py',
    'intelligent_content_batcher.py',
    'parallel_translation_manager.py',
    'async_translation_service.py',
    'translation_service_enhanced.py',
    'config_manager.py',
    'utils.py',
    'test_structured_pipeline.py',
    'MAIN_WORKFLOW_ENHANCED_MODULE_MAP.md',
    'UNIFIED_PIPELINE_INTEGRATION.md',
    'STRUCTURED_PIPELINE_README.md',
    'REFACTORING_SUCCESS_SUMMARY.md',
]

def file_exists(filename):
    return os.path.isfile(filename)

def main():
    files_to_add = [f for f in ESSENTIAL_FILES if file_exists(f)]
    if not files_to_add:
        print("No essential files found to upload.")
        return

    # Add files to git
    add_cmd = ['git', 'add'] + files_to_add
    subprocess.run(add_cmd, check=True)

    # Commit
    commit_msg = 'Upload enhanced main workflow pipeline and structured components'
    commit_cmd = ['git', 'commit', '-m', commit_msg]
    subprocess.run(commit_cmd, check=True)

    # Push
    push_cmd = ['git', 'push']
    subprocess.run(push_cmd, check=True)

    print("\n=== Upload Summary ===")
    for f in files_to_add:
        print(f"Uploaded: {f}")
    print("\nCommit message:")
    print(commit_msg)

if __name__ == '__main__':
    main() 