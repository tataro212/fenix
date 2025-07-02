#!/usr/bin/env python3
print("Starting minimal test...")

try:
    print("Testing import...")
    import intelligent_content_batcher
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.") 