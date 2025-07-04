#!/usr/bin/env python3
"""
Simple Digital Twin Pipeline Runner

Quick launcher for the Digital Twin document processing pipeline.
Just run: python run_digital_twin.py
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Digital Twin Document Pipeline...")
    
    # Check if the main script exists
    script_path = "run_digital_twin_pipeline.py"
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found in current directory")
        print("Please ensure you're running this from the correct directory")
        return 1
    
    try:
        # Run the Digital Twin pipeline
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Digital Twin pipeline failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 