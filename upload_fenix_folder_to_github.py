#!/usr/bin/env python3
"""
Script to upload the entire FENIX folder to GitHub repository
This script will:
1. Initialize git repository if not already done
2. Add all files in the fenix folder
3. Commit and push to GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a git command and handle errors"""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_status():
    """Check if we're in a git repository"""
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 FENIX Folder GitHub Upload Script")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not check_git_status():
        print("📁 Initializing git repository...")
        if not run_command(['git', 'init'], "Initializing git repository"):
            print("❌ Failed to initialize git repository")
            return
        
        # Set up remote origin (you'll need to provide the GitHub repo URL)
        print("\n⚠️  Please provide your GitHub repository URL")
        print("Example: https://github.com/yourusername/fenix.git")
        repo_url = input("GitHub repository URL: ").strip()
        
        if not repo_url:
            print("❌ No repository URL provided. Exiting.")
            return
            
        if not run_command(['git', 'remote', 'add', 'origin', repo_url], "Adding remote origin"):
            print("❌ Failed to add remote origin")
            return
    
    # Get current directory (should be the fenix folder)
    current_dir = os.getcwd()
    print(f"📂 Current directory: {current_dir}")
    
    # Check if we're in the right directory (fenix folder)
    if not os.path.exists('main_workflow_enhanced.py'):
        print("❌ Error: main_workflow_enhanced.py not found in current directory")
        print("Please run this script from the fenix folder")
        return
    
    # Add all files to git
    print("\n📦 Adding all files to git...")
    if not run_command(['git', 'add', '.'], "Adding all files"):
        print("❌ Failed to add files")
        return
    
    # Check what files were added
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], check=True, capture_output=True, text=True)
        added_files = [line.split()[-1] for line in result.stdout.strip().split('\n') if line]
        print(f"📄 Files to be committed: {len(added_files)}")
        for file in added_files[:10]:  # Show first 10 files
            print(f"   - {file}")
        if len(added_files) > 10:
            print(f"   ... and {len(added_files) - 10} more files")
    except:
        print("📄 Files added (could not get detailed list)")
    
    # Commit
    commit_message = "Upload complete FENIX document translation pipeline"
    print(f"\n💾 Committing with message: '{commit_message}'")
    if not run_command(['git', 'commit', '-m', commit_message], "Committing changes"):
        print("❌ Failed to commit")
        return
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    if not run_command(['git', 'push', '-u', 'origin', 'main'], "Pushing to main branch"):
        # Try master branch if main fails
        print("🔄 Trying master branch...")
        if not run_command(['git', 'push', '-u', 'origin', 'master'], "Pushing to master branch"):
            print("❌ Failed to push to GitHub")
            print("Please check your GitHub repository settings and permissions")
            return
    
    print("\n🎉 SUCCESS! FENIX folder uploaded to GitHub")
    print("=" * 50)
    print("📋 Summary:")
    print("   ✅ Git repository initialized/configured")
    print("   ✅ All files added to git")
    print("   ✅ Changes committed")
    print("   ✅ Pushed to GitHub")
    print("\n🔗 You can now view your repository on GitHub!")

if __name__ == '__main__':
    main() 