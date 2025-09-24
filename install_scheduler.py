#!/usr/bin/env python3
"""
Installation script for the background cleanup scheduler
Run this after setting up the scheduler files
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating necessary directories...")
    try:
        os.makedirs("marked_images", exist_ok=True)
        print("✅ Created marked_images directory")
        return True
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False

def test_import():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    try:
        import schedule
        print("✅ schedule module imported successfully")
        
        from scheduler import MarkemImagesCleanupScheduler
        print("✅ scheduler module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation function"""
    print("=== Background Cleanup Scheduler Installation ===\n")
    
    success = True
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    print()
    
    # Create directories
    if not create_directories():
        success = False
    
    print()
    
    # Test imports
    if not test_import():
        success = False
    
    print()
    
    if success:
        print("🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python test_scheduler.py' to test the scheduler")
        print("2. Run 'python index.py' to start the main application with scheduler")
        print("3. Check 'SCHEDULER_USAGE.md' for detailed usage instructions")
    else:
        print("❌ Installation failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
