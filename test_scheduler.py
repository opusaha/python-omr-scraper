#!/usr/bin/env python3
"""
Test script for the background cleanup scheduler
Run this to test the scheduler functionality
"""

import time
import os
from scheduler import MarkemImagesCleanupScheduler

def create_test_files():
    """Create some test files in marked_images directory"""
    os.makedirs("marked_images", exist_ok=True)
    
    test_files = [
        "marked_images/test1.png",
        "marked_images/test2.jpg", 
        "marked_images/test3.txt"
    ]
    
    for file_path in test_files:
        with open(file_path, 'w') as f:
            f.write("Test file content")
    
    print(f"Created {len(test_files)} test files:")
    for file_path in test_files:
        print(f"  - {file_path}")

def test_scheduler():
    """Test the scheduler functionality"""
    print("=== Background Cleanup Scheduler Test ===\n")
    
    # Create test files
    print("1. Creating test files...")
    create_test_files()
    
    # Initialize scheduler
    print("\n2. Initializing scheduler...")
    scheduler = MarkemImagesCleanupScheduler()
    
    # Test status before starting
    print("\n3. Checking status before starting...")
    status = scheduler.get_status()
    print(f"Status: {status}")
    
    # Test manual cleanup
    print("\n4. Testing manual cleanup...")
    scheduler.run_cleanup_now()
    
    # Check if files were deleted
    print("\n5. Checking if files were deleted...")
    remaining_files = os.listdir("marked_images") if os.path.exists("marked_images") else []
    print(f"Remaining files: {remaining_files}")
    
    if len(remaining_files) == 0:
        print("✅ Manual cleanup test PASSED!")
    else:
        print("❌ Manual cleanup test FAILED!")
    
    # Test scheduler start/stop
    print("\n6. Testing scheduler start/stop...")
    scheduler.start()
    status = scheduler.get_status()
    print(f"Status after start: {status}")
    
    if status['is_running']:
        print("✅ Scheduler start test PASSED!")
    else:
        print("❌ Scheduler start test FAILED!")
    
    time.sleep(2)  # Wait a bit
    
    scheduler.stop()
    status = scheduler.get_status()
    print(f"Status after stop: {status}")
    
    if not status['is_running']:
        print("✅ Scheduler stop test PASSED!")
    else:
        print("❌ Scheduler stop test FAILED!")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_scheduler()
