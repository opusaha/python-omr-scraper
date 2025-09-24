import os
import schedule
import time
import threading
import logging
from datetime import datetime
import glob

class MarkemImagesCleanupScheduler:
    """Background scheduler to clean marked_images folder daily at 2 AM"""
    
    def __init__(self, marked_images_dir="marked_images"):
        self.marked_images_dir = marked_images_dir
        self.is_running = False
        self.scheduler_thread = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for cleanup operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cleanup_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MarkemImagesCleanupScheduler')
        
    def cleanup_marked_images(self):
        """Clean up all files in the marked_images directory"""
        try:
            if not os.path.exists(self.marked_images_dir):
                self.logger.info(f"Directory {self.marked_images_dir} does not exist. Creating it.")
                os.makedirs(self.marked_images_dir, exist_ok=True)
                return
            
            # Get all files in the marked_images directory
            files_pattern = os.path.join(self.marked_images_dir, "*")
            files_to_delete = glob.glob(files_pattern)
            
            if not files_to_delete:
                self.logger.info(f"No files found in {self.marked_images_dir} directory.")
                return
            
            deleted_count = 0
            failed_count = 0
            
            for file_path in files_to_delete:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                        self.logger.info(f"Deleted file: {os.path.basename(file_path)}")
                    elif os.path.isdir(file_path):
                        # Skip subdirectories for safety
                        self.logger.warning(f"Skipped subdirectory: {os.path.basename(file_path)}")
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to delete {os.path.basename(file_path)}: {str(e)}")
            
            self.logger.info(f"Cleanup completed. Deleted: {deleted_count} files, Failed: {failed_count} files")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup operation: {str(e)}")
    
    def schedule_cleanup(self):
        """Schedule the cleanup to run daily at 2 AM"""
        schedule.every().day.at("02:00").do(self.cleanup_marked_images)
        self.logger.info("Scheduled daily cleanup at 2:00 AM")
    
    def run_scheduler(self):
        """Run the scheduler in a background thread"""
        self.logger.info("Starting cleanup scheduler...")
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        self.logger.info("Cleanup scheduler stopped")
    
    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.schedule_cleanup()
        
        # Start scheduler in a background thread
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Background cleanup scheduler started successfully")
    
    def stop(self):
        """Stop the background scheduler"""
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        self.logger.info("Background cleanup scheduler stopped")
    
    def run_cleanup_now(self):
        """Manually trigger cleanup operation"""
        self.logger.info("Manual cleanup triggered")
        self.cleanup_marked_images()
    
    def get_status(self):
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'next_run': str(schedule.next_run()) if schedule.jobs else None,
            'total_jobs': len(schedule.jobs),
            'marked_images_dir': self.marked_images_dir
        }

# Global scheduler instance
cleanup_scheduler = MarkemImagesCleanupScheduler()

def start_cleanup_scheduler():
    """Function to start the cleanup scheduler"""
    cleanup_scheduler.start()

def stop_cleanup_scheduler():
    """Function to stop the cleanup scheduler"""
    cleanup_scheduler.stop()

def manual_cleanup():
    """Function to manually trigger cleanup"""
    cleanup_scheduler.run_cleanup_now()

def get_scheduler_status():
    """Function to get scheduler status"""
    return cleanup_scheduler.get_status()

if __name__ == "__main__":
    # Test the scheduler
    print("Starting cleanup scheduler test...")
    cleanup_scheduler.start()
    
    try:
        # Keep the main thread alive for testing
        while True:
            status = cleanup_scheduler.get_status()
            print(f"Scheduler Status: {status}")
            time.sleep(30)  # Print status every 30 seconds
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        cleanup_scheduler.stop()
        print("Scheduler stopped.")
