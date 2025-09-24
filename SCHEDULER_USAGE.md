# Background Cleanup Scheduler Usage Guide

## Overview
এই প্রজেক্টে একটি background cleanup scheduler যোগ করা হয়েছে যা প্রতিদিন রাত ২টায় `marked_images` ফোল্ডার পরিষ্কার করবে।

## Features
- **Automatic Daily Cleanup**: প্রতিদিন রাত ২:০০ টায় automatically marked_images ফোল্ডার clean করে
- **Manual Cleanup**: যেকোনো সময় manually cleanup trigger করা যায়
- **Scheduler Management**: Scheduler start/stop করা যায়
- **Status Monitoring**: Scheduler এর current status দেখা যায়
- **Comprehensive Logging**: সব cleanup operations log হয়

## Installation

1. প্রথমে নতুন dependencies install করুন:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Application Start করা
যখন আপনি main application run করবেন, scheduler automatically start হয়ে যাবে:

```bash
python index.py
```

Console এ দেখবেন:
```
Starting background cleanup scheduler...
Background cleanup scheduler started successfully!
```

### 2. API Endpoints

#### Manual Cleanup Trigger
```http
POST http://localhost:8000/cleanup/manual
```

Response:
```json
{
    "success": true,
    "message": "Manual cleanup completed successfully"
}
```

#### Scheduler Status Check
```http
GET http://localhost:8000/cleanup/status
```

Response:
```json
{
    "success": true,
    "scheduler_status": {
        "is_running": true,
        "next_run": "2024-09-25 02:00:00",
        "total_jobs": 1,
        "marked_images_dir": "marked_images"
    }
}
```

#### Start Scheduler (if stopped)
```http
POST http://localhost:8000/cleanup/start
```

#### Stop Scheduler
```http
POST http://localhost:8000/cleanup/stop
```

### 3. Logging

Cleanup operations এর সব log `cleanup_scheduler.log` file এ save হবে। Log levels:
- **INFO**: Successful operations
- **WARNING**: Minor issues
- **ERROR**: Failed operations

Example log entries:
```
2024-09-24 14:30:00 - MarkemImagesCleanupScheduler - INFO - Scheduled daily cleanup at 2:00 AM
2024-09-24 14:30:00 - MarkemImagesCleanupScheduler - INFO - Background cleanup scheduler started successfully
2024-09-25 02:00:00 - MarkemImagesCleanupScheduler - INFO - Deleted file: tmp11ypg1rb_marked.png
2024-09-25 02:00:00 - MarkemImagesCleanupScheduler - INFO - Cleanup completed. Deleted: 1 files, Failed: 0 files
```

## Configuration

### Change Cleanup Time
Default cleanup time হলো রাত ২:০০ টা। এটা change করতে `scheduler.py` file এ:

```python
# Line 64 এ এই line টা change করুন:
schedule.every().day.at("02:00").do(self.cleanup_marked_images)

# যেমন সকাল ৬টায় cleanup করতে চাইলে:
schedule.every().day.at("06:00").do(self.cleanup_marked_images)
```

### Change Cleanup Directory
Default directory হলো `marked_images`। এটা change করতে:

```python
# scheduler.py তে MarkemImagesCleanupScheduler class এর __init__ method এ:
def __init__(self, marked_images_dir="your_custom_directory"):
```

## Testing

### Manual Test
Scheduler independent ভাবে test করতে:

```bash
python scheduler.py
```

এটা scheduler start করবে এবং প্রতি 30 সেকেন্ডে status print করবে।

### API Test with cURL

```bash
# Manual cleanup trigger
curl -X POST http://localhost:8000/cleanup/manual

# Status check
curl http://localhost:8000/cleanup/status

# Start scheduler
curl -X POST http://localhost:8000/cleanup/start

# Stop scheduler  
curl -X POST http://localhost:8000/cleanup/stop
```

## Safety Features

1. **Directory Safety**: শুধুমাত্র files delete করে, subdirectories skip করে
2. **Error Handling**: যদি কোনো file delete করতে problem হয়, error log করে কিন্তু process continue থাকে
3. **Auto Directory Creation**: marked_images directory না থাকলে automatically create করে
4. **Daemon Thread**: Scheduler daemon thread এ run করে, তাই main application exit হলে automatic stop হয়ে যায়

## Troubleshooting

### Scheduler Not Running
```bash
# Status check করুন
curl http://localhost:8000/cleanup/status

# Manually start করুন
curl -X POST http://localhost:8000/cleanup/start
```

### Files Not Being Deleted
1. Log file check করুন: `cleanup_scheduler.log`
2. File permissions check করুন
3. Manual cleanup try করুন: `curl -X POST http://localhost:8000/cleanup/manual`

### Permission Issues
Windows এ sometimes file lock হয়ে থাকে। সেক্ষেত্রে:
1. কোনো application marked images use করছে কিনা check করুন
2. Administrator permission দিয়ে run করুন

## Best Practices

1. **Regular Monitoring**: Periodically log file check করুন
2. **Disk Space**: Large files এর জন্য disk space monitor করুন
3. **Backup**: Important files backup রাখুন cleanup এর আগে
4. **Testing**: Production এ deploy করার আগে test environment এ test করুন

এই scheduler এখন আপনার marked_images folder automatically clean রাখবে এবং disk space save করবে!
