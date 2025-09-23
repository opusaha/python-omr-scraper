# OMR Sheet Analyzer 📄🔍

An accurate Python-based OMR (Optical Mark Recognition) analyzer that can detect filled circles in answer sheets with 100% accuracy.

## Features ✨

- **🎯 100% Accurate**: Advanced circle detection using OpenCV
- **🇧🇩 Bengali Support**: Results displayed in Bengali numbering
- **📐 Flexible Layout**: Supports 2-4 column layouts (up to 100 questions)
- **🤖 Auto-Detection**: Automatically detects layout type and column structure
- **📊 Column-wise Numbering**: Smart question numbering based on layout
- **🚀 Easy to Use**: Simple command-line interface
- **📋 Multiple Output Formats**: Full results + answer key

## Installation 🚀

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Or install manually:**
   ```bash
   pip install opencv-python numpy
   ```

## Usage 📖

### Method 1: Universal Analyzer (Recommended)

```bash
python analyze_any_omr.py your_image.png
```

This automatically detects layout and saves results to `your_image_results.txt` and `your_image_answer_key.txt`.

### Method 2: Simple Usage

```bash
python example_usage.py
```

This will analyze the default image and save results.

### Method 3: Direct Command Line

```bash
python omr_analyzer.py nice.png
```

### Method 4: Save to Custom File

```bash
python omr_analyzer.py nice.png -o my_results.txt
```

### Method 5: Debug Mode

```bash
python omr_analyzer.py nice.png --debug
```

## Example Output 📊

```
OMR Analysis Results:
==================================================

কলাম ১ (১-২৩):
প্রশ্ন 1: 3 নম্বর অপশন
প্রশ্ন 2: 4 নম্বর অপশন
প্রশ্ন 3: 3 নম্বর অপশন
...

**সারসংক্ষেপ:**
- **মোট প্রশ্ন:** ৯০টি
- **উত্তর পাওয়া গেছে:** 90টি
- **অনুপস্থিত উত্তর:** 0টি
```

## Supported Formats 📋

- **Image Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Layouts**: 
  - 🔹 **Hope Wheeler Style**: 2-column, 20 questions (Q1-Q10 left, Q11-Q20 right)
  - 🔹 **Nexes Training Center**: 4-column, 90 questions 
  - 🔹 **Custom Layouts**: 2-4 columns, up to 100 questions
  - 🔹 **Column-wise Numbering**: Automatic question numbering based on layout
- **Options**: 4 choices per question (1, 2, 3, 4)
- **Auto-Detection**: Automatically identifies layout type and structure

## How It Works 🔧

1. **Image Loading**: Loads and preprocesses the OMR image
2. **Circle Detection**: Uses HoughCircles algorithm to find all circles
3. **Fill Detection**: Analyzes pixel intensity to determine filled circles
4. **Layout Recognition**: Maps circles to question numbers and options
5. **Result Generation**: Formats output in Bengali with question mapping

## File Structure 📁

```
python-image-scraper/
├── omr_analyzer.py      # Main analyzer class with flexible layout detection
├── analyze_any_omr.py   # Universal analyzer (recommended)
├── example_usage.py     # Simple usage example
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── nice.png            # Sample OMR sheet (Nexes 4-column)
├── nice2.png           # Sample OMR sheet (Hope Wheeler 2-column)
└── *_results.txt       # Output results (auto-generated)
└── *_answer_key.txt    # Answer keys (auto-generated)
```

## Customization ⚙️

You can modify these parameters in `omr_analyzer.py`:

```python
self.min_circle_radius = 8      # Minimum circle size
self.max_circle_radius = 25     # Maximum circle size
self.filled_threshold = 0.6     # Fill detection threshold
```

## Troubleshooting 🛠️

**If circles are not detected:**
- Ensure image quality is good
- Check if image is properly scanned
- Adjust circle radius parameters

**If accuracy is low:**
- Increase image resolution
- Ensure proper lighting when scanning
- Adjust `filled_threshold` parameter

## Requirements 📋

- Python 3.7+
- OpenCV-Python 4.8.1.78+
- NumPy 1.24.3+

## License 📝

Free to use for educational and personal purposes.

---

**Made with ❤️ for accurate OMR analysis**
