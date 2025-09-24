import os
import tempfile
import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from omr_analyzer import OMRAnalyzer

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_bangla_options(answers):
    """Convert numeric answers to Bengali letters"""
    conversion_map = {
        1: 'ক',  # ka
        2: 'খ',  # kha
        3: 'গ',  # ga
        4: 'ঘ'   # gha
    }
    
    bangla_answers = {}
    for question, answer in answers.items():
        if isinstance(answer, (int, str)):
            try:
                numeric_answer = int(answer)
                if numeric_answer in conversion_map:
                    bangla_answers[question] = conversion_map[numeric_answer]
                else:
                    bangla_answers[question] = answer  # Keep original if not in range 1-4
            except (ValueError, TypeError):
                bangla_answers[question] = answer  # Keep original if not numeric
        else:
            bangla_answers[question] = answer
    
    return bangla_answers

def compare_answers(detected_answers, correct_answers):
    """Compare detected answers with correct answers and return match results"""
    if not correct_answers:
        return {}, {}, {}
    
    matched = {}
    unmatched = {}
    missing = {}
    
    # Convert correct_answers keys to integers if they're strings
    if isinstance(correct_answers, dict):
        correct_answers_normalized = {}
        for key, value in correct_answers.items():
            try:
                correct_answers_normalized[int(key)] = value
            except (ValueError, TypeError):
                correct_answers_normalized[key] = value
        correct_answers = correct_answers_normalized
    
    # Compare each detected answer with correct answer
    for question, detected in detected_answers.items():
        if question in correct_answers:
            if detected == correct_answers[question]:
                matched[question] = detected
            else:
                unmatched[question] = {
                    'detected': detected,
                    'correct': correct_answers[question]
                }
        else:
            # No correct answer provided for this question
            matched[question] = detected
    
    # Find missing answers (questions with correct answers but no detection)
    for question, correct in correct_answers.items():
        if question not in detected_answers:
            missing[question] = correct
    
    return matched, unmatched, missing

@app.route('/analyze-omr', methods=['POST'])
def analyze_omr():
    """API endpoint to analyze OMR sheet from uploaded image"""
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Please upload an image with key "image"'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file to temporary location
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Parse correct answers if provided
            correct_answers = None
            if 'correct_answers' in request.form:
                try:
                    correct_answers = json.loads(request.form.get('correct_answers'))
                except (json.JSONDecodeError, TypeError):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid JSON format for correct_answers'
                    }), 400
            
            # Create analyzer and process the image
            analyzer = OMRAnalyzer()
            raw_answers = analyzer.analyze_omr(temp_path)
            
            # Convert numeric answers to Bengali letters
            bangla_answers = convert_to_bangla_options(raw_answers)
            
            # Compare with correct answers if provided
            matched = {}
            unmatched = {}
            missing = {}
            comparison_results = None
            
            if correct_answers:
                matched, unmatched, missing = compare_answers(bangla_answers, correct_answers)
                comparison_results = {
                    'total_correct': len(matched),
                    'total_incorrect': len(unmatched),
                    'total_missing': len(missing),
                    'accuracy_percentage': round((len(matched) / len(correct_answers)) * 100, 2) if correct_answers else 0,
                    'matched_answers': matched,
                    'unmatched_answers': unmatched,
                    'missing_answers': missing
                }
                
                # Generate marked image with correct answers highlighted
                marked_image_path = None
                marked_image_url = None
                # Always generate marked image when correct answers are provided
                marked_image_path = analyzer.mark_unmatched_answers(temp_path, unmatched, correct_answers)
                if marked_image_path:
                    # Generate download URL
                    marked_filename = os.path.basename(marked_image_path)
                    marked_image_url = f"/download-marked-image/{marked_filename}"
            
            # Prepare JSON response
            response_data = {
                'success': True,
                'filename': filename,
                'total_questions': len(bangla_answers),
                'answers': bangla_answers,
                'comparison': comparison_results,
                'marked_image_url': marked_image_url
            }
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file but keep marked image
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Note: marked_image_path is kept for user to download
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/download-marked-image/<path:filename>')
def download_marked_image(filename):
    """Download marked image with correct answers highlighted"""
    try:
        # Construct the full path
        marked_dir = os.path.join(os.getcwd(), 'marked_images')
        file_path = os.path.join(marked_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error downloading file: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'OMR Analyzer API',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'endpoints': {
            'analyze': '/analyze-omr',
            'download': '/download-marked-image/<filename>'
        }
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8000)