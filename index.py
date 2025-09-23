import os
import tempfile
from flask import Flask, request, jsonify
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
            # Create analyzer and process the image
            analyzer = OMRAnalyzer()
            raw_answers = analyzer.analyze_omr(temp_path)
            
            # Convert numeric answers to Bengali letters
            bangla_answers = convert_to_bangla_options(raw_answers)
            
            # Prepare JSON response
            response_data = {
                'success': True,
                'filename': filename,
                'total_questions': len(bangla_answers),
                'answers': bangla_answers
            }
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'OMR Analyzer API',
        'supported_formats': list(ALLOWED_EXTENSIONS)
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8000)