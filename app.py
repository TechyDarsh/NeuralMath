import os
from flask import Flask, request, jsonify, render_template
from backend.preprocess import preprocess_image
from backend.segment import segment_image
from backend.recognize import recognize_symbols
from backend.equation_builder import build_equation
from backend.solver import solve_equation
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'dataset', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Preprocess
            image, thresh = preprocess_image(filepath)
            
            # 2. Segment
            symbols = segment_image(thresh)
            
            if not symbols:
                 return jsonify({'error': 'No symbols detected in the image.'}), 400
                 
            # 3. Recognize
            recognized = recognize_symbols(symbols)
            
            # 4. Equation Reconstruction
            equation_str = build_equation(recognized)
            
            # 5. Solve
            solution = solve_equation(equation_str)
            
            return jsonify({
                'equation': equation_str,
                'solution': solution
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Unknown error'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
