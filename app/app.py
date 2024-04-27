from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace these URLs with the actual URLs of your music generation and watermark detection APIs
MUSIC_GENERATION_API_URL = "TODO: generate api"
WATERMARK_DETECTION_API_URL = "TODO: watermark api"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_music', methods=['POST'])
def generate_music():
    if 'music_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    music_file = request.files['music_file']
    
    if music_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Call the music generation API
    try:
        response = requests.post(MUSIC_GENERATION_API_URL, files={'music_file': music_file})
        generated_music = response.json()
        return jsonify(generated_music)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_watermark', methods=['POST'])
def detect_watermark():
    if 'music_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    music_file = request.files['music_file']
    
    if music_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Call the watermark detection API
    try:
        response = requests.post(WATERMARK_DETECTION_API_URL, files={'music_file': music_file})
        watermark_result = response.json()
        return jsonify(watermark_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
