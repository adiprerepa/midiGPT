from flask import Flask, render_template, request, jsonify, make_response
import json
import torch
from midigpt import GPT, TetradPlayer, WatermarkLogitsProcessor, WatermarkDetector
from midigpt.datasets import BachChoralesEncoder

app = Flask(__name__)

default_metrics = [
    {"name": "z-score Threshold", "value": None},
    {"name": "z-score", "value": None},
    {"name": "p value", "value": None},
    {"name": "Tokens Counted (T)", "value": None},
    {"name": "Prediction", "value": None},
    {"name": "Fraction of T in Greenlist", "value": None},
    {"name": "# Tokens in Greenlist", "value": None}
]

@app.route('/')
def index():
    metrics = default_metrics
    return render_template('index.html', metrics=metrics, w_metrics=metrics, detect_metrics=metrics)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    if 'music_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    music_file = request.files['music_file']

    # write file to temp
    with open("/tmp/temp.wav", 'wb') as f:
        f.write(music_file.read())
    music_file = "/Users/ayushraman/Documents/GitHub/midiGPT/generated-bach-1.5.wav"
    
    #load model and tokenizer
    model = GPT.from_checkpoint("../projects/bach-chorales/best_model.ckpt", map_location=torch.device("cpu"))
    encoder = BachChoralesEncoder()

    max_tokens = int(request.form['max_tokens'])
    tempo = int(request.form['tempo'])
    do_sample = True
    temperature= float(request.form['temperature'])

    print(f"max_tokens: {max_tokens}, tempo: {tempo}, do_sample: {do_sample}, temperature: {temperature}")
    
    chorale = TetradPlayer().from_wav(music_file, tempo = tempo)
    
    seed_notes = encoder.encode(torch.as_tensor(chorale.flatten()))
    
    watermark_processor = WatermarkLogitsProcessor()
    watermark_detector = WatermarkDetector(device = 'cpu', tokenizer= encoder)
    
    unwatermarked_chords = encoder.decode(model.generate(seed_notes, max_tokens, do_sample= do_sample, temperature= temperature))
    
    unwatermark_path = "./out/unwatermarked_generation.wav"
    watermark_path = "./out/watermarked_generation.wav"

    TetradPlayer().to_wav(unwatermarked_chords.flatten().reshape(-1, 4).cpu(), unwatermark_path, tempo = tempo)
    
    unwatermarked_detection_result = watermark_detector.detect(text= unwatermarked_chords[-(max_tokens):])
    
    watermarked_chords = encoder.decode(model.generate(seed_notes, max_tokens, do_sample= do_sample, temperature= temperature, watermark_proccessor= watermark_processor))
    
    TetradPlayer().to_wav(watermarked_chords.flatten().reshape(-1, 4).cpu(), watermark_path, tempo = tempo)

    watermarked_detection_result = watermark_detector.detect(text= watermarked_chords[-(max_tokens):])
    
    
    combined_response = {
        'unwatermarked_buffer': unwatermark_path,
        'watermarked_buffer': watermark_path,
        'unwatermarked_detection_result': unwatermarked_detection_result,
        'watermarked_detection_result': watermarked_detection_result
    }

    try:
        metrics = combined_response['unwatermarked_detection_result']
        w_metrics = combined_response['watermarked_detection_result']
        return render_template('index.html', metrics=metrics, w_metrics=w_metrics, detect_metrics=metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_watermark', methods=['POST'])
def detect_watermark():
    if 'music_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    music_file = request.files['music_file']
    
    if music_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    

    encoder = BachChoralesEncoder()
    
    tempo = int(request.args['tempo'])
    
    chorale = TetradPlayer().from_wav(music_file, tempo = tempo)
    
    watermark_detector = WatermarkDetector(device = 'cpu', tokenizer= encoder)
    
    detection_result = watermark_detector.detect(text= torch.as_tensor(chorale.flatten()))
    
    response = {'detection_result': detection_result}

    # Call the watermark detection API
    try:
        metrics = response['detection_result']
        return render_template('index.html', detect_metrics=metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
