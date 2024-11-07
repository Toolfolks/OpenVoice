import os
import torch
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import traceback
import io
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Check if GPU is available and set the device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Paths and configurations
ckpt_converter = 'checkpoints_v2/converter'
output_dir = 'outputs_v2'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load models and converters once at startup
tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
logging.info("Tone color converter loaded successfully.")

# Use john1.mp3 as the reference speaker (voice to clone)
reference_speaker = 'resources/john1.mp3'  # Path to the new voice sample
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
logging.info("Speaker embedding extracted successfully.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def stream_audio(file_path):
    """Generator function to stream audio in chunks."""
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(4096)  # Read in 4KB chunks
            if not chunk:
                break
            yield chunk

@app.route('/convert', methods=['POST'])
def convert_text_to_speech():
    try:
        # Parse input JSON
        data = request.get_json()
        if not data:
            logging.error("Invalid input: Content-Type must be 'application/json' and body must be valid JSON.")
            return jsonify({'error': "Invalid input. Content-Type must be 'application/json' and body must be valid JSON."}), 415
        
        text = data.get('text')
        language = data.get('language', 'EN')
        speed = data.get('speed', 1.0)
        speaker_id = data.get('speaker_id', 0)  # Add default speaker_id

        if not text or not isinstance(text, str):
            logging.error("Text is a required field and must be a valid string.")
            return jsonify({'error': 'Text is a required field and must be a valid string.'}), 400

        # Create temporary files
        output_dir = 'outputs_v2'  # Assuming this is defined earlier in your code
        temp_tts_file = os.path.join(output_dir, 'temp_tts_output.wav')
        
        # Generate TTS audio to temporary file
        model = TTS(language=language, device=device)
        model.tts_to_file(text, speaker_id, temp_tts_file, speed=speed)  # Use existing method
        
        # Check if the file was created and has content
        if not os.path.exists(temp_tts_file) or os.path.getsize(temp_tts_file) == 0:
            logging.error("No audio segments found! The TTS model did not generate any audio.")
            return jsonify({'error': 'No audio segments found! The TTS model did not generate any audio.'}), 500
        
        # Stream the audio file in chunks
        return Response(stream_audio(temp_tts_file), mimetype='audio/wav')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    finally:
        # Clean up temporary file after streaming
        if os.path.exists(temp_tts_file):
            os.remove(temp_tts_file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)