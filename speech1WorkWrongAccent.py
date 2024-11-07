import os
import torch
from flask import Flask, request, jsonify
import logging
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Check if GPU is available and set the device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cuda:0":
    logging.info(f"CUDA is available. GPU will be used: {torch.cuda.get_device_name(0)}")
else:
    logging.info("CUDA is not available. Running on CPU.")

# Paths and configurations
ckpt_converter = 'checkpoints_v2/converter'
output_dir = 'outputs_v2'

# Initialize the tone color converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Use john1.mp3 as the reference speaker (voice to clone)
reference_speaker = 'resources/john1.mp3'  # Path to the new voice sample
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

# Initialize Flask app
app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_text_to_speech():
    try:
        # Parse input JSON
        data = request.get_json()
        if not data:
            logging.error("Invalid input: Content-Type must be 'application/json' and body must be valid JSON.")
            return jsonify({'error': "Invalid input. Content-Type must be 'application/json' and body must be valid JSON."}), 415
        
        text = data.get('text')
        language = data.get('language', 'EN_GB')
        speed = data.get('speed', 1.0)

        if not text:
            logging.error("Text is a required field.")
            return jsonify({'error': 'Text is a required field.'}), 400

        # Temporary path for generated audio file
        src_path = f'{output_dir}/tmp.wav'

        # Generate TTS audio
        try:
            model = TTS(language=language, device=device)
        except AssertionError:
            logging.error(f"Language '{language}' is not supported by the TTS model.")
            return jsonify({'error': f"Language '{language}' is not supported by the TTS model."}), 400

        speaker_ids = model.hps.data.spk2id
        speaker_key = list(speaker_ids.keys())[0]  # Use the first speaker as default
        speaker_id = speaker_ids[speaker_key]

        # Load specific speaker embedding
        speaker_key = speaker_key.lower().replace('_', '-')
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)

        # Define output file path
        save_path = f'{output_dir}/output.wav'

        # Convert tone color to match the target speaker (john1.mp3)
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)

        logging.info(f"Conversion successful, output saved to {save_path}")
        return jsonify({'message': 'Conversion successful', 'output_path': save_path}), 200

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
