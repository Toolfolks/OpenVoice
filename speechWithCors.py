import os
import shutil
import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS, cross_origin
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Get the current directory and set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONFIG = os.path.join(BASE_DIR, 'checkpoints_v2', 'converter', 'config.json')
OUTPUTS_DIR = 'outputs_v2'
OUTPUTS_DIR_CONFIG = os.path.join(OUTPUTS_DIR, 'converter')

# Create outputs directory if it doesn't exist
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)
    logging.info(f"Created outputs directory: {OUTPUTS_DIR}")

# Create converter directory if it doesn't exist
if not os.path.exists(OUTPUTS_DIR_CONFIG):
    os.makedirs(OUTPUTS_DIR_CONFIG)
    logging.info(f"Created converter directory: {OUTPUTS_DIR_CONFIG}")

# Copy existing config file if it doesn't exist in outputs
CONFIG_FILE = os.path.join(OUTPUTS_DIR_CONFIG, 'config.json')
if not os.path.exists(CONFIG_FILE) and os.path.exists(SOURCE_CONFIG):
    shutil.copy2(SOURCE_CONFIG, CONFIG_FILE)
    logging.info(f"Copied config file from {SOURCE_CONFIG} to {CONFIG_FILE}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/convert', methods=['POST', 'OPTIONS'])
@cross_origin()
def convert():
    logging.info(f"Received request: {request.method} {request.path}")
    
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        logging.info(f"Request data: {data}")
        
        if not data:
            logging.error("Invalid input: No JSON data")
            return jsonify({'error': "Invalid input"}), 415
        
        text = data.get('text')
        language = data.get('language', 'EN')
        speed = data.get('speed', 1.0)

        if not text:
            logging.error("Text is required")
            return jsonify({'error': 'Text is required'}), 400

        logging.info(f"Processing text: {text}")
        
        try:
            # Initialize TTS with config file path
            logging.info(f"Initializing TTS with config file: {CONFIG_FILE}")
            tts = TTS(config_path=CONFIG_FILE, language=language)
            
            # ... rest of your audio processing code ...
            
        except Exception as e:
            logging.error(f"Error during audio processing: {str(e)}")
            return jsonify({'error': f"Audio processing error: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Error during request handling: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
