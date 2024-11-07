import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Check if GPU is available and set the device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda:0":
    print(f"CUDA is available. GPU will be used: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

# Paths and configurations
ckpt_converter = 'checkpoints_v2/converter'
output_dir = 'outputs_v2'

# Initialize the tone color converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Use mam1.mp3 as the reference speaker (voice to clone)
reference_speaker = 'resources/john1.mp3'  # Path to the new voice sample
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

from melo.api import TTS

# Texts in different languages
texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",
    'EN': "Did you ever hear a folk tale about a giant turtle?",
    'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요。",
}

# Temporary path for generated audio file
src_path = f'{output_dir}/tmp.wav'

# Speed of the speech synthesis
speed = 1.0

# Process each language and text
for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    # Loop through each speaker in the language model
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        # Load specific speaker embedding
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        
        # Define output file path
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Convert tone color to match the target speaker (mam1.mp3)
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        
        print(f"Generated speech for {speaker_key} saved to: {save_path}")

# Print a final message indicating that processing is complete
print("Processing complete.")
if device == "cuda:0":
    print("The GPU was used for processing.")
else:
    print("The CPU was used for processing.")
