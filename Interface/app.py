from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
from utils import translate, record_audio, transcribe_audio, init_tts_model, text_to_speech
from tokenizer import CustomSPTokenizer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
CORS(app)

# Paths
standard_model_path = "model/kobart2_sort"
jeju_model_path = 'model/bartbase_model2'
tokenizer_path = "data/bpe_tokenizer.json"

# Tokenizer initialization
tokenizer = CustomSPTokenizer(tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.pad_token = '<pad>'

# TTS model initialization
CONFIG_PATH = "TTS/config.json"
TOKENIZER_PATH = "TTS/vocab.json"
XTTS_CHECKPOINT = "TTS/best_model.pth"
SPEAKER_REFERENCE = "TTS/data/10.wav"

tts_model, gpt_cond_latent, speaker_embedding = init_tts_model(CONFIG_PATH, TOKENIZER_PATH, XTTS_CHECKPOINT, SPEAKER_REFERENCE)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    text = data.get('text')
    source_language = data.get('source_language')
    target_language = data.get('target_language')

    if not text or not source_language or not target_language:
        return jsonify({'error': 'Missing text or language information'}), 400

    translated_text = translate(text, source_language, target_language, standard_model_path, jeju_model_path, tokenizer)
    return jsonify({'translated_text': translated_text})

@app.route('/voice-input', methods=['POST'])
def voice_input():
    record_seconds = int(request.form.get('record_seconds', 5))
    audio_file_path = f"{uuid.uuid4()}.wav"
    
    record_audio(audio_file_path, record_seconds=record_seconds)
    text = transcribe_audio(audio_file_path)
    
    os.remove(audio_file_path)  # 사용 후 파일 삭제
    return jsonify({'text': text})

@app.route('/text-to-speech', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    output_wav_path = text_to_speech(tts_model, text, gpt_cond_latent, speaker_embedding)
    print(output_wav_path)
    return send_file(output_wav_path, as_attachment=True, download_name=f"{text}.wav", mimetype="audio/wav")



@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


if __name__ == '__main__':
    app.run(debug=True)
