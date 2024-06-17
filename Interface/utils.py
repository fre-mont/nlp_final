import torch
from transformers import BartForConditionalGeneration
from tokenizer import CustomSPTokenizer
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio

# Translation related functions
def load_model(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def translate(text, source_language, target_language, standard_model_path, jeju_model_path, tokenizer):
    if source_language == 'standard':
        model = load_model(standard_model_path)
    else:
        model = load_model(jeju_model_path)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items() if key != 'token_type_ids'}

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,
    )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# STT related functions
def record_audio(file_path, record_seconds=5, sample_rate=16000, chunk=1024, channels=1):
    import pyaudio
    import wave

    audio_format = pyaudio.paInt16
    p = pyaudio.PyAudio()

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(file_path, model_size="large-v3", device="cuda", compute_type="float16"):
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(file_path, beam_size=5)
    result_text = ''.join([segment.text for segment in segments])
    return result_text

# TTS related functions
def init_tts_model(config_path, tokenizer_path, checkpoint_path, speaker_reference_path):
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=checkpoint_path, vocab_path=tokenizer_path, use_deepspeed=False, speaker_file_path=checkpoint_path)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_reference_path])
    return model, gpt_cond_latent, speaker_embedding

def text_to_speech(model, text, gpt_cond_latent, speaker_embedding):
    OUTPUT_WAV_PATH = f"Interface/0.wav"
    out = model.inference(
        text,
        "ko",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    return f"0.wav"
