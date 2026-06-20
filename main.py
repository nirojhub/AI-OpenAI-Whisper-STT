"""
Project Name: AI-OpenAI-Whisper-STT
Author: Niroj Kumar Sahoo (nirojkumarsahoo55@gmail.com)
Copyright (c) 2026 Niroj Kumar Sahoo
License: MIT License 
Source: https://github.com/nirojhub/AI-OpenAI-Whisper-STT
Description: Main program that orchestrates the Whisper STT functionality.

"""

import os
import whisper
import torch
from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram

STT_SAMPLE_RATE = 16000  # Native Whisper format

print(torch.cuda.is_available())      # Should print: True
print(torch.cuda.get_device_name(0))  # Should print your GPU name

# Initialize Models Locally
print("Loading Whisper STT (Tiny model)...")
device = "cuda"
# Swappable with "base" or "small" for better accuracy but higher resource usage
stt_model = whisper.load_model("tiny").to(device)
# Pass the input audio path
audio_path = os.path.abspath("user_input.wav")
# STT Transcribe the saved WAV file
if os.path.exists(audio_path):
    result = stt_model.transcribe(audio_path)
    user_text = result["text"].strip()
    print(f"👤 Text transcribed: {user_text}")
                
    if not user_text:
        print("Could not understand the audio. Please try again.")