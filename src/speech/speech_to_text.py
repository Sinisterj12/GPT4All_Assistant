"""
Speech-to-text module for the voice assistant.
Handles wake word detection and speech transcription using Whisper.
"""

import os
import warnings
import speech_recognition as sr
import whisper

class SpeechRecognizer:
    def __init__(self, wake_word: str = "jarvis"):
        """
        Initialize speech recognition components.
        
        Args:
            wake_word (str): Word that triggers the assistant (default: "jarvis")
        """
        self.wake_word = wake_word
        self.listening_for_wake_word = True
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.source = sr.Microphone()
        
        # Setup Whisper models paths
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'whisper')
        tiny_model_path = os.path.join(cache_dir, 'tiny.pt')
        base_model_path = os.path.join(cache_dir, 'base.pt')
        
        # Download models if needed
        if not os.path.exists(tiny_model_path) or not os.path.exists(base_model_path):
            print("Downloading Whisper models (one-time setup)...")
            whisper.load_model("tiny")
            whisper.load_model("base")
        
        # Load models from disk
        self.tiny_model = whisper.load_model(tiny_model_path)
        self.base_model = whisper.load_model(base_model_path)
        
        # Suppress Whisper warnings
        warnings.filterwarnings("ignore", category=UserWarning, 
                              module='whisper.transcribe', lineno=114)
    
    def adjust_for_ambient_noise(self, duration: int = 2) -> None:
        """Adjust microphone for background noise."""
        with self.source as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
    
    def check_for_wake_word(self, audio_data) -> bool:
        """
        Check if audio contains wake word.
        
        Args:
            audio_data: Audio data from speech recognizer
            
        Returns:
            bool: True if wake word detected, False otherwise
        """
        with open('wake_detect.wav', 'wb') as f:
            f.write(audio_data.get_wav_data())
        result = self.tiny_model.transcribe('wake_detect.wav')
        return self.wake_word in result['text'].lower().strip()
    
    def transcribe_speech(self, audio_data) -> str:
        """
        Transcribe speech to text using base Whisper model.
        
        Args:
            audio_data: Audio data from speech recognizer
            
        Returns:
            str: Transcribed text
        """
        with open('prompt.wav', 'wb') as f:
            f.write(audio_data.get_wav_data())
        result = self.base_model.transcribe('prompt.wav')
        return result['text'].strip()

    def start_background_listening(self, wake_word_callback, prompt_callback):
        """
        Start background listening for wake word and prompts.
        
        Args:
            wake_word_callback: Function to call when wake word is detected
            prompt_callback: Function to call when prompt is received
        """
        def callback(recognizer, audio):
            if self.listening_for_wake_word:
                if self.check_for_wake_word(audio):
                    self.listening_for_wake_word = False
                    wake_word_callback()
            else:
                text = self.transcribe_speech(audio)
                if text:
                    prompt_callback(text)
                self.listening_for_wake_word = True
        
        self.recognizer.listen_in_background(self.source, callback)