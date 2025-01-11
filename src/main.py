"""
Main module for the voice assistant.
Coordinates between speech recognition, text-to-speech, and GPT model components.
"""

import time
from speech.text_to_speech import speak, setup_voice
from speech.speech_to_text import SpeechRecognizer
from ai.gpt_model import GPTModel

def wake_word_detected():
    """Called when wake word is detected."""
    print('Wake word detected. Please speak your prompt to GPT4All.')
    speak('Listening')

def handle_prompt(text: str):
    """
    Handle user's spoken prompt.
    
    Args:
        text (str): Transcribed text from user's speech
    """
    if len(text.strip()) == 0:
        print('Empty prompt. Please speak again.')
        speak('Empty prompt. Please speak again.')
    else:
        print('User:', text)
        response = gpt_model.generate_response(text)
        print('GPT4All:', response)
        speak(response)
        print('\nSay "jarvis" to wake me up.\n')

def main():
    # Initialize components
    global gpt_model  # Make accessible to handle_prompt
    gpt_model = GPTModel(device="gpu")
    recognizer = SpeechRecognizer(wake_word="jarvis")
    
    # Optional: Setup voice properties
    setup_voice(rate=150, volume=1.0)
    
    # Adjust for background noise
    print("Adjusting for ambient noise...")
    recognizer.adjust_for_ambient_noise()
    
    # Start listening
    print('\nSay "jarvis" to wake me up.\n')
    recognizer.start_background_listening(
        wake_word_callback=wake_word_detected,
        prompt_callback=handle_prompt
    )
    
    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == '__main__':
    main()