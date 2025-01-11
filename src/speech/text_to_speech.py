"""
Text-to-speech module for the voice assistant.
Handles all text-to-speech conversion using pyttsx3.
"""

import pyttsx3

# Initialize text-to-speech engine for Windows
engine = pyttsx3.init()

def speak(text: str) -> None:
    """
    Convert text to speech and play it.
    
    Args:
        text (str): The text to be converted to speech
        
    Example:
        speak("Hello, how can I help you today?")
    """
    engine.say(text)
    engine.runAndWait()

# Optional: You can customize the voice settings here
def setup_voice(rate: int = 150, volume: float = 1.0) -> None:
    """
    Configure the text-to-speech voice settings.
    
    Args:
        rate (int): Speaking rate (words per minute). Default is 150.
        volume (float): Volume level from 0 to 1. Default is 1.0.
    """
    engine.setProperty('rate', rate)    
    engine.setProperty('volume', volume)