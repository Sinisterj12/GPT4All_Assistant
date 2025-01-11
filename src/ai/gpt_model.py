"""
GPT model module for the voice assistant.
Handles all interactions with GPT4All model.
"""

import os
from gpt4all import GPT4All

class GPTModel:
    def __init__(self, device: str = "gpu"):
        """
        Initialize GPT4All model.
        
        Args:
            device (str): Device to run model on ("gpu" or "cpu"). Default is "gpu"
        """
        # Setup GPT4All model path for Windows
        model_complete_filepath = os.path.join(
            os.getenv('LOCALAPPDATA'), 
            'nomic.ai', 
            'GPT4All', 
            'ggml-model-gpt4all-falcon-q4_0.gguf'
        )
        model_path_directory, model_filename_complete = os.path.split(model_complete_filepath)
        model_filename, _ = os.path.splitext(model_filename_complete)
        
        # Initialize GPT4All with CUDA support and offline mode
        self.model = GPT4All(
            model_filename, 
            model_path=model_path_directory, 
            allow_download=False,  # Prevent online checks
            device=device
        )
    
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generate response from GPT model.
        
        Args:
            prompt (str): User's input text
            max_tokens (int): Maximum length of response. Default is 200
            
        Returns:
            str: Generated response from model
        """
        try:
            return self.model.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            print('GPT model error:', e)
            return "I apologize, but I encountered an error processing your request."