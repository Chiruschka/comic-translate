from typing import Any
import numpy as np
import requests
import json

from .base import BaseLLMTranslation


class CustomTranslation(BaseLLMTranslation):
    """
    Translation engine for local / custom OpenAI-compatible APIs.
    
    Works with:
    - llama.cpp (llama-server)
    - Ollama
    - LM Studio
    - Any OpenAI-compatible endpoint
    
    Key differences from cloud LLM translators:
    - Uses plain string message format (not array-of-objects)
    - Uses 'max_tokens' instead of 'max_completion_tokens'
    - Does NOT send images (most local models are text-only)
    - Longer timeout for slower local inference
    - No API key required (sends a dummy key for compatibility)
    """
    
    def __init__(self):
        super().__init__()
        self.api_base_url = ""
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, tr_key: str, **kwargs) -> None:
        """
        Initialize custom translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            tr_key: Translation key for looking up credentials
        """
        # Call BaseLLMTranslation's initialize (skip GPT-specific logic)
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        # Get custom credentials
        credentials = settings.get_credentials(settings.ui.tr(tr_key))
        self.api_key = credentials.get('api_key', '') or 'no-key-needed'
        self.model = credentials.get('model', '')
        
        # Override the API base URL with the custom one
        self.api_base_url = credentials.get('api_url', '').rstrip('/')
        
        # Local models can be slow, especially large ones on CPU
        # 600s = 10 minutes, enough for a 35B model on CPU
        self.timeout = 600
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using a local/custom OpenAI-compatible API.
        
        Uses plain string content format and max_tokens (not max_completion_tokens)
        for broad compatibility with llama.cpp, Ollama, and other local servers.
        
        Images are NOT sent — most local models are text-only.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Use plain string content format for broad compatibility.
        # Cloud APIs accept both formats, but local servers
        # (llama.cpp, Ollama) often only support plain strings.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after {self.timeout}s. "
                f"If your model is large, try a smaller quantization or "
                f"increase GPU layer offloading."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to {self.api_base_url}. "
                f"Make sure your local LLM server is running "
                f"(e.g. llama-server, ollama serve)."
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f" - {json.dumps(error_details)}"
                except Exception:
                    error_msg += f" - Status code: {e.response.status_code}"
            raise RuntimeError(error_msg)