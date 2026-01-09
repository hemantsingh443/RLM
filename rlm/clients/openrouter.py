"""
OpenRouter API Client for RLM Engine.

Handles communication with OpenRouter for root agent LLM calls.
"""

import os
import requests
from typing import List, Dict, Optional


class OpenRouterClient:
    """
    Client for making LLM API calls via OpenRouter.
    """
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "xiaomi/mimo-v2-flash:free",
        timeout: int = 180
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
            default_model: Default model to use for completions
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY env var not set")
        
        self.default_model = default_model
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-engine",
            "X-Title": "RLM Engine"
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            The assistant's response content
        
        Raises:
            requests.RequestException: On API errors
        """
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            self.BASE_URL,
            json=payload,
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            raise RuntimeError(f"OpenRouter API error: {result['error']}")
        
        return result['choices'][0]['message']['content']
    
    def chat_with_metadata(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Send a chat completion request and return full response with metadata.
        
        Returns:
            Full response dict including usage stats
        """
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            self.BASE_URL,
            json=payload,
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()
