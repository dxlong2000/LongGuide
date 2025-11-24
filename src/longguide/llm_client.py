"""
LLM client for LongGuide framework.

Provides unified interface for different LLM providers (OpenAI, Claude, etc.)
with retry logic and error handling.
"""

import json
import time
import logging
from typing import Optional, Dict, Any
import openai
import boto3

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for different LLM providers."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.provider = self._detect_provider(model_name)
        self._setup_client()
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect LLM provider from model name."""
        if "gpt" in model_name.lower():
            return "openai"
        elif "claude" in model_name.lower():
            return "claude"
        else:
            return "openai"  # Default
    
    def _setup_client(self):
        """Setup client based on provider."""
        if self.provider == "openai":
            self.client = openai.OpenAI()
        elif self.provider == "claude":
            try:
                session = boto3.Session()
                self.client = session.client('bedrock-runtime', region_name='us-west-2')
            except Exception as e:
                logger.warning(f"Claude setup failed: {e}, falling back to OpenAI")
                self.provider = "openai"
                self.client = openai.OpenAI()
    
    def generate(self, prompt: str, max_tokens: int = 1024, 
                temperature: float = 0.7, max_retries: int = 3) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            
        Returns:
            Generated text
        """
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt, max_tokens, temperature)
                elif self.provider == "claude":
                    return self._generate_claude(prompt, max_tokens)
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All generation attempts failed")
                    return ""
        
        return ""
    
    def _generate_openai(self, prompt: str, max_tokens: int, 
                        temperature: float) -> str:
        """Generate using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _generate_claude(self, prompt: str, max_tokens: int) -> str:
        """Generate using Claude via AWS Bedrock."""
        response = self.client.invoke_model(
            modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
