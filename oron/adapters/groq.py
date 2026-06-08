from .base import BaseAdapter
from typing import Any, List, Optional
import os

try:
    from groq import Groq, AsyncGroq
except ImportError:
    Groq = None
    AsyncGroq = None

class GroqAdapter(BaseAdapter):
    """
    Adapter for Groq API with Sync and Async support.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        if Groq is None:
            raise ImportError("groq package not installed. Run 'pip install groq'")
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
            
        self.client = Groq(api_key=self.api_key)
        self.aclient = AsyncGroq(api_key=self.api_key)
        self.model = model

    def chat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        # ... (sync code remains same)
        context = self.format_context(memories)
        
        final_system_prompt = system_prompt or (
            "You are a helpful assistant with a long-term memory system. "
            "Use the provided context to personalize your responses. "
            f"{context}"
        )
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=kwargs.get("model", self.model),
        )
        
        return response.choices[0].message.content

    async def achat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        context = self.format_context(memories)
        
        final_system_prompt = system_prompt or (
            "You are a helpful assistant with a long-term memory system. "
            "Use the provided context to personalize your responses. "
            f"{context}"
        )
        
        response = await self.aclient.chat.completions.create(
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=kwargs.get("model", self.model),
        )
        
        return response.choices[0].message.content
