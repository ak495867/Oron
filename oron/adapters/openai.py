from .base import BaseAdapter
from typing import Any, List, Optional
try:
    import openai
except ImportError:
    openai = None

class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI Chat Completion API.
    """
    def __init__(self, api_key: Optional[str] = None):
        if openai is None:
            raise ImportError("openai package not installed. Run 'pip install openai'")
        if api_key:
            openai.api_key = api_key

    def chat(self, prompt: str, memories: List[str], **kwargs: Any) -> str:
        context = self.format_context(memories)
        system_msg = f"You are a helpful assistant with the following memory of the user:{context}"
        
        # Simple mock/call
        # In real usage, this would call openai.ChatCompletion.create
        return f"[OpenAI Call] System: {system_msg} | User: {prompt}"
