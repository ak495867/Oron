from .base import BaseAdapter
from typing import Any, List, Optional, Dict
import os

try:
    import litellm
    from litellm import completion, acompletion
except ImportError:
    litellm = None

class LiteLLMAdapter(BaseAdapter):
    """
    Adapter for LiteLLM, supporting 100+ LLM APIs.
    (e.g., Anthropic, Gemini, OpenAI, Azure, Ollama, HuggingFace)
    """
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """
        :param model: The LiteLLM model string (e.g., 'claude-3-sonnet-20240229', 'gemini/gemini-1.5-pro', 'ollama/llama3')
        :param api_key: Optional API key. Usually LiteLLM picks this up from env vars (e.g., ANTHROPIC_API_KEY).
        """
        if litellm is None:
            raise ImportError("litellm package not installed. Run 'pip install litellm'")
        
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def _build_messages(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        context = self.format_context(memories)
        final_system_prompt = system_prompt or (
            "You are a helpful assistant with a long-term memory system. "
            "Use the provided context to personalize your responses. "
            f"{context}"
        )
        return [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": prompt}
        ]

    def chat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, memories, system_prompt)
        
        # Merge init kwargs with call kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
            
        response = completion(model=self.model, messages=messages, **call_kwargs)
        return response.choices[0].message.content

    async def achat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, memories, system_prompt)
        
        call_kwargs = {**self.kwargs, **kwargs}
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
            
        response = await acompletion(model=self.model, messages=messages, **call_kwargs)
        return response.choices[0].message.content
