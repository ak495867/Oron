from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.
    """
    @abstractmethod
    def chat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        pass

    @abstractmethod
    async def achat(self, prompt: str, memories: List[str], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        pass

    def format_context(self, memories: List[str]) -> str:
        if not memories:
            return ""
        return "\nRelevant history and preferences:\n" + "\n".join([f"- {m}" for m in memories])
