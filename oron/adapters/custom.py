from .base import BaseAdapter
from typing import Any, List, Optional, Callable, Awaitable
import asyncio


class CustomAdapter(BaseAdapter):
    """
    A highly flexible adapter that allows you to use ANY AI provider.
    Simply pass your own functions for generating responses.
    """

    def __init__(
        self,
        chat_fn: Callable[[str, str], str],
        achat_fn: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        """
        :param chat_fn: A sync function that takes (prompt, system_prompt) and returns a string response.
        :param achat_fn: An async function that takes (prompt, system_prompt) and returns a string response.
        """
        self.chat_fn = chat_fn
        self.achat_fn = achat_fn

    def chat(
        self,
        prompt: str,
        memories: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        context = self.format_context(memories)
        final_system_prompt = system_prompt or (
            "You are a helpful assistant with a long-term memory system. "
            "Use the provided context to personalize your responses. "
            f"{context}"
        )
        return self.chat_fn(prompt, final_system_prompt)

    async def achat(
        self,
        prompt: str,
        memories: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        context = self.format_context(memories)
        final_system_prompt = system_prompt or (
            "You are a helpful assistant with a long-term memory system. "
            "Use the provided context to personalize your responses. "
            f"{context}"
        )

        if self.achat_fn:
            return await self.achat_fn(prompt, final_system_prompt)
        else:
            # Fallback to running the sync function in a thread if no async function provided
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.chat_fn, prompt, final_system_prompt
            )
