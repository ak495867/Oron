from typing import List, Any
from pydantic import Field
from ..core import Oron

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except ImportError:
    # Dummy classes for when langchain is not installed
    class BaseChatMessageHistory:
        pass

    class BaseRetriever:
        pass

    class Document:
        pass

    class BaseMessage:
        pass

    class HumanMessage:
        pass

    class AIMessage:
        pass


class OronRetriever(BaseRetriever):
    """
    LangChain Retriever that uses Oron to recall context.
    Provides semantic and episodic memories as LangChain Documents.
    """

    memory_os: Any = Field(description="Oron instance")
    k: int = Field(default=5, description="Number of memories to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        memories = self.memory_os.recall(query, limit=self.k)
        return [Document(page_content=m) for m in memories]


class OronChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain Chat Message History backed by Oron.
    Automatically ingests User messages into Oron for long-term storage.
    Note: Keeps an in-memory short-term transcript for the current session.
    """

    memory_os: Any
    _messages: List[BaseMessage] = []

    def __init__(self, memory_os: Oron):
        super().__init__()
        self.memory_os = memory_os
        self._messages = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

        # We only want to autonomously ingest the user's thoughts/facts
        if isinstance(message, HumanMessage):
            # Non-blocking ingestion if memory_os has an executor
            if hasattr(self.memory_os, "executor"):
                self.memory_os.executor.submit(
                    __import__("asyncio").run, self.memory_os.aremember(message.content)
                )
            else:
                self.memory_os.remember(message.content)

    def clear(self) -> None:
        self._messages = []
