import pytest
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock

from oron import Oron
from oron.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    def __init__(self):
        self.chat_history = []

    def chat(self, prompt, memories, system_prompt=None, **kwargs):
        self.chat_history.append(prompt)
        return "Mock response"

    async def achat(self, prompt, memories, system_prompt=None, **kwargs):
        self.chat_history.append(prompt)
        return "Mock async response"


@pytest.fixture
def mem_os(tmp_path):
    # Use a temporary directory for tests
    db_dir = str(tmp_path / "test_data")
    adapter = MockAdapter()
    mem = Oron(user_id="test_user", db_dir=db_dir, use_brain=False, adapter=adapter)
    return mem


def test_rule_based_ingestion_and_recall(mem_os):
    # Important fact
    mem_os.remember("My name is Pytest and I love testing.")

    # Noise (should be filtered by scorer)
    mem_os.remember("k")
    mem_os.remember("lol")

    memories = mem_os.recall("What is my name?")

    # Should recall the name, but not the noise
    assert any("My name is Pytest" in m for m in memories)
    assert not any("lol" in m for m in memories)
    assert not any("k" in m for m in memories)

    # Check identity block
    assert any("User Identity: The user's name is Pytest" in m for m in memories)


def test_chat_method_sync(mem_os):
    response = mem_os.chat("Hello there!")
    assert response == "Mock response"
    assert mem_os.adapter.chat_history == ["Hello there!"]


@pytest.mark.asyncio
async def test_chat_method_async(mem_os):
    response = await mem_os.achat("Hello async!")
    assert response == "Mock async response"
    assert mem_os.adapter.chat_history == ["Hello async!"]


def test_semantic_store_confidence(tmp_path):
    from oron.store.semantic import SemanticStore

    store = SemanticStore(path=str(tmp_path / "semantic.json"))
    user_id = "test_user"

    # Establish a fact
    store.add_fact(user_id, "user", "name", "Alice", {"confidence": 3})

    # Attempt to overwrite with lower confidence fact
    store.add_fact(user_id, "user", "name", "Bob", {"confidence": 1})

    facts = store.get_related(user_id, "user")

    # Should still be Alice
    assert any(f["object"] == "Alice" and f["relation"] == "name" for f in facts)
    assert not any(f["object"] == "Bob" for f in facts)

    # Overwrite with higher confidence
    store.add_fact(user_id, "user", "name", "Charlie", {"confidence": 5})
    facts_updated = store.get_related(user_id, "user")

    # Should now be Charlie
    assert any(
        f["object"] == "Charlie" and f["relation"] == "name" for f in facts_updated
    )
    assert not any(f["object"] == "Alice" for f in facts_updated)
