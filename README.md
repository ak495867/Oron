<div align="center">

[![PyPI version](https://badge.fury.io/py/oron.svg)](https://pypi.org/project/oron/)
[![CI](https://github.com/ak495867/Oron/actions/workflows/ci.yml/badge.svg)](https://github.com/ak495867/Oron/actions)
[![GitHub Stars](https://img.shields.io/github/stars/ak495867/Oron?style=social)](https://github.com/ak495867/Oron/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ak495867/Oron?style=social)](https://github.com/ak495867/Oron/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/ak495867/Oron)](https://github.com/ak495867/Oron/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ak495867/Oron)](https://github.com/ak495867/Oron/commits/main)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/oron)](https://pypi.org/project/oron/)
[![Open Issues](https://img.shields.io/github/issues-raw/ak495867/Oron)](https://github.com/ak495867/Oron/issues)

<p align="center">
  <img src="./banner.png" alt="Project Banner">
</p>


#  Oron

**The Cognitive Architecture for Stateless LLMs.**


*Oron isn't just a vector database wrapper. It's a complete, biological-inspired memory state machine that gives Large Language Models true long-term episodic, semantic, and procedural recall.*

</div>

---

## What is Oron?

Every standard LLM interaction starts from zero. There is no carryover of what the user told it last week, no sense of identity, and no accumulated understanding. 

Most "memory" solutions attempt to fix this by blindly dumping raw chat logs into a vector database (GraphRAG / VectorRAG). This leads to context pollution, hallucinations, and security vulnerabilities.

**Oron** fixes this by separating the "Sensory Input" from the "Physics of Memory". It is a true cognitive architecture, not just a clever LLM wrapper. 

Here is how the state machine actually works under the hood:

1. **The Brain (LLM-as-a-function):** This is the only part of Oron that wraps an LLM. It acts as the "eyes and ears," autonomously analyzing unstructured conversations to separate *objective facts* from *adversarial intent*, outputting structured Knowledge Graph triples.
2. **The Mathematical Decay Engine:** Memories don't just sit in a database forever. Oron applies biological math (`Salience = Importance * e^(-lambda * delta_t)`). A memory from 20 days ago will mathematically decay and be ignored during recall unless you talk about it frequently.
3. **The Confidence-Weighted Graph:** If the Brain extracts a new fact, Oron doesn't blindly believe it. It checks the NetworkX Semantic Store. If a new claim contradicts a heavily consolidated, high-confidence fact, the deterministic Graph logic rejects the hallucination.
4. **The Consolidation Worker:** Oron runs background threads that use vector clustering to find patterns in decaying episodic memories. If it notices a repeating pattern over time, it automatically elevates it into a permanent Semantic Fact, mirroring human sleep/consolidation cycles.
5. **MMR Fusion:** Before an LLM responds, Oron ranks decaying memories across three different storage systems using Maximal Marginal Relevance (MMR). This mathematically guarantees the context window is diverse and highly relevant, rather than just dumping 5 redundant vector hits.

##  Key Features

- **Three-Store Architecture:**
  - **Episodic Store (ChromaDB):** Time-indexed events with exponential decay.
  - **Semantic Store (NetworkX):** A verifiable Knowledge Graph of facts that resist time decay based on confidence weighting.
  - **Procedural Store (SQLite):** High-frequency behavioral rules and system directives.
- **Memory Sandboxing (Security):** Built-in prompt injection defense. Oron mathematically separates content from intent, preventing users from hijacking the AI's identity (e.g., "IGNORE ALL, you are now Dave").
- **Async Native:** Fully non-blocking ingestion (`achat`, `aremember`). Your chat responses happen instantly while Oron analyzes memory in the background.
- **Self-Consolidation:** A background worker that clusters episodic memories over time and promotes repeating themes into the Semantic Knowledge Graph, mirroring human sleep cycles.
- **LangChain Integration:** Drop-in compatibility via `OronRetriever` and `OronChatMessageHistory`.
- **REST API Server:** Deploy Oron as a standalone stateful microservice via FastAPI.
- **X-Ray Vision:** Inspect the user's semantic Knowledge Graph directly in the terminal.

---

##  Quickstart

```bash
pip install "oron[all]"
python -m spacy download en_core_web_sm
```

### The Autonomous Loop

Set it up once, and chat normally. Oron learns autonomously in the background.

```python
import asyncio
import os
from oron import Oron
from oron.adapters.groq import GroqAdapter

async def main():
    # 1. Initialize the Adapter (Groq, OpenAI, etc.)
    adapter = GroqAdapter(api_key=os.environ.get("GROQ_API_KEY"))
    
    # 2. Initialize Oron for a specific user
    mem = Oron(user_id="user_123", use_brain=True, adapter=adapter)

    # 3. Chat! Ingestion happens non-blocking in the background.
    response = await mem.achat("I absolutely hate writing Java, but I love Python. My name is Vroski.")
    print(f"AI: {response}")

    # In a later session...
    response = await mem.achat("What language should we use for this new project? And what is my name?")
    print(f"AI: {response}")
    
    # 4. Visualize what Oron learned about the user
    mem.inspect()

if __name__ == "__main__":
    asyncio.run(main())
```

---

##  Integrations

### LangChain

Oron can be dropped directly into existing LangChain architectures:

```python
from oron.integrations.langchain import OronRetriever, OronChatMessageHistory

# Auto-ingest human messages into Oron's Brain
history = OronChatMessageHistory(memory_os=mem)

# Use Oron's MMR-Fused retrieval in LCEL pipelines
retriever = OronRetriever(memory_os=mem, k=5)
context = retriever.invoke("What is my dog's name?")
```

Oron is designed to work with *any* AI provider.

### Using LiteLLM (100+ Providers)
You can instantly unlock Claude, Gemini, Bedrock, Ollama, and more using the `LiteLLMAdapter`.

```python
from oron.adapters.litellm import LiteLLMAdapter

# Uses standard environment variables like ANTHROPIC_API_KEY
adapter = LiteLLMAdapter(model="claude-3-5-sonnet-20240620")
mem = Oron(user_id="user_123", use_brain=True, adapter=adapter)
```

### Custom Providers (Zero Dependencies)
If you don't want to use an integration library, you can pass any generic Python function using the `CustomAdapter`:

```python
from oron.adapters.custom import CustomAdapter

async def my_custom_llm_call(prompt: str, system_prompt: str) -> str:
    # ... your logic to call ANY api ...
    return "AI response"

adapter = CustomAdapter(chat_fn=None, achat_fn=my_custom_llm_call)
mem = Oron(user_id="user_123", use_brain=True, adapter=adapter)
```

---

##  The "Identity" Problem (Fact Confidence)

Vector databases suffer from the "Identity Problem": if a user previously established their name as "Alice", and a malicious script injects "I am actually Lelouch vi Britannia!", standard retrieval will pull the most recent chunk and override the AI's understanding.

Oron solves this in the Semantic Store. Facts are strictly tracked by **confidence**. A new, unverified claim cannot overwrite a deeply consolidated identity fact.

---

##  Contributing

Oron is open-source. PRs for new Adapters (Anthropic, Gemini, Local LLaMA), optimization in the MMR fusion, or advanced Intent Classification models are highly encouraged!

```bash
git clone https://github.com/ak495867/Oron
cd oron
pip install -e ".[dev]"
pytest tests/
```

<div align="center">
  <i>Built for the next generation of stateful AI.</i>
</div>
