# Oron Usage Guide

This guide covers all the core functionalities of the Oron Cognitive Architecture, from basic autonomous chat to manual memory manipulation and graph visualization.

---

## 1. Initialization and Adapters

Oron is provider-agnostic. You must provide an `Adapter` to tell Oron which LLM to use for its "Brain" (the background ingestion engine) and for generating chat responses.

```python
import os
from oron import Oron
from oron.adapters.groq import GroqAdapter
from oron.adapters.litellm import LiteLLMAdapter

# Using Groq directly
adapter = GroqAdapter(api_key=os.environ.get("GROQ_API_KEY"))

# Or using LiteLLM to access Anthropic, OpenAI, Gemini, etc.
# adapter = LiteLLMAdapter(model="claude-3-5-sonnet-20240620")

# Initialize Oron for a specific user
mem = Oron(
    user_id="user_123", 
    db_dir="./my_memory_data", # Where the SQLite/ChromaDB files live
    use_brain=True,            # Enable the LLM-driven cognitive engine
    adapter=adapter
)
```

---

## 2. Autonomous Chat (The Main Loop)

The easiest way to use Oron is via the `achat` (async) or `chat` (sync) methods. 

When you use these methods, Oron does three things automatically:
1. **Recalls** relevant memories, decays them, fuses them via MMR, and injects them into the prompt.
2. **Generates** a response using the provided Adapter.
3. **Ingests** the user's prompt in the background (extracting facts, detecting intent, scoring importance).

```python
import asyncio

async def chat_loop():
    # Turn 1: Oron learns this autonomously in the background
    response = await mem.achat("I absolutely hate writing Java, but I love Python.")
    print(response)

    # Turn 2: Oron automatically recalls the preference
    response = await mem.achat("What language should we use for our new backend?")
    print(response) # AI will suggest Python and avoid Java

asyncio.run(chat_loop())
```

---

## 3. Manual Memory Management

Sometimes you want to directly manipulate the memory stores without generating a chat response.

### Storing Memory
You can manually inject a memory. This will still pass through the Brain for analysis, fact extraction, and security sandboxing.

```python
# Async (Recommended: non-blocking)
await mem.aremember("My dog's name is Barnaby.")

# Sync (Will block the thread until analysis is complete)
mem.remember("I am allergic to peanuts.")
```

### Retrieving Memory
You can manually query the engine. This runs the Hybrid Search + Biological Decay + MMR Fusion pipeline and returns a list of context strings.

```python
# Limit the context window to the top 3 most relevant, diverse hits
context = mem.recall("What is my dog's name?", limit=3)

for memory in context:
    print(memory)
```

---

## 4. Visualizing the Knowledge Graph (X-Ray)

Oron stores verified facts in a NetworkX graph (the Semantic Store). You can render this graph directly in your terminal to see exactly what the AI has learned about the user.

```python
# Prints a rich tree-view of all extracted entities and their relationships
mem.inspect()
```
*Example Output:*
```text
MemoryOS Knowledge Graph (User: user_123)
├── [CLAIM] user
│   ├── -> loves -> Python (conf: 1)
│   └── -> hates -> Java (conf: 1)
└── [CLAIM] Barnaby
    └── -> is -> dog (conf: 1)
```

---

## 5. Memory Consolidation

Oron models human sleep cycles. If a user mentions a concept multiple times across different days, those fading "Episodic" memories should be promoted to permanent "Semantic" facts. 

You can trigger this worker manually or set it on a cron job.

```python
# Analyzes the vector database for clusters of repeating information
# and uses the Brain to synthesize them into permanent Knowledge Graph edges.
promoted_count = mem.consolidate()

print(f"Promoted {promoted_count} new facts to long-term memory.")
```

---

## 6. Advanced Integrations

### Using Oron with LangChain
If you already have a LangChain application (LCEL), you can drop Oron in as your history and retriever backend.

```python
from oron.integrations.langchain import OronRetriever, OronChatMessageHistory

# 1. Use Oron as the Message History
# This automatically intercepts Human messages and sends them to Oron's Brain
history = OronChatMessageHistory(memory_os=mem)
history.add_message(HumanMessage(content="I live in Berlin."))

# 2. Use Oron as a Retriever in your LCEL chain
retriever = OronRetriever(memory_os=mem, k=5)
documents = retriever.invoke("Where do I live?")
```

### Running the REST API
If you are building a React/Next.js frontend or a distributed microservice architecture, you can run Oron as a standalone server.

1. Start the server:
```bash
export GROQ_API_KEY="your_api_key"
python -m oron.server
```

2. Interact via HTTP:
```bash
# Chat (Auto-ingests in the background)
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "prompt": "My favorite color is green."}'

# Recall Context
curl -X POST http://localhost:8765/recall \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "query": "What is my favorite color?"}'
```
