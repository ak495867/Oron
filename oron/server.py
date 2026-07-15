from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn

from oron import Oron
from oron.adapters.groq import GroqAdapter

app = FastAPI(title="Oron API", version="0.2.0", description="REST API for Oron")

# In a real production setting, you'd manage instances dynamically.
# For this server mode, we'll keep a cache of active Oron instances.
instances: Dict[str, Oron] = {}


def get_instance(user_id: str) -> Oron:
    if user_id not in instances:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, detail="GROQ_API_KEY environment variable not set."
            )

        adapter = GroqAdapter(api_key=api_key)
        # Using a dedicated directory for the server
        db_dir = os.environ.get("MEMORYOS_DB_DIR", "./oron_server_data")

        instances[user_id] = Oron(
            user_id=user_id,
            db_dir=f"{db_dir}/{user_id}",
            use_brain=True,
            adapter=adapter,
        )
    return instances[user_id]


class ChatRequest(BaseModel):
    user_id: str
    prompt: str
    model: Optional[str] = "llama-3.3-70b-versatile"


class ChatResponse(BaseModel):
    response: str


class RememberRequest(BaseModel):
    user_id: str
    text: str


class RecallRequest(BaseModel):
    user_id: str
    query: str
    limit: Optional[int] = 5


class RecallResponse(BaseModel):
    memories: List[str]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    High-level chat endpoint. Recalls context, generates response, and asynchronously ingests the prompt.
    """
    mem = get_instance(req.user_id)
    try:
        # Since Oron.achat is async, we can await it directly.
        response = await mem.achat(req.prompt, model=req.model)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remember")
async def remember_endpoint(req: RememberRequest, background_tasks: BackgroundTasks):
    """
    Manually inject a fact or memory. Processed asynchronously in the background.
    """
    mem = get_instance(req.user_id)
    background_tasks.add_task(mem.aremember, req.text)
    return {"status": "Processing in background"}


@app.post("/recall", response_model=RecallResponse)
def recall_endpoint(req: RecallRequest):
    """
    Search the memory stores for a specific query.
    """
    mem = get_instance(req.user_id)
    memories = mem.recall(req.query, limit=req.limit)
    return RecallResponse(memories=memories)


@app.post("/consolidate")
def consolidate_endpoint(user_id: str):
    """
    Trigger the consolidation worker for a specific user.
    """
    mem = get_instance(user_id)
    promoted = mem.consolidate()
    return {"status": "success", "promoted_facts": promoted}


def serve(host: str = "0.0.0.0", port: int = 8765):
    """Run the FastAPI server via Uvicorn."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
