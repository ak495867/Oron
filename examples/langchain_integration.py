import os
import time
from oron import Oron
from oron.adapters.groq import GroqAdapter
from oron.integrations.langchain import OronRetriever, OronChatMessageHistory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

def test_langchain():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.")
        return

    print("--- Oron: LangChain Integration Test ---")

    # 1. Initialize Oron (we still need the adapter for the Brain to work)
    adapter = GroqAdapter(api_key=api_key)
    mem = Oron(user_id="lc_user", db_dir="./langchain_data", use_brain=True, adapter=adapter)

    # 2. Setup standard LangChain components
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
    
    # We use the Retriever to inject long-term context into the system prompt
    retriever = OronRetriever(memory_os=mem)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI. Here is the user's long-term memory context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    # 3. Use Oron as the Chat History backend
    # This automatically ingests human messages into Oron
    memory_history = OronChatMessageHistory(memory_os=mem)

    def get_session_history(session_id: str):
        # In a real app, you'd map session_id to different Oron instances
        return memory_history

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Turn 1: Establish a fact
    q1 = "Hi, my name is Alice and I am a quantum physicist."
    print(f"\nUser: {q1}")
    
    # We manually fetch context for the prompt
    context = "\n".join([d.page_content for d in retriever.invoke(q1)])
    
    response1 = with_message_history.invoke(
        {"question": q1, "context": context},
        config={"configurable": {"session_id": "1"}}
    )
    print(f"AI: {response1.content}")

    print("\nWaiting for background ingestion...")
    time.sleep(2)

    # Turn 2: Recall via LangChain Retriever
    q2 = "What is my profession?"
    print(f"\nUser: {q2}")
    
    context2 = "\n".join([d.page_content for d in retriever.invoke(q2)])
    print(f"[Debug] Injected Context: {context2}")
    
    response2 = with_message_history.invoke(
        {"question": q2, "context": context2},
        config={"configurable": {"session_id": "1"}}
    )
    print(f"AI: {response2.content}")

if __name__ == "__main__":
    test_langchain()
