from oron import Oron
from oron.adapters.groq import GroqAdapter
import os

def main():
    # Ensure GROQ_API_KEY is set
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable.")
        return

    # Initialize Oron and Groq Adapter
    mem = Oron(user_id="ak_01", db_dir="./test_mem_data")
    adapter = GroqAdapter(api_key=api_key)

    print("--- Oron Groq Test ---")
    print("Ask me anything. I will remember our conversation autonomously.")
    print("(Type 'exit' to quit)\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        # Oron handles retrieval, injection, and ingestion automatically
        response = mem.chat(prompt, adapter=adapter)
        
        print(f"\nAI: {response}\n")

if __name__ == "__main__":
    main()
