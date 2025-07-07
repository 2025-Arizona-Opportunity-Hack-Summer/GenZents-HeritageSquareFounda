import os
from openai import OpenAI
from typing import List, Dict

MODEL = "llama3-8b-8192"  # or "mixtral-8x7b-32768"

def generate_answer_with_llm(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Tool: Uses Groq LLM to generate a natural language answer based on Drive content.
    """

    if not retrieved_chunks:
        return "Sorry, I couldn't find anything relevant in your Drive to answer that."

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
    )

    messages = [
        {"role": "system", "content": "You are an intelligent assistant..."},
        {"role": "user", "content": f"User Question: {query}\n\nRelevant Drive Documents:\n{context}\n\nAnswer clearly."}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content