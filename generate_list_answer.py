from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def generate_list_answer(question, context):
    system_message = """You are a precise legal information extractor and answer generator.

Your task has TWO steps:

STEP 1: Extract ALL relevant items from the context as a structured list.
- Each item must be concise and meaningful
- Do not miss any relevant item
- Do not include duplicates
- Do not hallucinate anything not in the context

STEP 2: Generate a final answer to the user's question using:
- the extracted list
- the provided context

STRICT RULES:
1. Do NOT use any external knowledge
2. Do NOT mention the context explicitly
3. Do NOT say "based on the context"
4. If no relevant items are found, respond ONLY:
   "I am sorry I don't have enough knowledge about it"

OUTPUT FORMAT:

<clear, complete answer based on the list and context with minimal citation>
"""

    user_prompt = f"""
Question:
{question}

Context:
{context}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  
        max_tokens=500
    )
    return response.choices[0].message.content
