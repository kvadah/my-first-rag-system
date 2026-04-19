from openAI import OpenAI



client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)



def generate_answer(context, question):
    prompt = f"""
You are a highly accurate technical assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT add external knowledge
- If the answer is not in the context, only say: "I am sorry I don't have enough knowldge about it "
- Be clear, structured, and professional

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "You are a precise RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content