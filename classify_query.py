from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)



def classify_query(query):
    prompt = f"""
    Classify the query into ONE of these:
    - FACT
    - LIST
    - COUNT
    - MULTI
    - REASONING

    Query: {query}
    Only return the label.
    """
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()
print(classify_query("what does article 246 states"))