from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
def expand_query(user_query:str, n:int =4)->list[str]:
  prompt = f"""
  Generate {n} different search queries for the following user question.
  Make them diverse and cover different angles.
  - Do Not give any explanation
  - Only give questions
  - Do not add any other texts other than the questions

  Question: {user_query}
  """
  response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
          {"role": "user", "content": prompt},
      ],
      temperature=0.6,
  )
  queries = response.choices[0].message.content.strip().split("\n")
  return [q.strip("- ").strip() for q in queries if q.strip()]
