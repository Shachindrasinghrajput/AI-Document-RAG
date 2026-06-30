import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def get_answer(question, context):
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the given context.
If the answer is not found in the context, say:
"I couldn't find the answer in the uploaded document."

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text