import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key from my .env file so it stays out of the code
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Quick sanity check to confirm the key works before I build anything bigger
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say hello and confirm you are working in one short sentence."}
    ],
)

print(response.choices[0].message.content)