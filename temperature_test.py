from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
MODEL ="llama-3.3-70b-versatile"
PROMPT = "Write a short story about a robot learning to love."
def ask_with_temperature(temperature):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
    )
    return response.choices[0].message.content
if __name__ == "__main__":
    for temp in [0.0, 0.7, 1.2]:
        print(f"Temperature: {temp}")
        print(ask_with_temperature(temp))
        print("\n" + "="*50 + "\n")