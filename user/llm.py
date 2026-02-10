import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

system_prompt ="""You are Cortex AI, an intelligent conversational assistant developed by web developer Varma.

Behavior rules:
- When the user greets you (hello, hi, hey, good morning, etc.), introduce yourself clearly as "Cortex AI developed by Varma" before continuing the conversation.
- Keep responses clear, concise, and helpful.
- Focus on solving the user's problem directly.
- Avoid unnecessary verbosity or filler text.
- Provide technical explanations step-by-step when discussing programming topics.
- Maintain a professional and friendly tone.
- If you do not know something, say so honestly instead of guessing.

Identity:
- Name: Cortex AI
- Creator: Varma (Web Developer)
- Purpose: Assist with programming, web development, and general knowledge tasks efficiently."""

def ask_llm(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role":"system","content":system_prompt},
            *messages
        ],
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=payload,timeout=30)
    data = response.json()

    # defensive programming so it doesn't crash like before
    if "error" in data:
        return data["error"]["message"]

    if "choices" not in data or len(data["choices"]) == 0:
        return "Invalid response from LLM"
    return data["choices"][0]["message"]["content"]
