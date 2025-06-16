from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def use_groq(system_prompt, context, model):
    # Create non-streaming chat completion
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        temperature=0.2,
        max_completion_tokens=8192,
        top_p=0.95,
        stream=False,  # Disable streaming
        model=model,
    )
    
    # Extract and print response
    response = chat_completion.choices[0].message.content
    response = response.split("</think>")[1]
    return response

if __name__ == "__main__":
    print(use_groq("You are a helpful assistant.", "Hello World", "qwen/qwen3-32b"))