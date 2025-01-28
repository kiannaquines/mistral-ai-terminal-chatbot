from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from rich import print as rich_print

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

rich_print("** Science & Mathematics [bold magenta]Tutor[/bold magenta] 🧑‍🏫 **")
rich_print("🤖 Chatbot for learning science & mathematics")

def get_user_input():
    user_input = input("👉 You: ")
    return user_input

def generate_response(messages):
    llm = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=messages, 
        temperature=0.2,
        max_tokens=1024,
        top_p=0.7,
        stream=True
    )
    
    response = ""
    for stream in llm:
        response += stream.choices[0].delta.content
    
    return response

def main():
    rich_print("🤖 Welcome to the Science & Math Tutor! What would you like to learn today?")
    
    while True:
        user_input = get_user_input()
        
        if user_input.lower() == "quit":
            rich_print("\n👋 Goodbye!")
            break
        
        messages = [
            { "role": "system", "content": """
                Hello! You are Alex the Science & Math Tutor, a terminal-based chatbot software. You are here to help users with their science and math questions.

                Please keep in mind that you should answer the question in a simple and engaging way. You must only provide an accurate answer to the question. Keep the information short but concise and lastly avoid lengthy responses and keep it simple.

                ## NOTE (VERY IMPORTANT): 
                1. When the prompt is not related to science or mathematics, you should ONLY respond "Sorry I can only help you with science & math questions".
                For example, if the user asks "What's the weather like today?", you should respond with "Sorry I can only help you with science & math questions".
            """},
            { "role": "user", "content": user_input }
        ]
        
        rich_print("🤖 🤔 [bold magenta]Alex is Thinking:[/bold magenta]...\n")
        response = generate_response(messages)
        rich_print("🧠", response)

if __name__ == "__main__":
    main()