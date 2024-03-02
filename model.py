from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model_id = "ft:gpt-3.5-turbo-0125:uchicago:uchi-large5:8yKA4g7Z"

client = OpenAI()

def create_chat_completion(model_id):
  completion = client.chat.completions.create(
    model=model_id,
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What are the requirements for the bio core?"}
    ]
  )
  return completion.choices[0].message.content

print(create_chat_completion(model_id))
