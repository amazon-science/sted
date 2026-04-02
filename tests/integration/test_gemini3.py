import os
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPENAI_API_KEY", ""),
)

# First API call with reasoning
response = client.chat.completions.create(
  model="google/gemini-3-pro-preview",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}},
  temperature=0.0,
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

print(f"response: {response}")

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="openai/gpt-5.2-pro",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)
