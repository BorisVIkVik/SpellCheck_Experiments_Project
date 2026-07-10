import os

from openai import OpenAI

api_key = 'fake'
url = "https://foundation-models.api.cloud.ru/v1"

client = OpenAI(
    api_key=api_key,
    base_url=url
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=2500,
    temperature=0.5,
    presence_penalty=0,
    top_p=0.95,
    messages=[
        {
            "role": "user",
            "content":"Как написать хороший код?"
        }
    ]
)

print(response.choices[0].message.content)