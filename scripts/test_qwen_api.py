from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:30000/v1"
)

response = client.chat.completions.create(
    model="/home/jiitcah.05/models/Qwen3-32B",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an information extraction engine. "
                "Never reveal reasoning. "
                "Never output <think>. "
                "Return only the final answer."
            ),
        },
        {
            "role": "user",
            "content": "Say hello."
        },
    ],
    temperature=0,
)

print(response.choices[0].message.content)
