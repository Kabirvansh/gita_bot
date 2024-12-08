import anthropic

client = anthropic.Anthropic(api_key='')
try:
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("API Connection Successful")
except Exception as e:
    print(f"Connection Error: {e}")