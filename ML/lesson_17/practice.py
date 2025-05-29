import requests
import json


def generate_response(prompt):
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(
        "http://localhost:11434/api/generate", json=payload, stream=True
    ) as response:
        # print(f"Status Code: {response.status_code}")
        # print("\nResponse from LLaMA 3.2:")

        print("\nLLaMA 3.2 >>> ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                content = line.decode("utf-8").removeprefix("data: ")

                try:
                    chunk = json.loads(content)
                    print(chunk.get("response", ""), end="", flush=True)
                except Exception as e:
                    print(f"\n[Error parsion line]: {e}")

while True:
    prompt = input("\n>>> ")

    if prompt.lower() == "bye":
        break

    generate_response(prompt)