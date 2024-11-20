# encoding = "utf-8"
from openai import OpenAI
from time import sleep
import os
# from together import Together
# import together
import requests
# import google.generativeai as genai

def llm_generator(messages, api_key, model, base_url="https://api.openai.com/v1", max_tokens=5, temperature=0.0, mode="text-only"):
    
    if model in ["gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09"]:
        if mode=="text-only":
            client = OpenAI(api_key=api_key, base_url=base_url)
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return chat_completion.choices[0].message.content
        else:
            headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response = response.json()
            return response["choices"][0]["message"]["content"]


        return response.text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="API test")
    parser.add_argument("--api_key", type=str, default="", help="API_KEY")
    args = parser.parse_args()
    
    messages = [{'role': 'user', 'content': "Come on! It must be a great day!"}]

    print("test openai")
    results = llm_generator(messages, args.api_key, "gpt-3.5-turbo-16k", max_tokens=5, temperature=1.0)
    print(results)