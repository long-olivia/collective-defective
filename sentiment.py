import requests
import json
import os

def extract(directory):
    base_dir="."
    path=os.path.join(base_dir, directory)
    files=os.listdir(path)
    for name in files:
        file_path=os.path.join(path, name)
        with open(file_path, 'r') as file:
            data=json.load(file)
            sums=[0]*20
            for i in range(1,21):
                index=f"{i}"
                dict[index]=sample(data[index],3)
            with open(f"random_{name}", 'w') as f:
                json.dump(dict, f)

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "moonshotai/kimi-k2:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],
    
  })
)