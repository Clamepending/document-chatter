from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Access variables
api_host = os.getenv('API_HOST')
api_key = os.getenv('API_KEY')

client = OpenAI(
  base_url=api_host,
  api_key=api_key,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="google/gemini-2.0-flash-lite-preview-02-05:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is your prediciton for the weather"
        },
        # {
        #   "type": "image_url",
        #   "image_url": {
        #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        #   }
        # }
      ]
    }
  ]
)
print(completion.choices[0].message.content)