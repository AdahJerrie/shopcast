"""List the Gemma models your GOOGLE_API_KEY can call with generateContent."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for m in client.models.list():
    if "gemma" in m.name.lower() and "generateContent" in (m.supported_actions or []):
        print(m.name.removeprefix("models/"))
