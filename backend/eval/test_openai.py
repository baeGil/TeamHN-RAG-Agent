import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

import openai
from app.config import get_settings
settings = get_settings()
print("Using API Key:", settings.openai_api_key[:15] + "...")
client = openai.OpenAI(api_key=settings.openai_api_key)
try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}]
    )
    print("Success:", resp.choices[0].message.content)
except Exception as e:
    print("Error type:", type(e))
    print("Error details:", e)
