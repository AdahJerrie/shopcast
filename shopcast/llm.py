"""Gemma text generation, either hosted on Google AI Studio or local through Ollama."""

from __future__ import annotations

import json
import os
import urllib.request


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "google").lower()


def model_name() -> str:
    if provider() == "ollama":
        return os.getenv("OLLAMA_MODEL", "gemma3:1b")
    return os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def generate(prompt: str, temperature: float = 0.2) -> str:
    if provider() == "ollama":
        return _ollama(prompt, temperature)
    return _google(prompt, temperature)


_client = None


def _google(prompt: str, temperature: float) -> str:
    global _client
    from google import genai
    from google.genai import types

    if _client is None:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (from aistudio.google.com) or LLM_PROVIDER=ollama.")
        _client = genai.Client(api_key=key)

    # Gemma on AI Studio rejects system instructions, so everything goes in the user turn.
    resp = _client.models.generate_content(
        model=model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return resp.text or ""


def _ollama(prompt: str, temperature: float) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    body = json.dumps(
        {
            "model": model_name(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"]
