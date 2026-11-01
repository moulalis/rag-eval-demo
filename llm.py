"""Tiny LLM client using only the Python standard library.

Pick a provider with the LLM_PROVIDER environment variable:
  ollama     (default) free, runs locally: https://ollama.com  e.g. `ollama pull llama3.1:8b`
  openai     needs OPENAI_API_KEY
  anthropic  needs ANTHROPIC_API_KEY
Set LLM_MODEL to choose the model; check your provider's docs for current model names.
"""
import json
import os
import urllib.request

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
DEFAULT_MODELS = {"ollama": "llama3.1:8b", "openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def ask(prompt: str) -> str:
    """Send one prompt to the chosen model and return its text reply."""
    if PROVIDER == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        data = _post(f"{host}/api/generate",
                     {"model": MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0}}, {})
        return data["response"].strip()
    if PROVIDER == "openai":
        data = _post("https://api.openai.com/v1/chat/completions",
                     {"model": MODEL, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return data["choices"][0]["message"]["content"].strip()
    if PROVIDER == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"model": MODEL, "max_tokens": 400, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                      "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in data["content"]).strip()
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
