"""Client for Sovereign Sentinel via Ollama's native /api/generate endpoint.

Primary path: Ollama with the fine-tuned `sovereign-sentinel` model.
Fallback: OpenAI-compatible /v1/chat/completions for llama.cpp servers.
"""

import base64
import sys
import time
from pathlib import Path

import httpx

from config import AppConfig


class GemmaRuntime:
    def __init__(self, config: AppConfig):
        self.base_url = config.gemma.base_url.rstrip("/")
        self.model_name = config.gemma.model_name
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the LLM server is reachable (no exit)."""
        endpoints = ["/api/tags", "/v1/models", "/health"]
        for ep in endpoints:
            try:
                resp = self.client.get(ep)
                if resp.status_code == 200:
                    return True
            except httpx.ConnectError:
                pass
        return False

    def health_check(self) -> bool:
        """Verify Ollama / LLM server is reachable. Exits the process if not."""
        if self.is_available():
            return True
        print("\n" + "=" * 70)
        print("FATAL: Cannot connect to LLM runtime at", self.base_url)
        print()
        print("  Sovereign Sentinel requires Ollama:")
        print("       brew install ollama")
        print("       ollama serve                          # Terminal 1")
        print("       ./deploy_to_ollama.sh                 # Terminal 2  (first time)")
        print()
        print("       Set in .env:")
        print("         GEMMA_BASE_URL=http://localhost:11434")
        print("         GEMMA_MODEL_NAME=sovereign-sentinel")
        print()
        print("  Fallback — llama.cpp server:")
        print("       ./llama-server -m models/sovereign-sentinel.gguf --port 8080")
        print("       Set GEMMA_BASE_URL=http://localhost:8080")
        print("=" * 70 + "\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Generation — Ollama native endpoint (primary)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.5,
        stop: list[str] | None = None,
        image_path: str | None = None,
        model_name: str | None = None,
    ) -> dict:
        """Generate a response using Ollama's /api/generate endpoint.

        Falls back to the OpenAI-compatible endpoint when /api/generate
        is not available (e.g. plain llama.cpp server).
        """
        if "/11434" in self.base_url or self._is_ollama():
            return self._generate_ollama(
                prompt, system_prompt, max_tokens, temperature, stop, image_path,
                model_name or self.model_name,
            )
        return self._generate_openai_compat(
            prompt, system_prompt, max_tokens, temperature, stop
        )

    # ------------------------------------------------------------------

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        image_path: str | None,
        model_name: str | None = None,
    ) -> dict:
        payload: dict = {
            "model": model_name or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["options"]["stop"] = stop
        if image_path:
            payload["images"] = [self._encode_image(image_path)]

        start = time.perf_counter()
        resp = self.client.post("/api/generate", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        return {
            "text": data.get("response", ""),
            "tokens_in": data.get("prompt_eval_count"),
            "tokens_out": data.get("eval_count"),
            "latency_ms": latency_ms,
            "model": self.model_name,
            "finish_reason": "stop" if data.get("done") else "length",
        }

    # ------------------------------------------------------------------

    def _generate_openai_compat(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> dict:
        """Fallback for llama.cpp / other OpenAI-compatible servers."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        start = time.perf_counter()
        resp = self.client.post("/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM server returned HTTP {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return {
            "text": choice["message"]["content"],
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "latency_ms": latency_ms,
            "model": self.model_name,
            "finish_reason": choice.get("finish_reason"),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_ollama(self) -> bool:
        """Detect if the server is Ollama by probing /api/tags."""
        try:
            resp = self.client.get("/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return its base64 encoding for Ollama."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def close(self):
        self.client.close()
