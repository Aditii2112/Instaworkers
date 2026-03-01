"""Optional Gemini validator for hybrid reliability mode.

Uses the Gemini REST API when enabled and configured.
"""

import json

import httpx

from config import AppConfig


class GeminiRuntime:
    def __init__(self, config: AppConfig):
        self.enabled = config.gemini.enabled and bool(config.gemini.api_key)
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.model_name
        self.timeout_seconds = config.gemini.timeout_seconds
        self.base_url = "https://generativelanguage.googleapis.com"
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=float(self.timeout_seconds),
        )

    def connectivity_state(self) -> str:
        if not self.enabled:
            return "offline"
        try:
            resp = self.client.get(
                "/v1beta/models",
                params={"key": self.api_key},
            )
            return "online" if resp.status_code == 200 else "offline"
        except Exception:
            return "offline"

    def validate_action(self, payload: dict) -> dict:
        if not self.enabled:
            raise RuntimeError("Gemini validator is disabled or missing API key")

        prompt = (
            "You are a strict action validator for an edge field-agent.\n"
            "Given the action payload, return ONLY valid JSON with:\n"
            '{"approved": bool, "confidence": float, "reason": str, '
            '"recommended_state": "teacher_validated|rejected", '
            '"requires_human_review": bool}\n'
            f"\nAction payload:\n{json.dumps(payload, indent=2, default=str)}"
        )
        resp = self.client.post(
            f"/v1beta/models/{self.model_name}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "approved": {"type": "BOOLEAN"},
                            "confidence": {"type": "NUMBER"},
                            "reason": {"type": "STRING"},
                            "recommended_state": {"type": "STRING"},
                            "requires_human_review": {"type": "BOOLEAN"},
                        },
                        "required": [
                            "approved",
                            "confidence",
                            "reason",
                            "recommended_state",
                            "requires_human_review",
                        ],
                    },
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        return self._parse_verdict(text)

    @staticmethod
    def _parse_verdict(text: str) -> dict:
        cleaned = (text or "").strip()
        try:
            # First pass: direct JSON decode.
            parsed = json.loads(cleaned)
            return {
                "approved": bool(parsed.get("approved", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reason": str(parsed.get("reason", "")),
                "recommended_state": parsed.get(
                    "recommended_state",
                    "teacher_validated" if parsed.get("approved") else "rejected",
                ),
                "requires_human_review": bool(
                    parsed.get("requires_human_review", False)
                ),
                "raw_output": text,
            }
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

        try:
            # Second pass: strip markdown code fences that some models still emit.
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(cleaned[start:end])
                return {
                    "approved": bool(parsed.get("approved", False)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": str(parsed.get("reason", "")),
                    "recommended_state": parsed.get(
                        "recommended_state",
                        "teacher_validated" if parsed.get("approved") else "rejected",
                    ),
                    "requires_human_review": bool(
                        parsed.get("requires_human_review", False)
                    ),
                    "raw_output": text,
                }
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        return {
            "approved": False,
            "confidence": 0.0,
            "reason": "Could not parse Gemini validator output",
            "recommended_state": "rejected",
            "requires_human_review": True,
            "raw_output": text,
        }

    def close(self):
        self.client.close()
