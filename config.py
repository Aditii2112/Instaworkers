import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent


@dataclass
class GemmaConfig:
    base_url: str = os.getenv("GEMMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("GEMMA_MODEL_NAME", "sovereign-sentinel")


@dataclass
class VLMConfig:
    """PaliGemma 2 (3B) via mlx-vlm for on-device vision-language inference."""
    model_path: str = os.getenv("VLM_MODEL_PATH", "google/paligemma2-3b-pt-224")
    max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "200"))
    temperature: float = float(os.getenv("VLM_TEMPERATURE", "0.1"))


MATFORMER_PROFILES = {
    "S":  {"max_tokens": 128,  "context_k": 2,  "temperature": 0.3, "label": "Small"},
    "M":  {"max_tokens": 512,  "context_k": 5,  "temperature": 0.5, "label": "Medium"},
    "L":  {"max_tokens": 1024, "context_k": 10, "temperature": 0.7, "label": "Large"},
    "XL": {"max_tokens": 2048, "context_k": 20, "temperature": 0.7, "label": "Extra-Large"},
}


@dataclass
class MatFormerConfig:
    granularity: str = os.getenv("MATFORMER_GRANULARITY", "M")

    @property
    def profile(self) -> dict:
        return MATFORMER_PROFILES.get(self.granularity, MATFORMER_PROFILES["M"])


@dataclass
class InstaBrainConfig:
    db_path: str = os.getenv("INSTABRAIN_DB_PATH", str(ROOT_DIR / "data" / "instabrain.db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim: int = 384


@dataclass
class ObservabilityConfig:
    events_path: str = os.getenv("EVENTS_JSONL_PATH", str(ROOT_DIR / "data" / "events.jsonl"))


@dataclass
class ToolsConfig:
    http_allowlist: list = field(default_factory=lambda: [
        h.strip() for h in os.getenv("TOOL_HTTP_ALLOWLIST", "httpbin.org,api.github.com").split(",")
    ])
    shell_allowlist: list = field(default_factory=lambda: [
        c.strip() for c in os.getenv("TOOL_SHELL_ALLOWLIST", "ls,cat,head,tail,wc,date,uptime,df,whoami,say").split(",")
    ])
    fs_base_dir: str = os.getenv("TOOL_FS_BASE_DIR", str(ROOT_DIR / "data"))


@dataclass
class CheckpointConfig:
    enabled: bool = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
    interval_calls: int = int(os.getenv("CHECKPOINT_INTERVAL_CALLS", "10"))


@dataclass
class AuditorConfig:
    enabled: bool = os.getenv("AUDITOR_ENABLED", "true").lower() == "true"
    model_name: str = os.getenv("AUDITOR_MODEL", "sovereign-sentinel")


@dataclass
class VideoConfig:
    demo_path: str = os.getenv("VIDEO_DEMO_PATH", str(ROOT_DIR / "data" / "demo.mp4"))
    fps_target: float = float(os.getenv("VIDEO_FPS", "2.0"))
    max_frames: int = int(os.getenv("VIDEO_MAX_FRAMES", "60"))


@dataclass
class GeminiConfig:
    enabled: bool = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
    timeout_seconds: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))


@dataclass
class HybridReliabilityConfig:
    enabled: bool = (
        os.getenv("HYBRID_RELIABILITY_ENABLED", "true").lower() == "true"
    )
    confidence_threshold: float = float(
        os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.8")
    )
    high_risk_budget_usd: float = float(os.getenv("HIGH_RISK_BUDGET_USD", "500"))
    pending_retry_seconds: int = int(os.getenv("PENDING_RETRY_SECONDS", "30"))
    max_pending_process_per_cycle: int = int(
        os.getenv("MAX_PENDING_PROCESS_PER_CYCLE", "5")
    )


@dataclass
class AppConfig:
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    matformer: MatFormerConfig = field(default_factory=MatFormerConfig)
    instabrain: InstaBrainConfig = field(default_factory=InstaBrainConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    hybrid: HybridReliabilityConfig = field(default_factory=HybridReliabilityConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


def load_config() -> AppConfig:
    return AppConfig()
