import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = os.environ.get("GEMINI_API_KEY", "")
    model: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    # Controls
    request_timeout_s: int = int(os.environ.get("GEMINI_TIMEOUT_S", "60"))
    max_retries: int = int(os.environ.get("GEMINI_MAX_RETRIES", "2"))


@dataclass(frozen=True)
class Paths:
    workspace: str = "/workspace"
    data_root: str = "/workspace/twitter"
    raw_data: str = "/workspace/twitter/data"
    meta: str = "/workspace/twitter/meta"
    annotations: str = "/workspace/twitter/annotations"
    logs_dir: str = "/workspace/logs"
    cache_dir: str = "/workspace/cache"


os.makedirs(Paths.logs_dir, exist_ok=True)
os.makedirs(Paths.cache_dir, exist_ok=True)