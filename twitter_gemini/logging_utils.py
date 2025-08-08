import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from .config import Paths


@dataclass
class LogRecord:
    timestamp: float
    phase: str
    tweet_id: Optional[str]
    message: str
    payload: Optional[Dict[str, Any]] = None


class HumanReadableLogger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def write(self, record: LogRecord) -> None:
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


_request_log = HumanReadableLogger(os.path.join(Paths.logs_dir, "gemini_requests.jsonl"))
_response_log = HumanReadableLogger(os.path.join(Paths.logs_dir, "gemini_responses.jsonl"))
_cache_log = HumanReadableLogger(os.path.join(Paths.logs_dir, "cache_access.jsonl"))


def log_request(tweet_id: Optional[str], message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    _request_log.write(LogRecord(time.time(), "request", tweet_id, message, payload))


def log_response(tweet_id: Optional[str], message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    _response_log.write(LogRecord(time.time(), "response", tweet_id, message, payload))


def log_cache(tweet_id: Optional[str], message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    _cache_log.write(LogRecord(time.time(), "cache", tweet_id, message, payload))