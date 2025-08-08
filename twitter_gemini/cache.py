import hashlib
import json
import os
from typing import Any, Dict, Optional
from .config import Paths
from .logging_utils import log_cache


class JsonlCache:
    def __init__(self, name: str):
        self.file_path = os.path.join(Paths.cache_dir, f"{name}.jsonl")
        os.makedirs(Paths.cache_dir, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        key = row.get("key")
                        if key:
                            self._index[key] = row
                    except Exception:
                        continue

    @staticmethod
    def _hash_dict(d: Dict[str, Any]) -> str:
        normalized = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def make_key(self, payload: Dict[str, Any]) -> str:
        return self._hash_dict(payload)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self._index.get(key)
        if row is not None:
            log_cache(row.get("tweet_id"), "cache_hit", {"key": key})
            return row
        log_cache(None, "cache_miss", {"key": key})
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        row = {"key": key, **value}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._index[key] = row