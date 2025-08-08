import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Paths


@dataclass
class Tweet:
    tweet_id: str
    text: Optional[str]
    image_urls: List[str]


class TwitterDataset:
    def __init__(self) -> None:
        self.tweet_text_index: Dict[str, str] = {}
        self.image_map: Dict[str, List[str]] = {}
        self._load_texts()
        self._load_images()

    def _load_texts(self) -> None:
        text_path = os.path.join(Paths.raw_data, "tweet_text.json")
        with open(text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expecting mapping from tweet_id to text or list of objects
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "text" in v:
                    self.tweet_text_index[str(k)] = v["text"]
                elif isinstance(v, str):
                    self.tweet_text_index[str(k)] = v
        elif isinstance(data, list):
            for row in data:
                tid = str(row.get("tweet_id"))
                text = row.get("text")
                if tid and text is not None:
                    self.tweet_text_index[tid] = text

    def _load_images(self) -> None:
        parts = ["tweet_image_p1.json", "tweet_image_p2.json"]
        for part in parts:
            path = os.path.join(Paths.meta, part)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for tid, multi in mapping.items():
                urls: List[str] = []
                if isinstance(multi, dict):
                    for _, url in sorted(multi.items(), key=lambda x: int(x[0])):
                        urls.append(url)
                elif isinstance(multi, list):
                    urls = list(multi)
                self.image_map[str(tid)] = urls

    def get_tweet(self, tweet_id: str) -> Tweet:
        text = self.tweet_text_index.get(str(tweet_id))
        image_urls = self.image_map.get(str(tweet_id), [])
        return Tweet(tweet_id=str(tweet_id), text=text, image_urls=image_urls)

    def get_local_image_paths(self, tweet_id: str) -> List[str]:
        base_dir = os.path.join(Paths.raw_data, "images", str(tweet_id))
        if not os.path.isdir(base_dir):
            return []
        files = [
            os.path.join(base_dir, name)
            for name in sorted(os.listdir(base_dir))
            if os.path.isfile(os.path.join(base_dir, name))
        ]
        return files

    def get_local_image_bytes(self, tweet_id: str, index: int) -> Optional[bytes]:
        paths = self.get_local_image_paths(tweet_id)
        if index < 0 or index >= len(paths):
            return None
        path = paths[index]
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None


@dataclass
class Annotation:
    tweet_id: str
    image_index: int
    stance_target: str
    stance_label: str
    topic: str


class AnnotationLoader:
    def __init__(self, folder: Optional[str] = None) -> None:
        self.folder = folder or os.path.join(
            Paths.annotations, "Multi-Modal-Stance-Detection-flattened"
        )

    def load_csv(self, filename: str, topic_hint: Optional[str] = None) -> List[Annotation]:
        path = os.path.join(self.folder, filename)
        df = pd.read_csv(path)
        if "stance_target" not in df.columns or "stance_label" not in df.columns:
            raise ValueError(f"Unexpected columns in {filename}")
        topic: str = topic_hint
        if topic is None:
            # infer from file prefix before first _in-target/zero-shot
            topic = filename.split("_in-target")[0].split("_zero-shot")[0]
            topic = topic.replace("Multi-modal-", "").replace("-", " ")
        annotations: List[Annotation] = []
        for _, row in df.iterrows():
            annotations.append(
                Annotation(
                    tweet_id=str(row["tweet_id"]),
                    image_index=int(row.get("image_numero", 0)),
                    stance_target=str(row["stance_target"]),
                    stance_label=str(row["stance_label"]),
                    topic=str(topic).strip(),
                )
            )
        return annotations

    def load_all(self) -> List[Annotation]:
        ann: List[Annotation] = []
        for name in os.listdir(self.folder):
            if name.endswith(".csv"):
                ann.extend(self.load_csv(name))
        return ann