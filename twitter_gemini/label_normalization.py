from typing import Literal, Optional, Tuple

Stance = Literal["Pro", "Against", "Neutral", "Unrelated"]


STANCE_SYNONYMS = {
    "pro": "Pro",
    "support": "Pro",
    "oppose": "Against",
    "against": "Against",
    "negative": "Against",
    "neutral": "Neutral",
    "unrelated": "Unrelated",
}


TOPIC_CANON = {
    # Canonical topics and their allowed variants (lowercased)
    "Russian Ukrainian Conflict": {"russian ukrainian conflict", "russo-ukrainian conflict", "ukraine war", "russia ukraine"},
    "False COVID Treatment": {"false covid treatment", "covid chloroquine", "covid hydroxychloroquine", "covid-cq", "covid cq"},
    "Taiwan Question": {"taiwan question", "taiwan of china", "taiwan", "moc", "toc", "taiwan issue"},
    "US 2024 Election": {"us 2024 election", "us election 2024", "joe biden election", "2024 election"},
    "Business Merger": {"business merger", "merger", "will-they-wont-they"},
    "Unrelated": {"unrelated"},
}

# Target normalization per topic
TARGET_NORMALIZATION = {
    "Taiwan Question": {
        # variations mapped to consistent targets
        "mainland of china": "Mainland China",
        "mainland": "Mainland China",
        "prc": "Mainland China",
        "china": "Mainland China",
        "taiwan of china": "Taiwan",
        "taiwan": "Taiwan",
        "roc": "Taiwan",
    },
    "Russian Ukrainian Conflict": {
        "russia": "Russia",
        "rus": "Russia",
        "ukraine": "Ukraine",
        "ukr": "Ukraine",
    },
    "False COVID Treatment": {
        "chloroquine": "Chloroquine/Hydroxychloroquine",
        "hydroxychloroquine": "Chloroquine/Hydroxychloroquine",
        "hcq": "Chloroquine/Hydroxychloroquine",
        "cq": "Chloroquine/Hydroxychloroquine",
    },
    "US 2024 Election": {
        "joe biden": "Joe Biden",
        "biden": "Joe Biden",
        "jb": "Joe Biden",
    },
    "Business Merger": {},
}


def normalize_stance(raw: str) -> Optional[Stance]:
    key = (raw or "").strip().lower()
    return STANCE_SYNONYMS.get(key)


def normalize_topic(raw: str) -> Optional[str]:
    key = (raw or "").strip().lower()
    for canon, variants in TOPIC_CANON.items():
        if key == canon.lower() or key in variants:
            return canon
    return None


def normalize_target(topic: str, raw_target: Optional[str]) -> Optional[str]:
    if raw_target is None:
        return None
    key = raw_target.strip().lower()
    mapping = TARGET_NORMALIZATION.get(topic)
    if not mapping:
        return raw_target
    return mapping.get(key, raw_target)


def convert_label_tuple(
    raw_topic: str,
    raw_stance: str,
    raw_target: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Stance], Optional[str]]:
    topic = normalize_topic(raw_topic)
    stance = normalize_stance(raw_stance)
    target = normalize_target(topic, raw_target) if topic else raw_target
    return topic, stance, target