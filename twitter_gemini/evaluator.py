from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import TwitterDataset, Annotation
from .gemini_client import GeminiLabeler
from .label_normalization import convert_label_tuple, Stance


@dataclass
class Prediction:
    tweet_id: str
    topic: str
    stance: Stance


@dataclass
class EvalResult:
    per_topic_reports: Dict[str, Dict[str, Any]]
    overall_report: Dict[str, Any]
    confusion_by_topic: Dict[str, Any]
    cost_estimate: Dict[str, Any]


def evaluate_annotations(
    annotations: List[Annotation],
    dataset: TwitterDataset,
    model: Optional[GeminiLabeler] = None,
    max_examples: Optional[int] = None,
) -> EvalResult:
    model = model or GeminiLabeler()

    y_true: List[Tuple[str, str]] = []  # (topic, stance)
    y_pred: List[Tuple[str, str]] = []
    topics: List[str] = []

    counted_calls = 0

    for idx, ann in enumerate(annotations):
        if max_examples is not None and idx >= max_examples:
            break
        tw = dataset.get_tweet(ann.tweet_id)

        # Warn and skip missing text
        if tw.text is None or str(tw.text).strip() == "":
            continue

        # If images are referenced but none available, we proceed with text-only, but this is logged inherently by the pipeline caller
        # because the prompt is text-only. We only warn via output aggregation later if needed.

        # Normalize ground truth
        canon_topic, canon_stance, _ = convert_label_tuple(
            raw_topic=ann.topic,
            raw_stance=ann.stance_label,
            raw_target=ann.stance_target,
        )
        if canon_topic is None or canon_stance is None:
            # skip examples with ambiguous normalization
            continue

        # Predict with Gemini
        preds = model.label(tweet_id=ann.tweet_id, text=tw.text)
        counted_calls += 1 if preds else 0
        # Find the first prediction with a recognized topic, else map to Unrelated
        pred_topic = "Unrelated"
        pred_stance = "Unrelated"
        for p in preds:
            t = p.get("topic")
            s = p.get("stance")
            if t in (
                "Russian Ukrainian Conflict",
                "False COVID Treatment",
                "Taiwan Question",
                "US 2024 Election",
                "Business Merger",
                "Unrelated",
            ) and s in ("Pro", "Against", "Neutral", "Unrelated"):
                pred_topic = t
                pred_stance = s
                break

        topics.append(canon_topic)
        y_true.append((canon_topic, canon_stance))
        y_pred.append((pred_topic, pred_stance))

    # Build per-topic metrics for stance where topic matches; and topic classification separately
    df = pd.DataFrame(
        {
            "true_topic": [t for t, _ in y_true],
            "true_stance": [s for _, s in y_true],
            "pred_topic": [t for t, _ in y_pred],
            "pred_stance": [s for _, s in y_pred],
        }
    )

    reports_by_topic: Dict[str, Dict[str, Any]] = {}
    confusion_by_topic: Dict[str, Any] = {}

    for topic in sorted(df.true_topic.unique()):
        df_topic = df[df.true_topic == topic]
        # Topic accuracy within this subset
        topic_acc = float((df_topic.pred_topic == df_topic.true_topic).mean())
        # Stance report only where topic matched
        matched = df_topic[df_topic.pred_topic == df_topic.true_topic]
        if len(matched) > 0:
            stance_report = classification_report(
                matched.true_stance, matched.pred_stance, labels=["Pro", "Against", "Neutral"], zero_division=0, output_dict=True
            )
            cm = confusion_matrix(
                matched.true_stance,
                matched.pred_stance,
                labels=["Pro", "Against", "Neutral"],
            ).tolist()
        else:
            stance_report = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
            cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        reports_by_topic[topic] = {
            "topic_accuracy": topic_acc,
            "stance_report": stance_report,
            "num_examples": int(len(df_topic)),
            "num_matched_topic": int(len(matched)),
        }
        confusion_by_topic[topic] = {"labels": ["Pro", "Against", "Neutral"], "matrix": cm}

    # Overall topic accuracy
    overall_topic_acc = float((df.pred_topic == df.true_topic).mean()) if len(df) else 0.0

    # Overall stance report where topic matched
    matched_all = df[df.pred_topic == df.true_topic]
    if len(matched_all) > 0:
        overall_stance_report = classification_report(
            matched_all.true_stance, matched_all.pred_stance, labels=["Pro", "Against", "Neutral"], zero_division=0, output_dict=True
        )
    else:
        overall_stance_report = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}

    overall_report = {
        "topic_accuracy": overall_topic_acc,
        "stance_report": overall_stance_report,
        "num_examples": int(len(df)),
        "num_matched_topic": int(len(matched_all)),
    }

    # Cost estimation (simple heuristic): assume 1K tokens per example average in+out combined for large-context runs
    # Adjust easily later as needed or from model usage logs if exposed.
    tokens_per_example = 1000
    price_per_million_tokens = 0.03  # placeholder; adjust if exact price available
    total_tokens = tokens_per_example * counted_calls
    cost_usd = total_tokens / 1_000_000 * price_per_million_tokens

    cost_estimate = {
        "calls": counted_calls,
        "tokens_per_example": tokens_per_example,
        "total_tokens": total_tokens,
        "price_per_million_tokens_usd": price_per_million_tokens,
        "estimated_cost_usd": cost_usd,
    }

    return EvalResult(
        per_topic_reports=reports_by_topic,
        overall_report=overall_report,
        confusion_by_topic=confusion_by_topic,
        cost_estimate=cost_estimate,
    )