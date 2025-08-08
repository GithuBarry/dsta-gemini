import argparse
import json
import os
from typing import Any, Dict

from .dataset import TwitterDataset, AnnotationLoader
from .gemini_client import GeminiLabeler
from .evaluator import evaluate_annotations


def run_single_example(tweet_id: str, image_index: int | None = None):
    ds = TwitterDataset()
    tw = ds.get_tweet(tweet_id)
    if tw.text is None or str(tw.text).strip() == "":
        raise SystemExit(f"tweet_id {tweet_id} missing text; cannot run example")
    image_bytes = None
    used_image_index = None
    if image_index is not None:
        image_bytes = ds.get_local_image_bytes(tweet_id, image_index)
        if image_bytes is not None:
            used_image_index = image_index
    labeler = GeminiLabeler()
    preds = labeler.label(tweet_id=tweet_id, text=tw.text, image_bytes=image_bytes)
    return {
        "tweet_id": tweet_id,
        "text": tw.text,
        "used_image_index": used_image_index,
        "has_image": image_bytes is not None,
        "predictions": preds,
    }


def run_evaluate(max_examples: int | None = None):
    ds = TwitterDataset()
    loader = AnnotationLoader()
    annotations = loader.load_all()
    result = evaluate_annotations(annotations, ds, max_examples=max_examples)
    return {
        "per_topic_reports": result.per_topic_reports,
        "overall_report": result.overall_report,
        "confusion_by_topic": result.confusion_by_topic,
        "cost_estimate": result.cost_estimate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Twitter Stance Classification CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="Run a single example by tweet_id")
    one.add_argument("tweet_id", help="Tweet ID to run")
    one.add_argument("--image-index", type=int, default=None, help="Optional local image index to include")

    evalp = sub.add_parser("eval", help="Evaluate all annotations")
    evalp.add_argument("--max", type=int, default=None, help="Max examples for quick run")

    args = parser.parse_args()
    if args.cmd == "one":
        out = run_single_example(args.tweet_id, image_index=args.image_index)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        out = run_evaluate(max_examples=args.max)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()