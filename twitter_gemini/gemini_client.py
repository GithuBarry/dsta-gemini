import base64
import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .config import GeminiConfig
from .cache import JsonlCache
from .logging_utils import log_request, log_response


class GeminiLabeler:
    def __init__(self, cache_name: str = "gemini_labels") -> None:
        self.config = GeminiConfig()
        if not self.config.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=self.config.api_key)
        self.model = self.config.model
        self.cache = JsonlCache(cache_name)

    def _build_prompt(self, example_image_b64: Optional[str], example_text: Optional[str], input_text: str, image_bytes: Optional[bytes] = None) -> List[types.Content]:
        example_parts: List[types.Part] = []
        if example_image_b64 is not None:
            example_parts.append(
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(example_image_b64),
                )
            )
        if example_text is not None:
            example_parts.append(types.Part.from_text(text=example_text))

        contents = []
        if example_parts:
            contents.append(types.Content(role="user", parts=example_parts))
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "**Analyzing Tweet's Stance**\n\n"
                                "I'm focusing on classifying the tweet's stance...\n\n"
                                "**Evaluating Tweet Sentiment**\n\n"
                                "I've determined the tweet's stance is \"Against\" ...\n\n"
                            )
                        ),
                        types.Part.from_text(
                            text="""[
    {
      \"stance\": \"Against\",
      \"topic\": \"Russian Ukrainian Conflict\"
    }
  ]""",
                        ),
                    ],
                )
            )
        # Build user parts with optional image
        user_parts: List[types.Part] = []
        if image_bytes is not None:
            user_parts.append(
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
                )
            )
        user_parts.append(types.Part.from_text(text=input_text))
        contents.append(types.Content(role="user", parts=user_parts))
        return contents

    def _build_config(self) -> types.GenerateContentConfig:
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["stance", "topic"],
                    properties={
                        "stance": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The stance toward the topic's target",
                            enum=["Pro", "Against", "Neutral", "Unrelated"],
                        ),
                        "topic": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The exact topic name or 'Unrelated'",
                            enum=[
                                "Russian Ukrainian Conflict",
                                "False COVID Treatment",
                                "Taiwan Question",
                                "US 2024 Election",
                                "Business Merger",
                                "Unrelated",
                            ],
                        ),
                        "suggested_keywords": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description=(
                                "Relevant keywords when stance is Unrelated (optional for other stances)"
                            ),
                            items=genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        ),
                    },
                ),
            ),
            system_instruction=[
                types.Part.from_text(
                    text=(
                        "# Tweet Stance Classification\n\n"
                        "Analyze the tweet and identify which of these topics it relates to. For each detected topic, classify the stance toward the specific target:\n\n"
                        "1. \"Russian Ukrainian Conflict\" - Target: Russia's position\n"
                        "2. \"False COVID Treatment\" - Target: Chloroquine/Hydroxychloroquine for COVID-19\n"
                        "3. \"Taiwan Question\" - Target: Mainland China's position on Taiwan  \n"
                        "4. \"US 2024 Election\" - Target: Joe Biden's candidacy\n"
                        "5. \"Business Merger\" - Target: Support for the merger\n\n"
                        "## Stance Options\n"
                        "- Pro: Supports/endorses the target\n"
                        "- Against: Opposes/criticizes the target\n"
                        "- Neutral: Mentions topic but no clear position\n"
                        "- Unrelated: Tweet does not relate to any listed topics\n\n"
                        "## Instructions\n"
                        "- Return one classification result for each detected topic listed above\n"
                        "- If the tweet relates to multiple listed topics, provide multiple results\n"
                        "- If unrelated to all topics, return single result with topic \"Unrelated\" and provide 1-5 keywords describing what the tweet is about\n"
                        "- Use exact topic names as listed above\n"
                        "- Consider context, sarcasm, and implied meanings"
                    )
                )
            ],
        )
        return cfg

    def label(self, *, tweet_id: str, text: str, image_bytes: Optional[bytes] = None) -> List[Dict[str, Any]]:
        input_payload = {
            "tweet_id": tweet_id,
            "text": text,
            "has_image": bool(image_bytes),
            "model": self.model,
        }
        cache_key = self.cache.make_key(input_payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_response(tweet_id, "cache_return", cached)
            return cached["response"]

        contents = self._build_prompt(None, None, text, image_bytes=image_bytes)
        config = self._build_config()

        log_request(tweet_id, "gemini.generate_content_stream", {"contents_len": len(contents)})
        chunks = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                chunks.append(chunk.text)
        raw_text = "".join(chunks)

        try:
            data = json.loads(raw_text)
        except Exception:
            # wrap non-json into Unrelated
            data = [
                {
                    "stance": "Unrelated",
                    "topic": "Unrelated",
                    "suggested_keywords": [raw_text[:120]],
                }
            ]
        self.cache.set(
            cache_key,
            {"tweet_id": tweet_id, "response": data, "raw_text": raw_text},
        )
        log_response(tweet_id, "gemini_response", {"raw_text": raw_text})
        return data