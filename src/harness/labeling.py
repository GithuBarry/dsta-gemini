"""Simple functions for tweet stance labeling using Gemini API."""

import base64
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..config import GEMINI_MODEL, VALID_STANCE_LABELS, STANDARDIZED_TOPICS, TARGET_CONFIG
from .llm_cache import LLMCache

load_dotenv()

# Initialize client and cache globally
cache = LLMCache()
client = genai.Client()


def generate_system_instruction(topics: Optional[List[str]] = None, exclude_topics: Optional[List[str]] = None) -> str:
    """Generate system instruction using standardized single targets per topic.
    
    Args:
        topics: If provided, only include these topics (overrides exclude_topics)
        exclude_topics: If provided, exclude these topics (ignored if topics is set)
    """
    # Determine which topics to include
    if topics is not None:
        # Use only specified topics
        filtered_topics = {k: v for k, v in STANDARDIZED_TOPICS.items() if k in topics}
    elif exclude_topics is not None:
        # Exclude specified topics
        filtered_topics = {k: v for k, v in STANDARDIZED_TOPICS.items() if k not in exclude_topics}
    else:
        # Use all topics
        filtered_topics = STANDARDIZED_TOPICS

    instruction_lines = [
        "# Tweet Stance Classification",
        "",
        "Analyze the tweet and classify the stance toward each relevant topic's standardized target:",
        ""
    ]

    for i, (topic, config) in enumerate(filtered_topics.items(), 1):
        instruction_lines.append(
            f'{i}. **"{topic}"** - Target: {config["target_name"]} ({config["description"]})'
        )

    instruction_lines.extend([
        "",
        "## Stance Options",
        "- **Pro**: Supports/endorses the target",
        "- **Against**: Opposes/criticizes the target",
        "- **Neutral**: Mentions topic but no clear position",
        "- **Unrelated**: Tweet does not relate to any listed topics",
        "",
        "## Instructions",
        "- Return one classification result for each detected topic listed above",
        "- If the tweet relates to multiple listed topics, provide multiple results",
        "- If unrelated to all topics, return single result with topic \"Unrelated\" and provide 1-5 keywords describing what the tweet is about",
        "- Use exact topic names as listed above",
        "- Always use the standardized target for each topic as specified",
        "- Consider context, sarcasm, and implied meanings"
    ])

    return "\n".join(instruction_lines)


def get_filtered_topics(topics: Optional[List[str]] = None, exclude_topics: Optional[List[str]] = None) -> List[str]:
    """Get list of topic names based on filtering criteria."""
    if topics is not None:
        # Use only specified topics
        return [k for k in STANDARDIZED_TOPICS.keys() if k in topics]
    elif exclude_topics is not None:
        # Exclude specified topics
        return [k for k in STANDARDIZED_TOPICS.keys() if k not in exclude_topics]
    else:
        # Use all topics
        return list(STANDARDIZED_TOPICS.keys())


def encode_image(image_path: str) -> Optional[bytes]:
    """Encode image to bytes."""
    if not os.path.exists(image_path):
        return None

    with open(image_path, 'rb') as f:
        return f.read()


def create_prompt_parts(text: str, image_paths: Optional[List[str]] = None, image_path: Optional[str] = None) -> List[types.Part]:
    """Create parts for the prompt, handling both text and multiple images.
    
    Args:
        text: Tweet text
        image_paths: List of image paths (preferred)
        image_path: Single image path (backward compatibility)
    """
    parts = []

    # Handle backward compatibility
    if image_path and not image_paths:
        image_paths = [image_path]

    # Add all images if provided
    if image_paths:
        for img_path in image_paths:
            image_data = encode_image(img_path)
            if image_data:
                ext = Path(img_path).suffix.lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(ext, 'image/jpeg')
                parts.append(types.Part.from_bytes(mime_type=mime_type, data=image_data))

    # Add text (with context about number of images if multiple)
    if image_paths and len(image_paths) > 1:
        text_with_context = f"[This tweet contains {len(image_paths)} images]\n\n{text}"
    else:
        text_with_context = text

    parts.append(types.Part.from_text(text=text_with_context))

    return parts


def label_tweet(tweet_text: str, image_path: Optional[str] = None, image_paths: Optional[List[str]] = None,
                tweet_id: Optional[str] = None, topics: Optional[List[str]] = None,
                exclude_topics: Optional[List[str]] = None) -> Dict:
    """Label a single tweet using the Gemini API.
    
    Args:
        tweet_text: The tweet text
        image_path: Single image path (backward compatibility)
        image_paths: List of image paths (preferred for multiple images)
        tweet_id: Optional tweet ID for logging purposes
        topics: If provided, only include these topics (overrides exclude_topics)
        exclude_topics: If provided, exclude these topics (ignored if topics is set)
    """
    user_parts = create_prompt_parts(tweet_text, image_paths, image_path)

    # Get filtered topics for response schema and system instruction
    allowed_topics = get_filtered_topics(topics, exclude_topics)

    contents = [
        types.Content(role="user", parts=user_parts)
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1, include_thoughts=True),
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
                        enum=VALID_STANCE_LABELS,
                    ),
                    "topic": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The exact topic name or 'Unrelated'",
                        enum=allowed_topics + ["Unrelated"],
                    ),
                    "suggested_keywords": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Relevant keywords when stance is Unrelated",
                        items=genai.types.Schema(type=genai.types.Type.STRING),
                    ),
                },
            ),
        ),
        system_instruction=[
            types.Part.from_text(text=generate_system_instruction(topics, exclude_topics))
        ],
    )

    # Check cache first
    system_inst = generate_system_instruction(topics, exclude_topics)
    cached_response = cache.get_cached_response(tweet_text, image_paths, system_inst, tweet_id)

    if cached_response:
        # Return cached result
        return {
            'result': cached_response['result'],
            'processing_time': cached_response['processing_time'],
            'input_text': tweet_text,
            'image_path': image_paths[0] if image_paths else None,
            'image_paths': image_paths,
            'image_count': len(image_paths),
            'response_text': cached_response['response_text'],
            'cached': True,
            'cache_key': cached_response['cache_key']
        }

    # Make API call
    start_time = time.time()
    response_text = ""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    processing_time = time.time() - start_time

    # Parse JSON
    result = response.parsed
    thoughts = [part.text for part in response.candidates[0].content.parts if part.thought]
    response_text = "".join([part.text for part in response.candidates[0].content.parts if not part.thought])

    # Cache the response
    cache_key = cache.save_response(tweet_text, image_paths, system_inst,
                                   response_text, result, processing_time, tweet_id, thoughts)

    return {
        'result': result,
        'processing_time': processing_time,
        'input_text': tweet_text,
        'image_path': image_paths[0] if image_paths else None,
        'image_paths': image_paths,
        'image_count': len(image_paths),
        'response_text': response_text,
        'cached': False,
        'cache_key': cache_key
    }


class GeminiLabeler:
    """Wrapper class for Gemini API labeling functionality."""

    def __init__(self):
        self.total_queries = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def label_tweet(self, tweet_text: str, image_path: Optional[str] = None, image_paths: Optional[List[str]] = None,
                    tweet_id: Optional[str] = None, topics: Optional[List[str]] = None,
                    exclude_topics: Optional[List[str]] = None) -> Dict:
        """Label a tweet using the Gemini API."""
        result = label_tweet(tweet_text, image_path, image_paths, tweet_id, topics, exclude_topics)

        # Update usage tracking (approximate)
        self.total_queries += 1
        self.input_tokens += len(tweet_text.split()) * 1.3  # Rough estimate
        if result['response_text']:
            self.output_tokens += len(result['response_text'].split()) * 1.3

        return result

    def get_cost_estimate(self) -> Dict:
        """Get estimated cost based on usage."""
        # Gemini Flash pricing (approximate)
        input_cost_per_1k = 0.000125  # $0.000125 per 1K input tokens
        output_cost_per_1k = 0.000375  # $0.000375 per 1K output tokens

        input_cost = (self.input_tokens / 1000) * input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost

        return {
            'total_queries': self.total_queries,
            'input_tokens': int(self.input_tokens),
            'output_tokens': int(self.output_tokens),
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost
        }

    def get_cache_stats(self):
        """Get cache statistics."""
        return cache.get_cache_stats()

    def clear_cache(self):
        """Clear the LLM cache."""
        cache.clear_cache()