"""Simple functions for processing Twitter stance detection datasets."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from ..config import DATA_DIR, OUTPUT_DIR, normalize_stance_label, STANCE_TARGET_TO_CODE, parse_filename, \
    STANDARDIZED_TOPICS
from ..harness.labeling import label_tweet, GeminiLabeler

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Suppress external library logging
logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def standardize_annotation_target(annotation: Dict) -> Dict:
    """Convert annotation to use standardized target and swap stance if needed."""
    topic = annotation['topic']
    target = annotation['target']
    stance = annotation['stance']

    if topic not in STANDARDIZED_TOPICS:
        # Topic not in standardized list, return as-is
        return annotation

    topic_config = STANDARDIZED_TOPICS[topic]
    standardized_target = topic_config['target']
    opposite_target = topic_config.get('opposite_target')

    # Create a copy to modify
    result = annotation.copy()

    if target == standardized_target:
        # Already using standardized target, no change needed
        pass
    elif target == opposite_target and opposite_target is not None:
        # Using opposite target, need to swap stance and update target
        if stance == 'Pro':
            result['stance'] = 'Against'
        elif stance == 'Against':
            result['stance'] = 'Pro'
        # Neutral stays Neutral

        # Update target to standardized
        result['target'] = standardized_target
        result['original_target'] = target  # Keep track of original
        result['stance_swapped'] = True
    else:
        # Unknown target for this topic, keep as-is but note it
        result['unknown_target'] = True

    return result


def load_tweet_text(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """Load tweet text from JSON file."""
    tweet_file = Path(data_dir) / "data" / "tweet_text.json"

    with open(tweet_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Convert nested structure to flat dict
    tweet_texts = {}
    for tweet_id, content in raw_data.items():
        if isinstance(content, dict):
            # Take the first text entry (usually "0")
            tweet_texts[tweet_id] = next(iter(content.values()))
        else:
            tweet_texts[tweet_id] = content

    return tweet_texts


def get_image_paths(tweet_id: str, data_dir: str = DATA_DIR) -> List[str]:
    """Get all image paths for a tweet ID."""
    image_dir = Path(data_dir) / "data" / "images" / tweet_id

    # Check if tweet ID directory exists
    if not image_dir.exists():
        return []

    # Look for all image files in the tweet directory
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        image_paths.extend([str(f) for f in image_dir.glob(f"*{ext}")])

    # Sort by filename to ensure consistent ordering
    return sorted(image_paths)


def get_image_path(tweet_id: str, data_dir: str = DATA_DIR) -> Optional[str]:
    """Get first image path for a tweet ID (backward compatibility)."""
    paths = get_image_paths(tweet_id, data_dir)
    return paths[0] if paths else None


def load_annotations(csv_file: str, data_dir: str = DATA_DIR) -> List[Dict]:
    """Load annotations from CSV file with proper parsing."""
    csv_path = Path(data_dir) / "annotations" / "Multi-Modal-Stance-Detection-flattened" / csv_file

    # Parse topic and target from filename
    topic, target_code = parse_filename(csv_file)

    annotations = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map stance_target text to target code
            stance_target_text = row.get('stance_target', '')
            target_code_from_text = STANCE_TARGET_TO_CODE.get(stance_target_text, target_code)

            # Create base annotation
            base_annotation = {
                'tweet_id': row['tweet_id'],
                'stance': normalize_stance_label(row['stance_label']),
                'stance_label': row['stance_label'],  # Keep original stance label
                'normalized_stance_label': normalize_stance_label(row['stance_label']),
                'topic': topic,
                'target': target_code_from_text,
                'stance_target_text': stance_target_text,
                'source_file': csv_file,
            }

            # Apply standardization (target conversion and stance swapping)
            standardized_annotation = standardize_annotation_target(base_annotation)
            annotations.append(standardized_annotation)

    return annotations


def process_dataset(csv_file: str, data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR,
                    topics: Optional[List[str]] = None, exclude_topics: Optional[List[str]] = None) -> str:
    """Process a complete dataset and save results."""
    logger.info(f"Processing dataset: {csv_file}")

    # Load data
    tweet_texts = load_tweet_text(data_dir)
    annotations = load_annotations(csv_file, data_dir)

    results = []
    total = len(annotations)

    for i, annotation in enumerate(annotations, 1):
        tweet_id = annotation['tweet_id']

        # Get tweet text
        tweet_text = tweet_texts.get(tweet_id, "")
        if not tweet_text:
            logger.warning(f"No text found for tweet {tweet_id}")
            continue

        # Get image path
        image_path = get_image_path(tweet_id, data_dir)

        # Label the tweet
        try:
            prediction = label_tweet(tweet_text, image_path, tweet_id=tweet_id, topics=topics,
                                     exclude_topics=exclude_topics)

            result = {
                'tweet_id': tweet_id,
                'tweet_text': tweet_text,
                'image_path': image_path,
                'ground_truth_stance': annotation['stance'],
                'ground_truth_topic': annotation['topic'],
                'ground_truth_target': annotation['target'],
                'prediction': prediction['result'],
                'processing_time': prediction['processing_time']
            }

            results.append(result)

            if i % 10 == 0:
                logger.info(f"Processed {i}/{total} tweets")

        except Exception as e:
            logger.error(f"Error processing tweet {tweet_id}: {e}")
            continue

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / f"{Path(csv_file).stem}_results.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")
    return str(output_file)


class TwitterDataProcessor:
    """Wrapper class for processing Twitter datasets."""

    def __init__(self, data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.labeler = GeminiLabeler()

    def load_tweet_texts(self) -> Dict[str, str]:
        """Load tweet texts from JSON file."""
        return load_tweet_text(self.data_dir)

    def load_missing_images(self) -> List[str]:
        """Generate list of tweets with incomplete image sets."""
        incomplete_image_tweets = []

        # Load all annotations to get unique tweet IDs
        annotations = self.load_annotations()
        unique_tweet_ids = set(ann['tweet_id'] for ann in annotations)

        for tweet_id in unique_tweet_ids:
            image_paths = self.get_image_paths(tweet_id)
            expected_count = self._get_expected_image_count(tweet_id)

            # Include tweets with missing or incomplete images
            if len(image_paths) < expected_count:
                incomplete_image_tweets.append(tweet_id)

        return incomplete_image_tweets

    def load_annotations(self) -> List[Dict]:
        """Load all annotations from CSV files."""
        annotations = []
        ann_dir = Path(self.data_dir) / "annotations" / "Multi-Modal-Stance-Detection-flattened"
        for csv_file in ann_dir.glob("*.csv"):
            annotations.extend(load_annotations(csv_file.name, self.data_dir))
        return annotations

    def get_image_path(self, tweet_id: str) -> Optional[str]:
        """Get image path for a tweet ID."""
        return get_image_path(tweet_id, self.data_dir)

    def process_dataset(self, sample_size: Optional[int] = None, file_filter: str = "*.csv",
                        require_complete_images: bool = True, deduplicate: bool = False,
                        filter_conflicts: bool = False, uniform_sampling: bool = False,
                        topics: Optional[List[str]] = None, exclude_topics: Optional[List[str]] = None,
                        seed=None) -> List[Dict]:
        """Process dataset files matching the filter.
        
        Args:
            sample_size: Limit to N samples
            file_filter: Glob pattern for CSV files
            require_complete_images: Filter out tweets with any missing images
            deduplicate: Skip duplicate annotations (same tweet+topic+target+stance)
            filter_conflicts: Resolve conflicting annotations (same tweet+topic+target, different stance)
            uniform_sampling: Sample uniformly across topics
            topics: If provided, only include these topics (overrides exclude_topics)
            exclude_topics: If provided, exclude these topics (ignored if topics is set)
        """
        ann_dir = Path(self.data_dir) / "annotations" / "Multi-Modal-Stance-Detection-flattened"
        csv_files = list(ann_dir.glob(file_filter))

        if not csv_files:
            logger.warning(f"No files found matching filter: {file_filter}")
            return []

        # Load data
        tweet_texts = self.load_tweet_texts()
        all_annotations = []
        for csv_file in csv_files:
            all_annotations.extend(load_annotations(csv_file.name, self.data_dir))

        # Deduplicate if requested
        if deduplicate:
            all_annotations = self._deduplicate_annotations(all_annotations)

        # Filter conflicts if requested
        if filter_conflicts:
            all_annotations = self._filter_conflicting_annotations(all_annotations)

        # Filter by topics if requested
        if topics or exclude_topics:
            all_annotations = self._filter_by_topics(all_annotations, topics, exclude_topics)

        # Filter for complete images if requested
        if require_complete_images:
            all_annotations = self._filter_complete_images(all_annotations)

        # Sample uniformly across topics if requested
        if uniform_sampling and sample_size:
            all_annotations = self._uniform_topic_sampling(all_annotations, sample_size, seed=seed)
        elif sample_size and sample_size < len(all_annotations):
            all_annotations = all_annotations[:sample_size]

        annotations = all_annotations

        results = []
        total = len(annotations)

        for i, annotation in enumerate(tqdm(annotations, desc="annotating"), 1):
            tweet_id = annotation['tweet_id']

            # Get tweet text
            tweet_text = tweet_texts.get(tweet_id, "")
            if not tweet_text:
                logger.warning(f"No text found for tweet {tweet_id}")
                continue

            # Get all image paths
            image_paths = self.get_image_paths(tweet_id)
            image_path = image_paths[0] if image_paths else None  # For backward compatibility

            # Label the tweet with all images
            try:
                prediction = self.labeler.label_tweet(tweet_text, image_path, image_paths, tweet_id, topics,
                                                      exclude_topics)

                result = {
                    'tweet_id': tweet_id,
                    'tweet_text': tweet_text,
                    'image_path': image_path,  # First image for backward compatibility
                    'all_image_paths': image_paths,
                    'image_count': len(image_paths),
                    'ground_truth_stance': annotation['stance'],
                    'ground_truth_topic': annotation['topic'],
                    'ground_truth_target': annotation['target'],
                    'source_file': annotation['source_file'],
                    'stance_target_text': annotation.get('stance_target_text', ''),
                    'original_stance_label': annotation.get('stance_label', ''),
                    'prediction': prediction['result'],
                    'processing_time': prediction['processing_time']
                }

                results.append(result)

                if i % 10 == 0:
                    logger.info(f"Processed {i}/{total} tweets")

            except Exception as e:
                logger.error(f"Error processing tweet {tweet_id}: {e}")
                continue

        return results

    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics."""
        annotations = self.load_annotations()
        tweet_texts = self.load_tweet_texts()

        # Get unique tweet IDs from annotations
        annotated_tweet_ids = set(ann['tweet_id'] for ann in annotations)

        # Detailed image analysis
        tweets_with_all_images = 0
        tweets_with_some_images = 0
        tweets_without_images = 0
        total_expected_images = 0
        total_available_images = 0
        image_count_distribution = {}

        for tweet_id in annotated_tweet_ids:
            image_paths = self.get_image_paths(tweet_id)
            expected_images = self._get_expected_image_count(tweet_id)

            total_expected_images += expected_images
            total_available_images += len(image_paths)

            # Track image count distribution
            if len(image_paths) not in image_count_distribution:
                image_count_distribution[len(image_paths)] = 0
            image_count_distribution[len(image_paths)] += 1

            if len(image_paths) == 0:
                tweets_without_images += 1
            elif len(image_paths) == expected_images:
                tweets_with_all_images += 1
            else:
                tweets_with_some_images += 1

        # Annotation equivalence analysis
        duplicate_annotations = self._count_duplicate_annotations(annotations)
        conflicting_annotations = self._count_conflicting_annotations(annotations)

        return {
            'total_tweets_in_corpus': len(tweet_texts),
            'total_annotations': len(annotations),
            'unique_annotated_tweets': len(annotated_tweet_ids),
            'tweets_with_all_images': tweets_with_all_images,
            'tweets_with_partial_images': tweets_with_some_images,
            'tweets_without_images': tweets_without_images,
            'image_completeness_rate': tweets_with_all_images / len(annotated_tweet_ids) if annotated_tweet_ids else 0,
            'total_expected_images': total_expected_images,
            'total_available_images': total_available_images,
            'image_count_distribution': dict(sorted(image_count_distribution.items())),
            'annotation_files': len(set(ann['source_file'] for ann in annotations)),
            'duplicate_annotations': duplicate_annotations,
            'conflicting_annotations': conflicting_annotations
        }

    def get_image_paths(self, tweet_id: str) -> List[str]:
        """Get all image paths for a tweet ID."""
        return get_image_paths(tweet_id, self.data_dir)

    def _get_expected_image_count(self, tweet_id: str) -> int:
        """Estimate expected number of images for a tweet based on directory structure."""
        # This is a heuristic - check for highest numbered image file
        image_dir = Path(self.data_dir) / "data" / "images" / tweet_id
        if not image_dir.exists():
            return 0

        max_index = -1
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            for image_file in image_dir.glob(f"image_*{ext}"):
                try:
                    index = int(image_file.stem.split('_')[1])
                    max_index = max(max_index, index)
                except (IndexError, ValueError):
                    continue

        return max_index + 1 if max_index >= 0 else 0

    def _count_duplicate_annotations(self, annotations: List[Dict]) -> Dict:
        """Count duplicate annotations (same tweet_id, topic, target, stance)."""
        from collections import defaultdict

        groups = defaultdict(list)
        for ann in annotations:
            key = (ann['tweet_id'], ann['topic'], ann['target'], ann['stance'])
            groups[key].append(ann)

        duplicates = {k: len(v) for k, v in groups.items() if len(v) > 1}
        total_duplicates = sum(count - 1 for count in duplicates.values())

        return {
            'unique_groups_with_duplicates': len(duplicates),
            'total_duplicate_annotations': total_duplicates,
            'potential_savings': total_duplicates
        }

    def _count_conflicting_annotations(self, annotations: List[Dict]) -> Dict:
        """Count conflicting annotations respecting target stance equivalencies."""
        from collections import defaultdict

        # Group by (tweet_id, topic) to handle cross-target equivalencies
        groups = defaultdict(list)
        for ann in annotations:
            key = (ann['tweet_id'], ann['topic'])
            groups[key].append(ann)

        conflicts = {}
        total_conflicting = 0

        for key, group_anns in groups.items():
            if len(group_anns) > 1:
                # Get normalized stances using the stance switching logic
                normalized_stances = []
                for ann in group_anns:
                    normalized_stance = self._get_annotation_key(ann)[-1]
                    normalized_stances.append(normalized_stance)

                unique_normalized_stances = set(normalized_stances)

                if len(unique_normalized_stances) > 1:
                    # This is a true conflict (after normalization)
                    conflicts[key] = {
                        'total_annotations': len(group_anns),
                        'unique_stances': len(unique_normalized_stances),
                        'stance_distribution': {stance: normalized_stances.count(stance) for stance in
                                                unique_normalized_stances}
                    }
                    total_conflicting += len(group_anns)

        return {
            'groups_with_conflicts': len(conflicts),
            'total_conflicting_annotations': total_conflicting,
            'conflicts_by_key': conflicts
        }

    def _get_annotation_key(self, ann: Dict) -> Tuple[str, str, str, str]:
        """Get a key for an annotation (now simplified since standardization happens at load time)."""
        # Since annotations are now standardized during loading,
        # we can use them directly without additional stance switching
        return (ann['tweet_id'], ann['topic'], ann['target'], ann['stance'])

    def _deduplicate_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Remove duplicate annotations (same tweet_id, topic, target, stance)."""
        seen = set()
        deduplicated = []

        for ann in annotations:
            key = self._get_annotation_key(ann)
            if key not in seen:
                seen.add(key)
                deduplicated.append(ann)

        return deduplicated

    def _filter_conflicting_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Resolve conflicting annotations respecting target stance equivalencies.
        
        For topics like US 2020 Election, Pro-Trump and Against-Biden are equivalent,
        not conflicting. Uses majority voting for true conflicts.
        """
        from collections import defaultdict, Counter

        # Group annotations by semantic equivalence (tweet_id, topic)
        # This groups all annotations for the same tweet+topic together
        # so we can handle cross-target equivalencies like Pro-Trump ≈ Against-Biden
        groups = defaultdict(list)
        for ann in annotations:
            key = (ann['tweet_id'], ann['topic'])
            groups[key].append(ann)

        filtered = []
        conflicts_resolved = 0

        for key, group_anns in groups.items():
            if len(group_anns) == 1:
                # No conflict, keep the single annotation
                filtered.extend(group_anns)
            else:
                # Get normalized stances using the stance switching logic
                normalized_stances = []
                for ann in group_anns:
                    # Get the normalized stance from _get_annotation_key
                    normalized_stance = self._get_annotation_key(ann)[-1]
                    normalized_stances.append(normalized_stance)

                unique_stances = set(normalized_stances)

                if len(unique_stances) == 1:
                    # All have same normalized stance (equivalent), keep all
                    filtered.extend(group_anns)
                else:
                    # True conflict detected - resolve with majority voting
                    stance_counts = Counter(normalized_stances)
                    majority_stance = stance_counts.most_common(1)[0][0]

                    # Keep only annotations with majority normalized stance
                    resolved_anns = []
                    for ann in group_anns:
                        if self._get_annotation_key(ann)[-1] == majority_stance:
                            resolved_anns.append(ann)

                    filtered.extend(resolved_anns)
                    conflicts_resolved += len(group_anns) - len(resolved_anns)

        # Silently resolve conflicts - no logging needed for normal operation
        # if conflicts_resolved > 0:
        #     print(f"Resolved {conflicts_resolved} conflicting annotations using majority voting")

        return filtered

    def _filter_by_topics(self, annotations: List[Dict], topics: Optional[List[str]] = None,
                          exclude_topics: Optional[List[str]] = None) -> List[Dict]:
        """Filter annotations by topic.
        
        Args:
            annotations: List of annotations to filter
            topics: If provided, only include these topics (overrides exclude_topics)
            exclude_topics: If provided, exclude these topics (ignored if topics is set)
        """
        if topics:
            return [ann for ann in annotations if ann['topic'] in topics]
        elif exclude_topics:
            return [ann for ann in annotations if ann['topic'] not in exclude_topics]
        else:
            return annotations

    def _filter_complete_images(self, annotations: List[Dict]) -> List[Dict]:
        """Filter annotations to only include tweets with complete image sets."""
        filtered = []

        for ann in annotations:
            tweet_id = ann['tweet_id']
            image_paths = self.get_image_paths(tweet_id)
            expected_count = self._get_expected_image_count(tweet_id)

            # Only include if all expected images are present
            if len(image_paths) == expected_count and expected_count > 0:
                filtered.append(ann)

        return filtered

    def _uniform_topic_sampling(self, annotations: List[Dict], sample_size: int, seed=None) -> List[Dict]:
        """Sample uniformly across topics."""
        from collections import defaultdict
        import random

        # Group by topic
        topic_groups = defaultdict(list)
        for ann in annotations:
            topic_groups[ann['topic']].append(ann)

        rng = random.Random(seed)

        # Calculate samples per topic
        num_topics = len(topic_groups)
        samples_per_topic = sample_size // num_topics
        remaining_samples = sample_size % num_topics

        sampled = []
        topic_list = list(topic_groups.keys())
        rng.shuffle(topic_list)

        for i, topic in enumerate(topic_list):
            topic_anns = topic_groups[topic]
            # Give extra samples to first few topics if there's remainder
            topic_sample_size = samples_per_topic + (1 if i < remaining_samples else 0)

            if len(topic_anns) <= topic_sample_size:
                sampled.extend(topic_anns)
            else:
                sampled.extend(rng.sample(topic_anns, topic_sample_size))

        return sampled

    def save_results(self, name: str) -> Dict[str, str]:
        """Save results with given name prefix."""
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)

        # This would be called after process_dataset, so we'd save the last results
        # For now, return placeholder paths
        return {
            'results': str(output_path / f"{name}_results.json"),
            'summary': str(output_path / f"{name}_summary.json")
        }
