"""Configuration for stance detection system with standardized single targets per topic."""

from typing import List

# Default paths
DATA_DIR = "twitter"
OUTPUT_DIR = "output"
LOG_DIR = "logs"

# Gemini API settings
GEMINI_MODEL = "gemini-2.5-flash"

# Stance label normalization (from CSV analysis)
STANCE_MAPPING = {
    'Favor': 'Pro',
    'Support': 'Pro',
    'Against': 'Against',
    'Oppose': 'Against',
    'Neutral': 'Neutral',
    'Comment': 'Neutral',
    'Refute': 'Against',
    'Unrelated': 'Unrelated'
}

# STANDARDIZED TARGET CONFIGURATION
# Each topic uses only ONE standardized target for LLM prompts
# Stance labels for opposite targets are automatically swapped during processing
STANDARDIZED_TOPICS = {
    "Russian Ukrainian Conflict": {
        "target": "RUS",  # Standardized: always ask about Russia's position
        "target_name": "Russia's position in the Russian-Ukrainian conflict",
        "description": "Russia's justification for military actions against Ukraine",
        "opposite_target": "UKR"  # Ukraine stance labels will be swapped
    },
    "US 2020 Election": {
        "target": "DT",  # Standardized: always ask about Donald Trump
        "target_name": "Donald Trump as 2020 presidential candidate",
        "description": "Support for Donald Trump's 2020 presidential campaign",
        "opposite_target": "JB"  # Biden stance labels will be swapped
    },
    "Taiwan Question": {
        "target": "MOC",  # Standardized: always ask about China's position
        "target_name": "Mainland China's claim that Taiwan is part of China",
        "description": "Mainland China's territorial claim over Taiwan",
        "opposite_target": "TOC"  # Taiwan stance labels will be swapped
    },
    "False COVID Treatment": {
        "target": "CQ",  # Standardized: only one target exists
        "target_name": "The claim that Chloroquine/Hydroxychloroquine being a valid COVID-19 treatment",
        "description": "Using chloroquine or hydroxychloroquine to treat COVID-19",
        "opposite_target": None  # No opposite target
    },
    "Aetna-Humana Merger": {
        "target": "AET_HUM",
        "target_name": "Aetna-Humana business merger",
        "description": "Support for the Aetna-Humana business merger",
        "opposite_target": None
    },
    "Anthem-Cigna Merger": {
        "target": "ANTM_CI",
        "target_name": "Anthem-Cigna business merger",
        "description": "Support for the Anthem-Cigna business merger",
        "opposite_target": None
    },
    "Cigna-Express Scripts Merger": {
        "target": "CI_ESRX",
        "target_name": "Cigna-Express Scripts business merger",
        "description": "Support for the Cigna-Express Scripts business merger",
        "opposite_target": None
    },
    "CVS-Aetna Merger": {
        "target": "CSV_AET",
        "target_name": "CVS-Aetna business merger",
        "description": "Support for the CVS-Aetna business merger",
        "opposite_target": None
    },
    "Fox-Disney Merger": {
        "target": "FOXA_DIS",
        "target_name": "Fox-Disney business merger",
        "description": "Support for the Fox-Disney business merger",
        "opposite_target": None
    }
}

# Stance target to code mapping (from CSV analysis)
STANCE_TARGET_TO_CODE = {
    'Donald Trump': 'DT',
    'Joe Biden': 'JB',
    'Mainland of China': 'MOC',
    'Taiwan of China': 'TOC',
    'Russia': 'RUS',
    'Ukraine': 'UKR',
    'The use of "Chloroquine" and "Hydroxychloroquine" for the treatment or prevention from the coronavirus.': 'CQ',
    'Merger and acquisition between Aetna and Humana.': 'AET_HUM',
    'Merger and acquisition between Anthem and Cigna.': 'ANTM_CI',
    'Merger and acquisition between CVS Health and Aetna.': 'CSV_AET',
    'Merger and acquisition between Cigna and Express Scripts.': 'CI_ESRX',
    'Merger and acquisition between Disney and 21st Century Fox.': 'FOXA_DIS'
}

# Topic mapping for parsing filenames  
TOPIC_MAPPING = {
    'COVID-CQ': 'False COVID Treatment',
    'Russo-Ukrainian-Conflict': 'Russian Ukrainian Conflict',
    'Taiwan-Question': 'Taiwan Question',
    'Twitter-Stance-Election-2020': 'US 2020 Election',
    'Will-They-Wont-They': 'Business Merger'  # Generic - resolved by target
}

# Merger target to specific topic mapping
MERGER_TARGET_TO_TOPIC = {
    'AET_HUM': 'Aetna-Humana Merger',
    'ANTM_CI': 'Anthem-Cigna Merger',
    'CI_ESRX': 'Cigna-Express Scripts Merger',
    'CSV_AET': 'CVS-Aetna Merger',
    'FOXA_DIS': 'Fox-Disney Merger'
}


# Extract topic and target from filename
def parse_filename(filename: str):
    """Extract topic and target code from CSV filename."""
    if not filename.startswith('Multi-modal-'):
        return None, None

    # Find topic
    topic = None
    for key, value in TOPIC_MAPPING.items():
        if key in filename:
            topic = value
            break

    # Find target
    target = None
    # Combined codes must be checked first
    target_codes = ['AET_HUM', 'ANTM_CI', 'CI_ESRX', 'CSV_AET', 'FOXA_DIS',
                    'RUS', 'UKR', 'CQ', 'DT', 'JB', 'MOC', 'TOC']

    for code in target_codes:
        if f'_{code}_' in filename:
            target = code
            break

    # For Business Merger, resolve to specific merger topic
    if topic == 'Business Merger' and target in MERGER_TARGET_TO_TOPIC:
        topic = MERGER_TARGET_TO_TOPIC[target]

    return topic, target


# Valid stance labels for API schema
VALID_STANCE_LABELS = ["Pro", "Against", "Neutral", "Unrelated"]

# Target configurations for generating system instructions
TARGET_CONFIG = {
    "Russian Ukrainian Conflict": {
        "RUS": {
            "name": "Russia's position in the Russian-Ukrainian conflict",
            "description": "Russia's position in the conflict, including justifications for military actions"
        },
        "UKR": {
            "name": "Ukraine's position in the Russian-Ukrainian conflict",
            "description": "Ukraine's position in the conflict, including resistance to invasion"
        }
    },
    "False COVID Treatment": {
        "CQ": {
            "name": "Chloroquine as COVID-19 treatment",
            "description": "Using chloroquine or hydroxychloroquine to treat COVID-19"
        }
    },
    "Taiwan Question": {
        "MOC": {
            "name": "Mainland China's position on Taiwan",
            "description": "China's claim that Taiwan is part of China"
        },
        "TOC": {
            "name": "Taiwan's independence position",
            "description": "Taiwan's claim to independence from China"
        }
    },
    "US 2020 Election": {
        "DT": {
            "name": "Donald Trump",
            "description": "Support for Donald Trump as presidential candidate"
        },
        "JB": {
            "name": "Joe Biden",
            "description": "Support for Joe Biden as presidential candidate"
        }
    },
    "Aetna-Humana Merger": {
        "AET_HUM": {
            "name": "Aetna-Humana merger",
            "description": "Support for the Aetna-Humana business merger"
        }
    },
    "Anthem-Cigna Merger": {
        "ANTM_CI": {
            "name": "Anthem-Cigna merger",
            "description": "Support for the Anthem-Cigna business merger"
        }
    },
    "Cigna-Express Scripts Merger": {
        "CI_ESRX": {
            "name": "Cigna-Express Scripts merger",
            "description": "Support for the Cigna-Express Scripts business merger"
        }
    },
    "CVS-Aetna Merger": {
        "CSV_AET": {
            "name": "CVS-Aetna merger",
            "description": "Support for the CVS-Aetna business merger"
        }
    },
    "Fox-Disney Merger": {
        "FOXA_DIS": {
            "name": "Fox-Disney merger",
            "description": "Support for the Fox-Disney business merger"
        }
    }
}


def normalize_stance_label(label: str) -> str:
    """Normalize stance labels to consistent format."""
    return STANCE_MAPPING.get(label, label)


def normalize_topic(topic: str) -> str:
    """Normalize topic names to consistent format."""
    return TOPIC_MAPPING.get(topic, topic)


def get_merger_topics() -> List[str]:
    """Get list of all merger topic names."""
    return [
        'Aetna-Humana Merger',
        'Anthem-Cigna Merger', 
        'Cigna-Express Scripts Merger',
        'CVS-Aetna Merger',
        'Fox-Disney Merger'
    ]


def get_non_merger_topics() -> List[str]:
    """Get list of all non-merger topic names."""
    merger_topics = set(get_merger_topics())
    return [topic for topic in STANDARDIZED_TOPICS.keys() if topic not in merger_topics]
