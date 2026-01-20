"""
Project constants and path configurations.
"""
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Zemberek JAR path (configurable - users should set this in their environment or config)
# Default placeholder that users need to update
ZEMBEREK_PATH = "path/to/zemberek-full.jar"  # Update this to your Zemberek JAR location

# Data file paths
POSITIVE_WORDS_FILE = DATA_DIR / "positive_words.txt"
NEGATIVE_WORDS_FILE = DATA_DIR / "negative_words.txt"
LABELED_SENTENCES_FILE = DATA_DIR / "labeled_sentences.xlsx"
