"""
Data preprocessing utilities for Turkish text analysis.
"""
import re
from zemberek.morphology import TurkishMorphology


def preprocess_text(text):
    """
    Preprocesses text by converting to lowercase and removing special characters.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def preprocess_txt_words(file_path, morphology):
    """
    Processes a text file containing words and reduces them to their roots using
    morphological analysis. Creates a dictionary mapping roots to weights.
    
    Args:
        file_path: Path to the text file containing words (one per line)
        morphology: TurkishMorphology instance for morphological analysis
        
    Returns:
        dict: Dictionary mapping word roots (lowercase) to weights (default 1)
    """
    roots = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    analysis = morphology.analyze_and_disambiguate(word).best_analysis()
                    for result in analysis:
                        root = result.item.root
                        if root:
                            roots.add(root.lower())
        return {root: 1 for root in roots}
    except Exception as e:
        print(f"Error processing text file: {e}")
        return {}
