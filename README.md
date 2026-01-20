# Turkish Sentiment Analysis

A rule-based sentiment analysis tool for Turkish text using Zemberek morphological analysis. This project classifies Turkish sentences as positive, negative, or neutral using a dictionary-based approach with morphological root extraction and heuristic negation handling.

## Overview

This project implements sentiment analysis for Turkish language by leveraging Zemberek's morphological analysis capabilities. Instead of using machine learning models, it employs a rule-based approach that:

- Extracts morphological roots from words using Zemberek
- Matches roots against positive and negative word dictionaries
- Handles negation through predicate-based heuristics
- Provides confidence scores based on feature matching

## Features

- Morphological root extraction using Zemberek
- Dictionary-based sentiment classification
- Heuristic negation detection through verb analysis
- Interactive sentence analysis mode
- Model evaluation with standard performance metrics (accuracy, precision, recall, F1)
- Performance visualization

## Requirements

### System Requirements

- Python 3.7 or higher
- Java 8 or higher (required for Zemberek)
- Zemberek JAR file

### Python Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- jpype1 (for Java-Python bridge)
- pandas (for data handling)
- matplotlib (for visualization)
- openpyxl (for Excel file support)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd piton_proje
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download Zemberek JAR file:
   - Download `zemberek-full.jar` from the official Zemberek repository
   - Place it in a location accessible to your system

4. Configure Zemberek path:
   - Open `src/constants.py`
   - Update `ZEMBEREK_PATH` with the path to your `zemberek-full.jar` file:
   ```python
   ZEMBEREK_PATH = "path/to/your/zemberek-full.jar"
   ```

## Usage

### Interactive Mode

Run the main script to enter interactive analysis mode:

```bash
python -m src.main
```

You can then type sentences for analysis. Type 'q' to quit and proceed to model evaluation.

### Programmatic Usage

You can also use the sentiment analysis functions directly:

```python
from jpype import startJVM, shutdownJVM
from zemberek.morphology import TurkishMorphology
from src.constants import ZEMBEREK_PATH, POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE
from src.data_preprocessing import preprocess_txt_words
from src.sentiment_analysis import analyze_sentiment

# Initialize
startJVM("-Djava.class.path=" + ZEMBEREK_PATH)
morphology = TurkishMorphology.create_with_defaults()
positive_dict = preprocess_txt_words(str(POSITIVE_WORDS_FILE), morphology)
negative_dict = preprocess_txt_words(str(NEGATIVE_WORDS_FILE), morphology)

# Analyze a sentence
result = analyze_sentiment(
    sentence="Bu film gerçekten harikaydı",
    morphology=morphology,
    positive_dict=positive_dict,
    negative_dict=negative_dict
)

print(result['sentiment'])  # Output: Positive
print(result['score'])
print(result['confidence'])

shutdownJVM()
```

## Project Structure

```
piton_proje/
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── sentiment_analysis.py   # Core sentiment analysis logic
│   ├── data_preprocessing.py   # Text preprocessing utilities
│   ├── visualization.py        # Performance visualization
│   └── constants.py            # Path and configuration constants
│
├── data/
│   ├── positive_words.txt      # Positive word dictionary
│   ├── negative_words.txt      # Negative word dictionary
│   └── labeled_sentences.xlsx  # Test dataset with labeled sentences
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How It Works

1. **Preprocessing**: Input text is normalized (lowercased, special characters removed) and tokenized.

2. **Morphological Analysis**: Each word is analyzed using Zemberek to extract its root form.

3. **Dictionary Matching**: Extracted roots are matched against positive and negative word dictionaries.

4. **Negation Handling**: The system searches for verbs with negation markers (Neg morpheme) and applies a multiplier to invert sentiment when negation is detected.

5. **Score Calculation**: 
   - Positive word matches add to the score
   - Negative word matches subtract from the score
   - Negation multiplies the effect by -1

6. **Classification**:
   - Score > 0: Positive
   - Score < 0: Negative
   - Score == 0: Neutral

## Dataset

The project includes labeled Turkish sentences in `data/labeled_sentences.xlsx`. This dataset contains sentences with their corresponding sentiment labels (Pozitif/Negatif) and is used for model evaluation.

The word dictionaries (`positive_words.txt` and `negative_words.txt`) contain root forms of positive and negative words, one per line. These are processed through Zemberek to extract canonical roots.

## Limitations

This is a rule-based system with several limitations:

- **Heuristic-based negation**: Negation detection relies on morphological markers in verbs, which may not cover all negation cases in Turkish.
- **No context awareness**: The system doesn't understand sentence context or sarcasm.
- **Dictionary dependency**: Performance heavily depends on the quality and completeness of the word dictionaries.
- **Binary/neutral classification**: The system works best for clearly positive or negative sentiment; neutral cases are determined only when no sentiment-bearing words are found.

## Evaluation Metrics

The evaluation includes standard classification metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall

These metrics are calculated on the labeled test dataset and displayed both numerically and visually.

## Future Improvements

Potential areas for enhancement:

- Expand and refine the word dictionaries
- Improve negation detection with more comprehensive rules
- Add handling for comparative and superlative forms
- Incorporate context-aware features
- Support for multi-class sentiment classification
- Consider hybrid approaches combining rule-based and ML methods

## Contributing

Contributions are welcome. Please ensure that:
- Code follows the existing style and structure
- New features include appropriate documentation
- Tests are updated accordingly

## Acknowledgments

This project uses Zemberek, a Turkish natural language processing library, for morphological analysis. More information about Zemberek can be found at the official repository.
