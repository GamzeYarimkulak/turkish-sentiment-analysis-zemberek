"""
Sentiment analysis module using rule-based approach with Zemberek morphological analysis.
"""
from collections import defaultdict
from zemberek.tokenization import TurkishTokenizer

from .data_preprocessing import preprocess_text


def analyze_sentiment(sentence, morphology, positive_dict, negative_dict):
    """
    Analyzes the sentiment of a Turkish sentence using a rule-based approach.
    
    The method uses morphological analysis to extract word roots and matches them
    against positive/negative word dictionaries. It also handles negation through
    predicate (verb) analysis using a heuristic approach.
    
    Args:
        sentence (str): The Turkish sentence to analyze
        morphology: TurkishMorphology instance for morphological analysis
        positive_dict (dict): Dictionary of positive word roots with weights
        negative_dict (dict): Dictionary of negative word roots with weights
        
    Returns:
        dict: Analysis result containing:
            - sentiment (str): "Positive", "Negative", or "Neutral"
            - score (float): Numerical sentiment score
            - confidence (float): Confidence score (0-1)
            - features (dict): Found positive/negative words and predicates
            - predicate_analysis (dict): Predicate analysis info if found
    """
    try:
        sentence = preprocess_text(sentence)
        raw_tokens = TurkishTokenizer.DEFAULT.tokenize(sentence)
        tokens = [t.content.lower() for t in raw_tokens if t.content.strip() != '']
        
        # Perform morphological analysis once
        results = morphology.analyze_and_disambiguate(" ".join(tokens)).best_analysis()
        
        # Heuristic negation detection through predicate analysis
        # This is a rule-based approach that checks for negation markers in verbs
        predicate_multiplier = 1
        predicate_info = None
        
        # Search for verbs in reverse order (typically at the end of sentence)
        for analysis in reversed(results):
            morphemes = str(analysis)
            if 'Verb' in morphemes:
                has_negation = 'Neg' in morphemes
                predicate_info = {
                    'root': analysis.item.root,
                    'is_negative': has_negation,
                    'full_analysis': str(analysis)
                }
                # Apply negation multiplier if negation marker found
                if has_negation:
                    predicate_multiplier = -1 * predicate_multiplier
                break

        score = 0
        confidence = 0
        found_features = defaultdict(list)

        # Add predicate information if found
        if predicate_info:
            found_features['predicate'] = [{
                'root': predicate_info['root'],
                'is_negative': predicate_info['is_negative'],
                'analysis': predicate_info['full_analysis']
            }]

        # Analyze each word in the sentence
        for result in results:
            root = result.item.root or result.item.normalized_form
            root = root.lower() if root else ""
            
            if root in positive_dict:
                score += positive_dict[root] * predicate_multiplier
                confidence += 1
                found_features['positive_words'].append(root)
            elif root in negative_dict:
                score += negative_dict[root] * predicate_multiplier * -1
                confidence += 1
                found_features['negative_words'].append(root)

        # Calculate confidence score
        confidence_score = confidence / len(tokens) if tokens else 0
        
        # Determine sentiment based on score
        if score > 0:
            sentiment = "Positive"
        elif score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence_score,
            'features': dict(found_features),
            'predicate_analysis': predicate_info
        }
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {
            'sentiment': "Error",
            'score': 0,
            'confidence': 0,
            'features': {},
            'predicate_analysis': None
        }


def evaluate_model(test_data, morphology, positive_dict, negative_dict):
    """
    Evaluates the sentiment analysis model on test data.
    
    Args:
        test_data: List of tuples (sentence, true_label)
        morphology: TurkishMorphology instance
        positive_dict (dict): Dictionary of positive word roots
        negative_dict (dict): Dictionary of negative word roots
        
    Returns:
        dict: Evaluation results containing:
            - DP: True Positives (Correct Positive predictions)
            - DN: True Negatives (Correct Negative predictions)
            - YP: False Positives (Incorrect Positive predictions)
            - YN: False Negatives (Incorrect Negative predictions)
            - predictions: List of all predictions
            - wrong_predictions: List of incorrect predictions
    """
    results = {
        'DP': 0,  # Doğru Pozitif (True Positive)
        'DN': 0,  # Doğru Negatif (True Negative)
        'YP': 0,  # Yanlış Pozitif (False Positive)
        'YN': 0,  # Yanlış Negatif (False Negative)
        'predictions': [],
        'wrong_predictions': []
    }
    
    for sentence, true_label in test_data:
        analysis = analyze_sentiment(sentence, morphology, positive_dict, negative_dict)
        prediction = analysis['sentiment']

        if prediction == "Positive" and true_label == "Pozitif":
            results['DP'] += 1
        elif prediction == "Negative" and true_label == "Negatif":
            results['DN'] += 1
        elif prediction == "Positive" and true_label == "Negatif":
            results['YP'] += 1
            results['wrong_predictions'].append({
                'text': sentence,
                'true_label': true_label,
                'predicted': prediction,
                'confidence': analysis['confidence']
            })
        elif prediction == "Negative" and true_label == "Pozitif":
            results['YN'] += 1
            results['wrong_predictions'].append({
                'text': sentence,
                'true_label': true_label,
                'predicted': prediction,
                'confidence': analysis['confidence']
            })
        elif prediction == "Neutral":
            # Handle neutral predictions (count as incorrect for binary classification)
            if true_label == "Pozitif":
                results['YN'] += 1
            else:
                results['YP'] += 1
            results['wrong_predictions'].append({
                'text': sentence,
                'true_label': true_label,
                'predicted': prediction,
                'confidence': analysis['confidence']
            })

        results['predictions'].append({
            'text': sentence,
            'true_label': true_label,
            'predicted': prediction,
            'confidence': analysis['confidence'],
            'features': analysis['features']
        })
    
    return results
