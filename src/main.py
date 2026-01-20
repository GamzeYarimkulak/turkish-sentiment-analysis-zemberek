"""
Main entry point for Turkish Sentiment Analysis application.
"""
from jpype import startJVM, shutdownJVM, isJVMStarted
from zemberek.morphology import TurkishMorphology
import pandas as pd

from .constants import ZEMBEREK_PATH, POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE, LABELED_SENTENCES_FILE
from .data_preprocessing import preprocess_txt_words
from .sentiment_analysis import analyze_sentiment, evaluate_model
from .visualization import plot_performance_metrics


def main():
    """
    Main function that runs the sentiment analysis application.
    
    - Starts JVM and initializes Turkish morphology analyzer
    - Loads positive/negative word dictionaries
    - Provides interactive sentence analysis
    - Evaluates model on test dataset
    - Displays performance metrics
    """
    try:
        # Start JVM with Zemberek JAR
        startJVM("-Djava.class.path=" + ZEMBEREK_PATH)
        morphology = TurkishMorphology.create_with_defaults()
        
        # Load word dictionaries
        positive_dict = preprocess_txt_words(str(POSITIVE_WORDS_FILE), morphology)
        negative_dict = preprocess_txt_words(str(NEGATIVE_WORDS_FILE), morphology)

        print("Enter sentences to analyze. Type 'q' to quit.")
        while True:
            user_input = input("Sentence: ").strip()
            if user_input.lower() == 'q':
                break
            if user_input:
                analysis = analyze_sentiment(
                    sentence=user_input,
                    morphology=morphology,
                    positive_dict=positive_dict,
                    negative_dict=negative_dict,
                )
                print(f"\nSentence: {user_input}")
                print(f"Sentiment: {analysis['sentiment']}")
                print(f"Confidence Score: {analysis['confidence']:.2f}")
                if analysis['predicate_analysis']:
                    print(f"Predicate Analysis: {analysis['predicate_analysis']}")
                print(f"Found Features: {analysis['features']}")

        # Load test data and evaluate model
        df = pd.read_excel(str(LABELED_SENTENCES_FILE))
        test_data = list(zip(df["Cümle"], df["Sınıf"]))

        results = evaluate_model(
            test_data=test_data,
            morphology=morphology,
            positive_dict=positive_dict,
            negative_dict=negative_dict,
        )

        # Calculate performance metrics
        total = sum(results[k] for k in ['DP', 'DN', 'YP', 'YN'])
        accuracy = (results['DP'] + results['DN']) / total if total > 0 else 0
        precision = results['DP'] / (results['DP'] + results['YP']) if (results['DP'] + results['YP']) > 0 else 0
        recall = results['DP'] / (results['DP'] + results['YN']) if (results['DP'] + results['YN']) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        # Print performance metrics as percentages
        print("\nPerformance Metrics (Percentage):")
        for metric, value in metrics.items():
            print(f"{metric}: {value * 100:.2f}%")

        # Visualize performance metrics
        plot_performance_metrics(metrics)

        # Print incorrect predictions
        print("\nIncorrect Predictions:")
        for wp in results['wrong_predictions']:
            print(f"  - Sentence: {wp['text']}")
            print(f"    True Label: {wp['true_label']}")
            print(f"    Predicted: {wp['predicted']}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if isJVMStarted():
            shutdownJVM()
            print("JVM shut down.")


if __name__ == "__main__":
    main()
