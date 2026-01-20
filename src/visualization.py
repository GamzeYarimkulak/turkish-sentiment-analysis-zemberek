"""
Visualization utilities for sentiment analysis results.
"""
import matplotlib.pyplot as plt


def plot_performance_metrics(metrics):
    """
    Creates a bar chart visualization of performance metrics.
    
    Args:
        metrics (dict): Dictionary mapping metric names to values (0-1 range)
    """
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), width=0.5)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.title("Performance Metrics", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
