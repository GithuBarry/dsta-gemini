"""Simple functions for evaluating stance detection predictions."""

import json
from typing import Dict, List, Tuple
from collections import defaultdict

from ..config import normalize_stance_label


def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score."""
    if len(true_labels) != len(pred_labels):
        raise ValueError("True and predicted labels must have same length")
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels) if true_labels else 0.0
    
    # Per-class metrics
    label_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == pred_label:
            label_counts[true_label]['tp'] += 1
        else:
            label_counts[true_label]['fn'] += 1
            label_counts[pred_label]['fp'] += 1
    
    # Calculate per-class metrics
    class_metrics = {}
    precisions, recalls, f1s = [], [], []
    
    for label, counts in label_counts.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Macro averages
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_f1': accuracy,  # Micro F1 equals accuracy for multi-class
        'per_class_metrics': class_metrics
    }


def extract_predictions(results: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract ground truth and predicted labels from results."""
    true_labels = []
    pred_labels = []
    
    for result in results:
        true_stance = normalize_stance_label(result['ground_truth_stance'])
        
        # Get primary prediction (first one if multiple)
        predictions = result.get('prediction')
        if predictions and len(predictions) > 0:
            pred_stance = normalize_stance_label(predictions[0]['stance'])
        else:
            pred_stance = 'Unrelated'
        
        true_labels.append(true_stance)
        pred_labels.append(pred_stance)
    
    return true_labels, pred_labels


def evaluate_predictions(results: List[Dict]) -> Dict:
    """Evaluate predictions from a results list."""
    true_labels, pred_labels = extract_predictions(results)
    metrics = calculate_metrics(true_labels, pred_labels)
    
    # Add some summary info
    evaluation_summary = {
        'total_samples': len(results),
        'metrics': metrics,
        'label_distribution': {
            'ground_truth': {label: true_labels.count(label) for label in set(true_labels)},
            'predictions': {label: pred_labels.count(label) for label in set(pred_labels)}
        }
    }
    
    return evaluation_summary


def print_evaluation_report(evaluation: Dict):
    """Print a formatted evaluation report."""
    metrics = evaluation['metrics']
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total Samples: {evaluation['total_samples']}")
    print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.3f}")
    print(f"Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"Macro Recall: {metrics['macro_recall']:.3f}")
    
    print(f"\n{'Per-Class Metrics':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print(f"{'-'*60}")
    
    for label, class_metrics in metrics['class_metrics'].items():
        print(f"{label:<20} {class_metrics['precision']:<10.3f} {class_metrics['recall']:<10.3f} "
              f"{class_metrics['f1']:<10.3f} {class_metrics['support']:<10}")
    
    print(f"\nLabel Distribution:")
    print(f"Ground Truth: {evaluation['label_distribution']['ground_truth']}")
    print(f"Predictions:  {evaluation['label_distribution']['predictions']}")
    print(f"{'='*60}\n")


class StanceEvaluator:
    """Wrapper class for stance evaluation functionality."""
    
    def __init__(self):
        self.evaluation_df = None
        self.last_evaluation = None
        
    def evaluate_predictions(self, results: List[Dict]) -> Dict:
        """Evaluate predictions and store results."""
        import pandas as pd
        
        # Convert results to DataFrame for analysis
        evaluation_data = []
        for result in results:
            eval_row = {
                'tweet_id': result['tweet_id'],
                'ground_truth_stance': normalize_stance_label(result['ground_truth_stance']),
                'predicted_stance': 'Unrelated',
                'processing_time': result.get('processing_time', 0),
                'has_image': result.get('image_path') is not None,
                'topic': result.get('ground_truth_topic', 'Unknown'),
                'target': result.get('ground_truth_target', 'Unknown')
            }
            
            # Get predicted stance
            predictions = result.get('prediction')
            if predictions and len(predictions) > 0:
                eval_row['predicted_stance'] = normalize_stance_label(predictions[0].get('stance', 'Unrelated'))
            
            evaluation_data.append(eval_row)
        
        self.evaluation_df = pd.DataFrame(evaluation_data)
        
        # Calculate overall metrics
        true_labels, pred_labels = extract_predictions(results)
        overall_metrics = calculate_metrics(true_labels, pred_labels)
        
        # Calculate topic-specific metrics
        by_topic = {}
        if self.evaluation_df is not None:
            for topic in self.evaluation_df['topic'].unique():
                topic_df = self.evaluation_df[self.evaluation_df['topic'] == topic]
                topic_true = topic_df['ground_truth_stance'].tolist()
                topic_pred = topic_df['predicted_stance'].tolist()
                by_topic[topic] = calculate_metrics(topic_true, topic_pred)
        
        evaluation_summary = {
            'total_samples': len(results),
            'overall': overall_metrics,
            'by_topic': by_topic,
            'label_distribution': {
                'ground_truth': {label: true_labels.count(label) for label in set(true_labels)},
                'predictions': {label: pred_labels.count(label) for label in set(pred_labels)}
            }
        }
        
        self.last_evaluation = evaluation_summary
        return evaluation_summary
    
    def save_evaluation_report(self, name: str) -> Dict[str, str]:
        """Save evaluation report to files."""
        from pathlib import Path
        import json
        
        if not self.last_evaluation:
            return {'error': 'No evaluation results to save'}
        
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_file = output_dir / f"{name}_evaluation.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.last_evaluation, f, indent=2, ensure_ascii=False)
        
        # Save CSV if DataFrame exists
        csv_file = None
        if self.evaluation_df is not None:
            csv_file = output_dir / f"{name}_results.csv"
            self.evaluation_df.to_csv(csv_file, index=False)
        
        return {
            'evaluation_json': str(json_file),
            'results_csv': str(csv_file) if csv_file else None
        }