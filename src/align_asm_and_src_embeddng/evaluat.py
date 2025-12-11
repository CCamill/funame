"""
Evaluation script for assembly function naming
Calculates various metrics including exact match, BLEU, edit distance, etc.
"""
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Calculate exact match accuracy"""
    assert len(predictions) == len(references)
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_normalized_edit_distance(predictions: List[str], references: List[str]) -> float:
    """Calculate normalized edit distance (0 = identical, 1 = completely different)"""
    distances = []
    for pred, ref in zip(predictions, references):
        dist = calculate_edit_distance(pred, ref)
        max_len = max(len(pred), len(ref))
        normalized = dist / max_len if max_len > 0 else 0
        distances.append(normalized)
    return np.mean(distances)


def calculate_prefix_match(predictions: List[str], references: List[str], k: int = 3) -> float:
    """Calculate prefix match rate (first k characters)"""
    matches = sum(
        p[:k] == r[:k] 
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


def calculate_token_f1(predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
    """
    Calculate token-level precision, recall, and F1
    Treats function names as sequences of tokens (split by underscore or camelCase)
    """
    def tokenize(name: str) -> set:
        """Tokenize function name"""
        import re
        # Split by underscore and camelCase
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', name)
        return set(t.lower() for t in tokens)
    
    precisions = []
    recalls = []
    f1s = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        
        if len(pred_tokens) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        
        true_positives = len(pred_tokens & ref_tokens)
        
        precision = true_positives / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = true_positives / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def calculate_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    """
    Calculate BLEU score for character-level n-grams
    Simplified version for short function names
    """
    def get_ngrams(text: str, n: int) -> Dict[str, int]:
        ngrams = defaultdict(int)
        for i in range(len(text) - n + 1):
            ngrams[text[i:i+n]] += 1
        return ngrams
    
    scores = []
    
    for pred, ref in zip(predictions, references):
        if len(pred) == 0 or len(ref) == 0:
            scores.append(0.0)
            continue
        
        # Calculate precision for each n-gram level
        precisions = []
        for n in range(1, min(max_n + 1, len(pred) + 1, len(ref) + 1)):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total = sum(pred_ngrams.values())
            
            precision = matches / total if total > 0 else 0
            precisions.append(precision)
        
        if len(precisions) == 0:
            scores.append(0.0)
        else:
            # Geometric mean of precisions
            score = np.exp(np.mean(np.log(np.array(precisions) + 1e-10)))
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref) / len(pred))) if len(pred) > 0 else 0
            scores.append(bp * score)
    
    return np.mean(scores)


def analyze_errors(predictions: List[str], references: List[str], top_k: int = 10) -> Dict:
    """Analyze common error patterns"""
    errors = []
    for pred, ref in zip(predictions, references):
        if pred != ref:
            errors.append({
                'predicted': pred,
                'reference': ref,
                'edit_distance': calculate_edit_distance(pred, ref),
                'len_diff': abs(len(pred) - len(ref))
            })
    
    # Sort by edit distance
    errors.sort(key=lambda x: x['edit_distance'], reverse=True)
    
    return {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(predictions),
        'worst_cases': errors[:top_k],
        'avg_edit_distance': np.mean([e['edit_distance'] for e in errors]) if errors else 0,
        'avg_length_diff': np.mean([e['len_diff'] for e in errors]) if errors else 0
    }


def evaluate_predictions(
    predictions_file: str,
    output_file: str = None
) -> Dict:
    """
    Evaluate predictions from a JSON file
    
    Expected format:
    [
        {
            "predicted_name": "...",
            "true_name": "...",
            ...
        },
        ...
    ]
    """
    # Load predictions
    logger.info(f"Loading predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = [item['predicted_name'] for item in data]
    references = [item['true_name'] for item in data]
    
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    # Calculate metrics
    metrics = {
        'num_samples': len(predictions),
        'exact_match': calculate_exact_match(predictions, references),
        'normalized_edit_distance': calculate_normalized_edit_distance(predictions, references),
        'prefix_match_3': calculate_prefix_match(predictions, references, k=3),
        'prefix_match_5': calculate_prefix_match(predictions, references, k=5),
        'bleu_score': calculate_bleu(predictions, references),
    }
    
    # Token-level F1
    precision, recall, f1 = calculate_token_f1(predictions, references)
    metrics.update({
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    })
    
    # Error analysis
    error_analysis = analyze_errors(predictions, references)
    metrics['error_analysis'] = error_analysis
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)
    logger.info(f"Number of samples: {metrics['num_samples']}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f}")
    logger.info(f"Prefix Match (3 chars): {metrics['prefix_match_3']:.4f}")
    logger.info(f"Prefix Match (5 chars): {metrics['prefix_match_5']:.4f}")
    logger.info(f"BLEU Score: {metrics['bleu_score']:.4f}")
    logger.info(f"Token Precision: {metrics['token_precision']:.4f}")
    logger.info(f"Token Recall: {metrics['token_recall']:.4f}")
    logger.info(f"Token F1: {metrics['token_f1']:.4f}")
    logger.info("\nError Analysis:")
    logger.info(f"Total Errors: {error_analysis['total_errors']}")
    logger.info(f"Error Rate: {error_analysis['error_rate']:.4f}")
    logger.info(f"Avg Edit Distance (errors only): {error_analysis['avg_edit_distance']:.2f}")
    logger.info("\nWorst Cases:")
    for i, case in enumerate(error_analysis['worst_cases'][:5], 1):
        logger.info(f"{i}. Pred: '{case['predicted']}' | True: '{case['reference']}' | "
                   f"Edit Dist: {case['edit_distance']}")
    logger.info("="*50)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate function name predictions")
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for evaluation results'
    )
    args = parser.parse_args()
    
    evaluate_predictions(args.predictions, args.output)


if __name__ == "__main__":
    main()