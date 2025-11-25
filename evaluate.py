#!/usr/bin/env python3

import json
import argparse
import string
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

def rouge_translation(prediction, ground_truth):
    xlingual_tokenizer = GPTTokenizer()
    xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer)
    scores = xlingual_rouge_scorer.score(prediction=str(prediction), target=str(ground_truth))
    return scores["rougeL"].fmeasure

def rouge_standard(prediction, ground_truth):
    default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = default_rouge_scorer.score(prediction=str(prediction), target=str(ground_truth))
    return scores["rougeL"].fmeasure

def evaluate_rouge_l(results_file, task_type="summarization"):
    """Evaluate ROUGE-L scores for generated results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    rouge_scores = []
    
    for result in results:
        target = result['target']
        generated = result['generated']
        
        if task_type == "translation":
            score = rouge_translation(generated, target)
        else:
            score = rouge_standard(generated, target)
        
        rouge_scores.append(score)
    
    # Calculate average ROUGE-L
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
    
    print(f"ROUGE-L Score: {avg_rouge_l:.4f}")
    print(f"Evaluated {len(rouge_scores)} examples")
    
    return avg_rouge_l, rouge_scores

def main():
    parser = argparse.ArgumentParser(description='Evaluate ROUGE-L scores')
    parser.add_argument('--results', required=True, help='Path to results JSON file')
    parser.add_argument('--task', default='summarization', help='Task type (summarization, translation, etc.)')
    
    args = parser.parse_args()
    evaluate_rouge_l(args.results, args.task)

if __name__ == "__main__":
    main()
