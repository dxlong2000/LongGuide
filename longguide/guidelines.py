"""
Metrics generation and evaluation for LongGuide.

This module implements metric generation, linguistic attribute extraction,
and evaluation functions for the LongGuide framework.
"""

import re
import random
import math
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from collections import Counter
from .llm_client import LLMClient

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class MetricsGuidelines:
    """Generates task-specific and linguistic guidelines for LongGuide."""
    
    # Pre-defined assessment metrics from research papers and proposals
    PRE_DEFINED_METRICS = [
        # Basic metrics (3)
        "Accuracy", "Brevity", "Clarity",
        # BARTScore metrics (2) 
        "Relevance", "Coherence",
        # GPTScore metrics (10)
        "Semantic Coverage", "Factuality", "Fluency", "Informativeness",
        "Consistency", "Engagement", "Specificity", "Correctness",
        "Understandability", "Diversity",
        # Proposed metrics (12)
        "Completeness", "Conciseness", "Neutrality", "Naturalness", 
        "Readability", "Creativity", "Rationalness", "Truthfulness", 
        "Respect of Chronology", "Non-repetitiveness", "Indicativeness", "Resolution"
    ]
    
    def __init__(self, task_type: str, config: dict = None):
        self.task_type = task_type
        self.config = config or {}
        self.llm_client = LLMClient(
            model_name=self.config.get('model_name', 'gpt-3.5-turbo'),
            api_key=self.config.get('api_key')
        )
    
    def generate_metrics(self, validation_data: List[Dict], 
                            batch_size: int, num_iterations: int) -> List[str]:
        """
        Collect task-specific evaluation metrics using iterative sampling.
        
        Args:
            validation_data: Validation dataset for context
            batch_size: Number of examples to sample per iteration
            num_iterations: Number of iterations to run
            
        Returns:
            List of collected metrics
        """
        iter_valid_data = validation_data.copy()
        collected_metrics = set()
        
        for _ in range(num_iterations):
            selected_valid = random.sample(iter_valid_data, batch_size)
            iter_valid_data = [elem for elem in iter_valid_data if elem not in selected_valid]

            demonstration_string = ""
            for idx in range(len(selected_valid)):
                input_content = selected_valid[idx].get("input", selected_valid[idx].get("r_content", ""))
                output_content = selected_valid[idx].get("reference", selected_valid[idx].get("s_content", ""))
                demonstration_string += f"""INPUT {idx+1}: {str(input_content)}\nOUTPUT {idx+1}: {str(output_content)}\n\n"""

            collecting_metrics_prompt = f"""Select top-5 metrics which are the most important from the list below to evaluate a special way of {self.task_type}.
{str(self.PRE_DEFINED_METRICS)}

Here are some demonstrations of the task {self.task_type}:
{demonstration_string}

Output your list of metrics in JSON block including ```json and ```:
```json
["metric1", "metric2", "metric3", "metric4", "metric5"]
```"""

            list_metrics = self.llm_client.generate(collecting_metrics_prompt, max_tokens=512)
            try: 
                import json
                import re
                # Extract JSON from code block
                json_match = re.search(r'```json\s*(.*?)\s*```', list_metrics, re.DOTALL)
                if json_match:
                    metrics_list = json.loads(json_match.group(1))
                    collected_metrics.update(metrics_list)
            except: 
                continue

        collected_metrics = list(collected_metrics)
        collected_metrics.sort()
        return collected_metrics
    
    def get_guidelines(self, validation_data: List[Dict] = None, batch_size: int = 10, num_iterations: int = 3) -> str:
        """Get metrics guidelines following the 3-step process."""
        if not validation_data:
            return "Focus on accuracy, clarity, and relevance for the task."
        
        # Step 1: Collecting metrics
        collected_metrics = self.generate_metrics(validation_data, batch_size, num_iterations)
        
        # Step 2: Collecting metrics' scores
        mc_collected_scores = self.generate_llmjudge_scores(validation_data, collected_metrics)
        
        # Step 3: Collecting the metrics' definitions
        metrics_string = ", ".join(list(collected_metrics))
        collecting_definitions_prompt = f"""Now you are given the following metrics: {metrics_string} for the {self.task_type} task.
Based on these scores on a scale of 5 for the quality of a generated text: {str(mc_collected_scores)}, define the expected quality of the generated text for each metric in natural language. Give me the list in bullet points."""
        
        raw_metrics_definitions = self.llm_client.generate(collecting_definitions_prompt, max_tokens=1024)
        if "\n\n" in raw_metrics_definitions:
            raw_metrics_definitions = raw_metrics_definitions.split("\n\n")[1].strip()
        
        return raw_metrics_definitions
    
    def generate_llmjudge_scores(self, validation_data: List[Dict], collected_metrics: List[str]) -> Dict[str, float]:
        """
        Collect LLM judge scores for metrics on validation data.
        
        Args:
            validation_data: Validation dataset
            collected_metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        evaluation_format = {key: "1-5" for key in collected_metrics}
        metrics_definitions = self._get_metric_definitions(collected_metrics)
        
        mc_collected_scores = {metric: 0 for metric in collected_metrics}
        
        random.shuffle(validation_data)
        valid_update_cnt = 0
        
        for dt in validation_data:
            input_content = str(dt.get("input", dt.get("r_content", "")))
            target_content = str(dt.get("reference", dt.get("s_content", "")))
            
            evaluation_prompt = f"""You are given an input, and an output of a {self.task_type} task.
Input: {input_content}
Output: {target_content}

Your task is to evaluate the following criteria in a scale of 1-5, with 1 is worst and 5 is best.
{evaluation_format}

The definitions of the criteria are:
{metrics_definitions}

Your output must be in JSON block including ```json and ```:
```json
{{"metric1": 3, "metric2": 4, ...}}
```"""
            try:
                evaluation_outcome = self.llm_client.generate(evaluation_prompt, max_tokens=512)
                # Extract JSON from response
                import json
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', evaluation_outcome, re.DOTALL)
                if json_match:
                    eva_outcome = json.loads(json_match.group(1))
                    for metric in eva_outcome:
                        if metric in mc_collected_scores:
                            mc_collected_scores[metric] += eva_outcome[metric]
                    valid_update_cnt += 1
            except:
                continue
        
        if valid_update_cnt >= 1:
            for metric in mc_collected_scores:
                mc_collected_scores[metric] = mc_collected_scores[metric] / valid_update_cnt
        else:
            for metric in mc_collected_scores:
                mc_collected_scores[metric] = 5
        
        return mc_collected_scores
    
    def _get_metric_definitions(self, metrics: List[str]) -> str:
        """Get detailed definitions for metrics using LLM."""
        input_prompt = f"""Define the list of following metrics in details as the quality of the generation expected for the {self.task_type} task.
{metrics}
Give me the list in bullet points.
"""
        return self.llm_client.generate(input_prompt, max_tokens=1024)
    

class OutputConstraintsGuidelines:
    """Generates output constraints and guidelines for LongGuide."""
    
    def __init__(self, task_type: str, config: dict = None):
        self.task_type = task_type
        self.config = config or {}
        self.llm_client = LLMClient(
            model_name=self.config.get('model_name', 'gpt-3.5-turbo'),
            api_key=self.config.get('api_key')
        )
    
    def count_sentences(self, text: str) -> int:
        return len(sent_tokenize(text))
    
    def count_words(self, text: str) -> int:
        words = word_tokenize(text)
        return len(words)
    
    def count_verbs(self, text: str) -> int:
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        verb_count = sum(1 for word, pos in tagged_words if pos.startswith('VB'))
        return verb_count
    
    def count_nouns(self, text: str) -> int:
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        noun_count = sum(1 for word, pos in tagged_words if pos.startswith('NN'))
        return noun_count
    
    def get_majority_voice(self, paragraph: str) -> str:
        sentences = sent_tokenize(paragraph)
        all_verbs = [pos_tag(word_tokenize(sentence)) for sentence in sentences]
        verb_tags = [pos for sentence_pos_tags in all_verbs for _, pos in sentence_pos_tags if pos.startswith('VB')]
        voice_counts = Counter(verb_tags)
        majority_voice = max(voice_counts, key=voice_counts.get)
        return majority_voice
    
    def generate_linguistic_statistics(self, validation_data: List[Dict]) -> Dict[str, float]:
        """Generate linguistic statistics from validation data."""
        word_cnt = 0
        verb_cnt = 0
        noun_cnt = 0
        sentence_cnt = 0
        
        for dt in validation_data:
            content = dt["output"]
            word_cnt += self.count_words(content)
            verb_cnt += self.count_verbs(content)
            noun_cnt += self.count_nouns(content)
            sentence_cnt += self.count_sentences(content)
        
        return {
            "words": int(word_cnt / len(validation_data)),
            "sentences": sentence_cnt / len(validation_data),
            "verbs": verb_cnt / len(validation_data),
            "nouns": noun_cnt / len(validation_data)
        }
    
    def construct_linguistic_constraints(self, statistics: Dict[str, float]) -> str:
        """Construct linguistic constraints from statistics."""
        return f"""Your response must have {math.floor(statistics["sentences"])} sentences and on average {math.floor(statistics["words"])} words."""
    
    def generate_output_constraints(self, validation_data: List[Dict]) -> Dict[str, Any]:
        """
        Generate Output Constraint Guidelines (OCG) with six key constraints.
        
        Returns:
            Dictionary containing min, max, avg for sentences and tokens
        """
        sentence_counts = []
        token_counts = []
        
        for dt in validation_data:
            content = dt["output"]
            sentence_counts.append(self.count_sentences(content))
            token_counts.append(self.count_words(content))
        
        return {
            "sentences": {
                "min": min(sentence_counts),
                "max": max(sentence_counts), 
                "avg": sum(sentence_counts) / len(sentence_counts)
            },
            "tokens": {
                "min": min(token_counts),
                "max": max(token_counts),
                "avg": sum(token_counts) / len(token_counts)
            }
        }
    
    def get_guidelines(self, validation_data: List[Dict] = None) -> str:
        """Get output constraint guidelines."""
        if not validation_data:
            return "Your response should be well-structured and appropriate for the task."
        
        # Step 1: Get statistics
        constraints = self.generate_output_constraints(validation_data)
        
        # Step 2: Format it
        return self.format_ocg_constraints(constraints)
    
    def format_ocg_constraints(self, constraints: Dict[str, Any]) -> str:
        """
        Format OCG constraints into natural language instruction.
        
        Args:
            constraints: Dictionary from generate_output_constraints()
            
        Returns:
            Formatted constraint string
        """
        return f"""The summary must have from {int(constraints["sentences"]["min"])} to {int(constraints["sentences"]["max"])} sentences and from {int(constraints["tokens"]["min"])} to {int(constraints["tokens"]["max"])} words with an average of {int(constraints["tokens"]["avg"])} words and {int(constraints["sentences"]["avg"])} sentences."""

