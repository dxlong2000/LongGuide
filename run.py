#!/usr/bin/env python3

import yaml
import json
import argparse
from pathlib import Path
from longguide import MetricsGuidelines, OutputConstraintsGuidelines
from standardize_data import standardize_dataset

def get_task_instruction(data_path):
    """Get dataset-specific instruction"""
    dataset_name = Path(data_path).name
    instructions = {
        "SWiPE": "Simplify this text.",
        "SAMSum": "Summarize the following dialogue.",
        "CNN": "Summarize the following news.",
        "xlsum": "Summarize the following document.",
        "IWSLT": "Translate the following from English to Japanese.",
        "CommonGen": "Generate the text from the following table.",
        "SyntheticDialogue": "Generate the next dialogue response."
    }
    return instructions.get(dataset_name, "Process this text.")

def get_data_path(task_type):
    """Map task type to dataset path"""
    # summarization, translation, dialogue generation, table-to-text generation, text simplification
    task_to_dataset = {
        "summarization": "data/SAMSum", # Can be CNN/xlsum/SAMSum
        "translation": "data/IWSLT", 
        "text simplification": "data/SWiPE",
        "table-to-text generation": "data/CommonGen",
        "dialogue generation": "data/SyntheticDialogue",
    }
    return task_to_dataset.get(task_type, "data/SAMSum")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(data_path):
    """Load and standardize dataset from JSON files"""
    data_dir = Path(data_path)
    dataset_name = data_dir.name
    data = []
    
    for json_file in data_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            data.extend(file_data)
    
    # Use existing standardization
    temp_file = f"temp_{dataset_name}.json"
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    
    standardized_data = standardize_dataset(dataset_name, temp_file)
    
    Path(temp_file).unlink()
    return standardized_data

def run_longguide(config_path):
    """Run LongGuide with specified configuration"""
    config = load_config(config_path)
    
    # Auto-determine data path from task type
    data_path = get_data_path(config['task_type'])
    
    # Load dataset
    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} examples from {data_path}")
    
    # Initialize guidelines with task type and config
    metrics_guidelines = MetricsGuidelines(config['task_type'], config)
    constraints_guidelines = OutputConstraintsGuidelines(config['task_type'], config)
    
    # Get task-specific guidelines
    metrics = metrics_guidelines.get_guidelines()
    constraints = constraints_guidelines.get_guidelines(dataset[:10])  # Test with first 10 examples
    
    print(f"Using guidelines for task: {config['task_type']}")
    print(f"Metrics: {metrics}")
    print(f"Constraints: {constraints}")
    
    # Process dataset
    results = []
    for i, item in enumerate(dataset):
        print(f"Processing example {i+1}/{len(dataset)}")
        
        input_text = item['input']
        task_instruction = get_task_instruction(data_path)
        
        # Full attributes prompt
        full_prompt = f"""{task_instruction} Your generated output must strictly fulfill the following task metrics. {constraints}

{metrics}

Input: {input_text}"""
        
        # Only metrics prompt  
        only_metrics_prompt = f"""{task_instruction} Your generated output must strictly fulfill the following task metrics.

{metrics}

Input: {input_text}"""
        
        # Only constraints prompt
        only_constraints_prompt = f"""{task_instruction} {constraints}

Input: {input_text}"""
        
        # TODO: Replace with actual LLM calls
        results.append({
            'input': input_text,
            'target': item['output'],
            'full_attributes_prompt': full_prompt,
            'only_metrics_prompt': only_metrics_prompt,
            'only_constraints_prompt': only_constraints_prompt,
            'full_attributes': f"[Full attributes output for: {input_text[:30]}...]",
            'only_metrics': f"[Only metrics output for: {input_text[:30]}...]", 
            'only_constraints': f"[Only constraints output for: {input_text[:30]}...]"
        })
    
    # Save results
    output_path = f"outputs/results_{config['task_type']}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run LongGuide with configuration')
    parser.add_argument('--config', default='configs/default.yaml', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    run_longguide(args.config)

if __name__ == "__main__":
    main()
