#!/usr/bin/env python3

import yaml
import json
import argparse
from pathlib import Path
from longguide import LongGuideGenerator

def standardize_data(data, dataset_name):
    """Standardize dataset to input/output format"""
    standardized = []
    
    for item in data:
        if dataset_name == "SAMSum":
            standardized.append({"input": item["dialogue"], "output": item["summary"]})
        elif dataset_name == "CNN":
            standardized.append({"input": item["article"], "output": item["highlights"]})
        elif dataset_name == "xlsum":
            standardized.append({"input": item["text"], "output": item["target"]})
        elif dataset_name == "SWiPE":
            standardized.append({"input": item["r_content"], "output": item["s_content"]})
        elif dataset_name == "IWSLT":
            standardized.append({"input": item["translation"]["en"], "output": item["translation"]["ja"]})
        elif dataset_name == "CommonGen":
            concepts_str = ", ".join(item["concepts"])
            standardized.append({"input": concepts_str, "output": item["target"]})
        elif dataset_name == "SyntheticDialogue":
            standardized.append({"input": item.get("context", item.get("prompt", "")), "output": item.get("response", item.get("dialogue", ""))})
        else:
            # Assume already standardized
            standardized.append(item)
    
    return standardized

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
    
    # Standardize the data
    data = standardize_data(data, dataset_name)
    return data

def get_data_path(task_type):
    """Map task type to dataset path"""
    task_to_dataset = {
        "summarization": "data/SAMSum",  # or CNN, xlsum
        "translation": "data/IWSLT", 
        "text_simplification": "data/SWiPE",
        "concept_to_text": "data/CommonGen",
        "dialogue_generation": "data/SyntheticDialogue"
    }
    return task_to_dataset.get(task_type, "data/SAMSum")

def run_longguide(config_path):
    """Run LongGuide with specified configuration"""
    config = load_config(config_path)
    
    # Auto-determine data path from task type
    data_path = get_data_path(config['task_type'])
    
    # Load dataset
    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} examples from {data_path}")
    
    # Initialize LongGuide
    generator = LongGuideGenerator(
        model_name=config['model_name'],
        api_key=config['api_key'],
        task_type=config['task_type']
    )
    
    # Process dataset
    results = []
    for i, item in enumerate(dataset):
        print(f"Processing example {i+1}/{len(dataset)}")
        
        result = generator.generate(
            input_text=item['input'],
            target_output=item['output']
        )
        
        results.append({
            'input': item['input'],
            'target': item['output'],
            'generated': result
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
