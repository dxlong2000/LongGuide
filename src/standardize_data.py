"""
Script to standardize all dataset formats to use 'input' and 'output' fields.
"""

import json
import os
from pathlib import Path

def standardize_samsum(data):
    """Convert SAMSum format: dialogue -> input, summary -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item["dialogue"],
            "output": item["summary"]
        })
    return standardized

def standardize_cnn(data):
    """Convert CNN format: article -> input, highlights -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item["article"],
            "output": item["highlights"]
        })
    return standardized

def standardize_xlsum(data):
    """Convert XLSUM format: text -> input, target -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item["text"],
            "output": item["target"]
        })
    return standardized

def standardize_swipe(data):
    """Convert SWiPE format: r_content -> input, s_content -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item["r_content"],
            "output": item["s_content"]
        })
    return standardized

def standardize_iwslt(data):
    """Convert IWSLT format: translation.en -> input, translation.ja -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item["translation"]["en"],
            "output": item["translation"]["ja"]
        })
    return standardized

def standardize_commongen(data):
    """Convert CommonGen format: concepts -> input, target -> output"""
    standardized = []
    for item in data:
        # Join concepts with commas for input
        concepts_str = ", ".join(item["concepts"])
        standardized.append({
            "input": concepts_str,
            "output": item["target"]
        })
    return standardized

def standardize_synthetic_dialogue(data):
    """Convert SyntheticDialogue format: context -> input, response -> output"""
    standardized = []
    for item in data:
        standardized.append({
            "input": item.get("context", item.get("prompt", "")),
            "output": item.get("response", item.get("dialogue", ""))
        })
    return standardized

def standardize_dataset(dataset_name, file_path):
    """Standardize a dataset file"""
    print(f"Processing {dataset_name}: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if dataset_name == "SAMSum":
        standardized = standardize_samsum(data)
    elif dataset_name == "CNN":
        standardized = standardize_cnn(data)
    elif dataset_name == "xlsum":
        standardized = standardize_xlsum(data)
    elif dataset_name == "SWiPE":
        standardized = standardize_swipe(data)
    elif dataset_name == "IWSLT":
        standardized = standardize_iwslt(data)
    elif dataset_name == "CommonGen":
        standardized = standardize_commongen(data)
    elif dataset_name == "SyntheticDialogue":
        standardized = standardize_synthetic_dialogue(data)
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(standardized, f, indent=2, ensure_ascii=False)
    
    print(f"Standardized {len(standardized)} examples for {dataset_name}")

def main():
    """Main function to standardize all datasets"""
    data_dir = Path("data")
    
    datasets = [
        "SAMSum", "CNN", "xlsum", "SWiPE", 
        "IWSLT", "CommonGen", "SyntheticDialogue"
    ]
    
    for dataset in datasets:
        dataset_path = data_dir / dataset
        if dataset_path.exists():
            for json_file in dataset_path.glob("*.json"):
                standardize_dataset(dataset, json_file)
        else:
            print(f"Dataset directory not found: {dataset_path}")

if __name__ == "__main__":
    main()
