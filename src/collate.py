from typing import List, Dict
import torch

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable length data"""
    # Initialize the output dictionary with empty lists
    output = {
        'id': [],
        'dataset': [],
        'question': [],
        'reasoning': [],
        'label': [],
        'input_text': [],
        'target_text': []
    }
    
    # Collect each field from all samples
    for sample in batch:
        for key in output.keys():
            output[key].append(sample[key])
            
    return output

def collate_fn_test(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable length data"""
    # Initialize the output dictionary with empty lists
    output = {
        'id': [],
        'dataset': [],
        'question': [],
        'label': [],
        'input_text': [],
    }
    
    # Collect each field from all samples
    for sample in batch:
        for key in output.keys():
            output[key].append(sample[key])
            
    return output