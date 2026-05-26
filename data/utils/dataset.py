from torch.utils.data import Dataset
import pandas as pd
import random
from typing import List, Dict, Tuple
import os
import ast


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = f'{project_root}'

def get_prompt(question: str) -> str:
    """construct prompt"""
    # prompt = f"""Question:{question}
    # Provide the final answer (letter or number) only.
    # Format your response as:
    # <answer> only your final answer (letter or number) </answer>
    # """
 
    # return prompt
    return (
        f"Question: {str(question).strip()}\n"
        "Instruction:\n"
        "1. Carefully think step by step to determine the correct answer.\n"
        "2. Show your graph reasoning in a <reasoning> block.\n"
        "3. Each <step> must be one edge: source node -> leads to -> target node.\n"
        "4. After reasoning, always provide the final answer in <answer> tag.\n"
        "5. Format strictly as:\n"
        "<reasoning>\n"
        "<step>node A -> leads to -> node B</step>\n"
        "...\n"
        "</reasoning>\n"
        "<answer>...</answer>\n"
        "5. Do NOT generate anything outside <reasoning> and <answer>.\n"
    )

def get_dataset(dataset_name):
    if dataset_name == 'logiqa':
        result_df = pd.read_csv(f'{PATH}/graph_data/graph_reasoning_data.csv')
        questions = result_df['Question'].values
        reasonings = result_df['Optimal Reasoning'].values
        labels = result_df['Label'].values
        return questions, reasonings, labels

    if dataset_name == 'logiqa_test':
        result_df = pd.read_csv(f'{PATH}/datasets/benchmarks/logiqa/logiqa_test.csv')
        questions = result_df['Question'].values
        labels = result_df['Label'].values
        return questions, labels

    if dataset_name == 'aiw_easy':
        result_df = pd.read_pickle(f'{PATH}/datasets/benchmarks/aiw/AIW_easy.pkl')
        questions = result_df['questions'].values
        labels = result_df['answers'].values
        return questions, labels
    
    if dataset_name == 'aiw_hard':
        result_df = pd.read_pickle(f'{PATH}/datasets/benchmarks/aiw+/AIW_hard.pkl')
        questions = result_df['questions'].values
        labels = result_df['answers'].values
        return questions, labels
    
    if dataset_name == 'lasr_ar':
        result_df = pd.read_csv(f'{PATH}/datasets/benchmarks/lsat_ar/lsat-ar.csv')
        questions = result_df['Question'].values
        labels = result_df['Label'].values
        return questions, labels
    
    if dataset_name == 'medqa':
        result_df = pd.read_csv(f'{PATH}/datasets/benchmarks/medqa/medqa.csv')
        questions = result_df['Question'].values
        labels = result_df['Label'].values
        return questions, labels
    
    if dataset_name == 'mathqa':
        result_df = pd.read_csv(f'{PATH}/datasets/benchmarks/mathqa/mathqa.csv')
        questions = result_df['Question'].values
        labels = result_df['Label'].values
        return questions, labels


class ReasoningDataset(Dataset):
    def __init__(
        self,
        data_name: str
    ):
        """initialize dataset"""
        self.dataset_name = data_name
        self.questions, self.reasonings, self.labels = get_dataset(self.dataset_name)

    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict:
        question, reasoning, label = self.questions[idx], self.reasonings[idx], self.labels[idx]

        # Convert string reasoning to list format
        if isinstance(reasoning, str):
            # Split by newlines and filter empty lines
            reasoning_lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
            
            # Convert each line from string format to list
            reasoning_list = []
            for line in reasoning_lines:
                try:
                    parsed = ast.literal_eval(line)
                    if isinstance(parsed, list) and len(parsed) == 3:
                        reasoning_list.append(parsed)
                except (ValueError, SyntaxError):
                    continue
                
            reasoning = reasoning_list

        input_text = get_prompt(question)
        
        # build target text
        target_text = ""
        target_text += "<reasoning>\n"
        if reasoning:
            for i, step in enumerate(reasoning):
                target_text += f"<step>{' -> '.join(step)}</step>"
                if i != len(reasoning) - 1:
                    target_text += "\n"
        target_text += "</reasoning>\n"
        target_text += f"<answer>{label}</answer>"

        return {
            'id': idx,
            'dataset': self.dataset_name,
            'question': question,
            'reasoning': target_text,
            'label': label,
            'input_text': input_text,
            'target_text': target_text
        }
    
    def get_splits(
        self,
        train_ratio: float = 0.9,
        ) -> Tuple[List[int], List[int]]:
        """get train/val splits"""
        indices = list(range(len(self)))
        random.shuffle(indices)
        
        train_size = int(len(self) * train_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        return {'train': train_indices, 'val': val_indices}


class TestDataset(Dataset):
    def __init__(
        self,
        data_name: str
    ):
        """initialize dataset"""
        self.dataset_name = data_name
        self.questions, self.labels = get_dataset(self.dataset_name)

    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict:
        question, label = self.questions[idx], self.labels[idx]
        input_text = get_prompt(question)
        
        
        return {
            'id': idx,
            'dataset': self.dataset_name,
            'question': question,
            'label': label,
            'input_text': input_text,
        }

if __name__ == "__main__":
    dataset = ReasoningDataset('logiqa')
    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')
