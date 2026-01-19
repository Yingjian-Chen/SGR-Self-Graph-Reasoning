from src.seed import seed_everything
from data.utils.dataset import ReasoningDataset
from torch.utils.data import DataLoader
from model.reasoningLLM_ori import ReasoningLLM
from src.config import parse_args
from model.trainer import Trainer
from src.collate import collate_fn
import os
import torch
from sklearn.metrics import accuracy_score
from data.utils.dataset import TestDataset
from src.collate import collate_fn_test
from tqdm import tqdm
import pandas as pd
import os
from peft import PeftModel

def main(args):
    # 1. set up
    seed = args.seed
    seed_everything(seed)
    
    # 2. Dataset Building
    dataset = TestDataset(args.dataset_name)
    test_dataset = [dataset[i] for i in range(len(dataset))]
    
    # 3. DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_test)
    
    # 4. Model Building
    model = ReasoningLLM(args)
    
    # 5. Load checkpoint
    checkpoint_path = os.path.join(
        args.output_dir, f'{args.project_name}_{args.seed}_{args.llm_model_name}/best_model'
    )
    model.load_lora(lora_dir=checkpoint_path)

    # 6. Test examples
    model.eval()
    all_predictions = []
    all_labels = []
    all_outputs = []  # Store all model outputs
    all_questions = []  # Store all questions
    
    total_batches = len(test_loader)
    progress_bar = tqdm(total=total_batches, desc="Testing", unit="batch")
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.inference(batch)
            all_outputs.extend(outputs['output_text'])
            all_questions.extend(batch['question'])
            
            for pred, true_label in zip(outputs['answer'], batch['label']):
                pred_answer = str(pred).strip().lower()
                true_answer = str(true_label).strip().lower()
                
                if pred_answer == true_answer:
                    correct_predictions += 1
                total_predictions += 1

                # print(f"Pred: {pred_answer}, True: {true_answer}")
                
                all_predictions.append(pred_answer)
                all_labels.append(true_answer)
                
                # current_accuracy = correct_predictions / total_predictions
                
                progress_bar.set_description(
                    f"Testing [{args.dataset_name}--Pred: {pred_answer}, True: {true_answer}]"
                )
            
            progress_bar.update(1)
    
    progress_bar.close()
            
    # Prepare results for saving
    # all_reasonings = []
    # for outputs in all_outputs:
    #     all_reasonings.extend(outputs['reasoning'])
    
    results = {
        'Question': all_questions,
        'output': all_outputs,
        'Prediction': all_predictions,
        'Ground Truth': all_labels,
        # 'Reasoning': all_reasonings,
        'Is Correct': [p == t for p, t in zip(all_predictions, all_labels)]
    }
    
    results_df = pd.DataFrame(results)

    os.makedirs('outputs', exist_ok=True)
    
    csv_dir = os.path.join(args.output_dir, f'./{args.project_name}_{args.seed}_{args.llm_model_name}')
    os.makedirs(csv_dir, exist_ok=True)
    output_path = os.path.join(csv_dir, f'final_test_results_{args.dataset_name}.xlsx')
    results_df.to_excel(output_path, index=False)
    
    final_acc = accuracy_score(all_labels, all_predictions)
    print(f"\n===== {args.dataset_name} =====")
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print(f"Total Examples Tested: {len(all_labels)}")
    print(f"\nResults saved to:")
    print(f"- Excel: {output_path}")
    
    print("\nResults Summary:")
    print(f"Correct Predictions: {sum(results['Is Correct'])}")
    print(f"Incorrect Predictions: {len(results['Is Correct']) - sum(results['Is Correct'])}")
    print(f"Accuracy: {sum(results['Is Correct']) / len(results['Is Correct']):.4f}")
            

if __name__ == "__main__":
    args = parse_args()
    print("===== Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=========================")
    
    main(args)
