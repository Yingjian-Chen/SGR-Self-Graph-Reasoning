import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        args
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        # set optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # set learning rate scheduler
        num_training_steps = len(self.train_loader) * args.num_epochs
        num_warmup_steps = int(args.warmup * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
    def train(self):
        """train model"""
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # evaluate
            if self.val_loader:
                val_loss, samples, acc = self.evaluate()
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy: {acc*100:.2f}%")

                for s in samples:
                    print("Input:", s["input"])
                    print("Target:", s["target"])
                    print("Generated:", s["generated"])
                    print("="*80)
                
                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.model.save_lora(f'{self.args.output_dir}/{self.args.project_name}_{self.args.seed}_{self.args.llm_model_name}/best_model')
                else:
                    patience_counter += 1
                
                # early stopping
                if patience_counter >= self.args.patience:
                    print("Early stopping triggered")
                    break
    
    def train_epoch(self):
        """train one epoch"""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # forward + mixed precision
            with self.model.maybe_autocast():
                outputs = self.model(batch)
                loss = outputs["loss"] / self.args.grad_steps 

            # backward
            loss.backward()

            if (step + 1) % self.args.grad_steps == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # update parameters
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.args.grad_steps

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """evaluate model"""
        self.model.eval()
        total_loss = 0
        generations = []

        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                with self.model.maybe_autocast():
                    outputs = self.model(batch)
                    loss = outputs["loss"]
                    total_loss += loss.item()
                
                    input_texts = batch["input_text"]
                    target_texts = batch["target_text"]
                    labels = batch["label"]
                    if i < 20:
                        outputs = self.model.inference(batch)

                        for ans, label in zip(outputs['answer'], labels):
                            if ans.strip() == label.strip():
                                correct += 1
                            total += 1

                        for inp, tgt, gen in zip(input_texts, target_texts, outputs['output_text']):
                            generations.append({
                                "input": inp,
                                "target": tgt,
                                "generated": gen,
                            })

        acc = correct / total if total > 0 else 0.0
    
        return total_loss / len(self.val_loader), generations, acc
    
    def compute_metrics(self, predictions, labels):
        """calculate metrics"""
        # parse predictions and labels
        parsed_preds = []
        parsed_labels = []
        
        for pred, label in zip(predictions, labels):
            # convert to int
            try:
                pred_answer = int(pred)
                true_answer = int(label)
            except ValueError:
                # if conversion fails, count as wrong prediction
                pred_answer = -1
                true_answer = -1
                
            parsed_preds.append(pred_answer)
            parsed_labels.append(true_answer)
        
        # calculate accuracy using sklearn
        accuracy = accuracy_score(parsed_labels, parsed_preds)
        
        return {
            'accuracy': accuracy
        }
