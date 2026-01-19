import torch
import torch.nn as nn
import pandas as pd
import json
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training, PeftModel
)
import contextlib
import torch.nn.functional as F

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable before running.")

model_path = {
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B"
}


class ReasoningLLM(nn.Module):
    def __init__(
        self,
        args,
        model_name: str = None,
        device: str = "auto"
    ):
        super().__init__()
        self.args = args
        self.model_name = model_name or args.llm_model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        
        # Set LoRA config
        self.lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # Load the model and tokenizer
        self.model = self.load_model()

        special_tokens = {
            "additional_special_tokens": ["<|user|>", "<|assistant|>", "<reasoning>", "</reasoning>", "<answer>", "</answer>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def load_model(self):
        print(f"Loading model: {self.model_name}")
        
        # get model path
        model_path_str = model_path[self.model_name]
        print(f"Model: {model_path_str}")
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_str,
            trust_remote_code=True,
            padding_side="left",
            use_auth_token=HF_TOKEN
        )
        
        # set pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # load base model
        model_kwargs = {
            "dtype": torch.float16,
            # "dtype": "auto",
            # "load_in_8bit": True,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            print(f"Number of GPUs: {num_devices}\n")
            max_memory = {}
            for i in range(num_devices):
                #print
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                print(f"GPU {i}: {props.name}, Total Memory: {total_mem:.2f} GiB")

                total_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                max_memory[i] = f"{max(total_memory - 1, 2)}GiB"
                
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory
        else:
            model_kwargs["device_map"] = None
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            use_auth_token=HF_TOKEN,
            **model_kwargs
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        if self.args.mode:
            print(">>> Lora")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
        else:
            print(">>> No Lora")
            for param in model.parameters():
                param.requires_grad = False

        return model
    
    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device.startswith("cuda")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def __call__(self, batch):
        return self.forward(batch)

    def forward(self, batch):
        input_texts = batch["input_text"]
        reasoning_labels = batch["target_text"]
        answer_labels = batch["label"]

        full_texts = [
            f"<|user|>\n{text}\n<|assistant|>\n<reasoning>{r}</reasoning>\n<answer>{a}</answer>"
            for text, r, a in zip(input_texts, reasoning_labels, answer_labels)
        ]

        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # masking
        labels = inputs.input_ids.clone()
        assistant_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        for i, input_ids in enumerate(inputs.input_ids):
            starts = (input_ids == assistant_id).nonzero(as_tuple=True)
            if len(starts[0]) > 0:
                labels[i, :starts[0][0]+1] = -100
            else:
                labels[i, :] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True
        )

        total_loss = outputs.loss

        return {
            "loss": total_loss,
        }
    
    def inference(self, data):
        input_texts = data["input_text"]

        prompts = [
            f"<|user|>\n{text}\n<|assistant|>\n<reasoning>"
            for text in input_texts
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        eos_token_id = [
            self.tokenizer.convert_tokens_to_ids("</reasoning>"),
            self.tokenizer.convert_tokens_to_ids("</answer>"),
            self.tokenizer.eos_token_id
        ]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # --------------------------------------------------------

        reasoning_list, answer_list, full_outputs = [], [], []

        for i, output_ids in enumerate(outputs):
            gen_ids = output_ids[inputs.input_ids.shape[1]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", gen_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            answer_match = re.search(r"<answer>(.*?)</answer>", gen_text, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ""

            reasoning_list.append(reasoning)
            answer_list.append(answer)
            full_outputs.append(gen_text)

        return {
            "reasoning": reasoning_list,
            "answer": answer_list,
            "output_text": full_outputs
        }
    
    def save_lora(self, save_dir: str):
        if not hasattr(self.model, "save_pretrained"):
            raise ValueError("Model does not support save_pretrained. "
                             "Make sure you are using a PEFT model with LoRA.")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"[LoRA] adapter saved to {save_dir}")


    def load_lora(self, lora_dir: str):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path[self.model_name],
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )

        # load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, lora_dir)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

        print(f"[LoRA] adapter loaded from {lora_dir}")

    def _find_assistant_start(self, input_ids):
        assistant_token = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        
        for i in range(len(input_ids) - len(assistant_token) + 1):
            if torch.equal(input_ids[i:i+len(assistant_token)], torch.tensor(assistant_token).to(input_ids.device)):
                return i + len(assistant_token)
        return -1
    
    def _clean_generated_text(self, text):
        special_tokens = [
            '<|start_header_id|>', '<|end_header_id|>', 
            '<|eot_id|>', '<|end_of_text|>'
        ]
        
        for token in special_tokens:
            text = text.replace(token, '')
        
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        if text.startswith((': ', '：', '\n')):
            text = text.lstrip(': ：\n')
        
        return text



