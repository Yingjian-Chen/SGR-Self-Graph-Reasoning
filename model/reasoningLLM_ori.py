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
        if self.model_name == "llama_70b":
            model_kwargs = {
                "dtype": "auto",
                "load_in_8bit": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

        elif self.model_name == "qwen2.5_72b":
            model_kwargs = {
                "dtype": "auto",
                "load_in_4bit": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

        else:
            model_kwargs = {
                "dtype": torch.float16,
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

        # model.resize_token_embeddings(len(self.tokenizer))
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        model.gradient_checkpointing_enable()

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

    def _find_assistant_start(self, input_ids):
        assistant_token = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        
        for i in range(len(input_ids) - len(assistant_token) + 1):
            if torch.equal(input_ids[i:i+len(assistant_token)], torch.tensor(assistant_token).to(input_ids.device)):
                return i + len(assistant_token)
        return -1


    def forward(self, batch):
        input_texts = batch['input_text']
        target_texts = batch['target_text']

        full_texts = [
            f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{target_text}<|eot_id|>"
            for input_text, target_text in zip(input_texts, target_texts)
        ]
        
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
            
        labels = inputs.input_ids.clone()
        
        # mask input part
        for i, input_ids in enumerate(inputs.input_ids):
            assistant_start = self._find_assistant_start(input_ids)
            if assistant_start >= 0:
                labels[i, :assistant_start] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True
        )

        return {"loss": outputs.loss}

    def inference(self, data):
        input_texts = data['input_text']

        prompts = [
            f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            for text in input_texts
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            with self.maybe_autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    early_stopping=True
                )

        generated_texts = [
            self.tokenizer.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=False)
            for ids in outputs
        ]
        
        gen_texts = [self._clean_generated_text(text) for text in generated_texts]

        reasoning_list, answer_list = [], []
        for text in gen_texts:
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
            answer_list.append(answer_match.group(1).strip() if answer_match else "")
            reasoning_list.append(reasoning_match.group(1).strip() if reasoning_match else "")

        return {
            "output_text": gen_texts,
            "answer": answer_list,
            "reasoning": reasoning_list
        }
    
    def save_lora(self, save_dir: str):
        if not hasattr(self.model, "save_pretrained"):
            raise ValueError("Model does not support save_pretrained. "
                             "Make sure you are using a PEFT model with LoRA.")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"[LoRA] adapter saved to {save_dir}")


    def load_lora(self, lora_dir: str):
        # print(f"[LoRA] Loading base model: {self.model_name}")
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     model_path[self.model_name],
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True,
        #     device_map="auto" if self.device == "cuda" else None
        # )

        # print(f"[LoRA] Loading lora adapter from {lora_dir}")
        # model = PeftModel.from_pretrained(base_model, lora_dir)

        # model = model.merge_and_unload()
        # model.eval()

        # self.model = model
        # self.tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
        print(f"[LoRA] Loading tokenizer from LoRA dir")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path[self.model_name],
            trust_remote_code=True,
            padding_side="left",
            use_auth_token=HF_TOKEN
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"[LoRA] Loading base model: {self.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path[self.model_name],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        base_model.resize_token_embeddings(len(tokenizer))

        print(f"[LoRA] Loading lora adapter from {lora_dir}")
        model = PeftModel.from_pretrained(base_model, lora_dir)

        model = model.merge_and_unload()
        model.eval()

        self.model = model
        self.tokenizer = tokenizer

    def load_lora_train(self, lora_dir: str):
        print(f"[LoRA] Loading base model: {self.model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path[self.model_name],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )

        print(f"[LoRA] Loading lora adapter from {lora_dir}")
        
        model = PeftModel.from_pretrained(base_model, lora_dir, is_trainable=True)
        model.train()

        self.model = model

    def get_hf_model(self):
        return self.model
    
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



