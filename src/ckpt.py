import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def save_lora(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise ValueError(
            "Model does not support save_pretrained. "
            "Make sure you are using a PEFT model with LoRA."
        )
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[LoRA] adapter saved to {save_dir}")


def load_lora(base_model_dir: str, lora_dir: str, device="cuda"):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    model = PeftModel.from_pretrained(base_model, lora_dir)
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    print(f"[LoRA] adapter loaded from {lora_dir}")
    return model, tokenizer
