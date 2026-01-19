import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning Fine-tuning")

    # Basic Parameters
    parser.add_argument("--project_name", type=str, default="R")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--dataset_name", type=str, default="logiqa")
    parser.add_argument("--mode", type=lambda x: x.lower() == "true", default=True, help="Enable LoRA training")

    # Dataset and Training Parameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.01)
    parser.add_argument("--grad_steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0.05)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    # LLM
    parser.add_argument("--llm_model_name", type=str, default="llama_8b")

    # LoRA Parameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=['q_proj', 'v_proj'])
    # parser.add_argument("--load_in_8bit", action="store_true")
    # parser.add_argument("--device", type=str, default="auto")

    # Inference Parameters
    parser.add_argument("--max_txt_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    
    # Text Generation Parameters
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    args = parser.parse_args()
    return args