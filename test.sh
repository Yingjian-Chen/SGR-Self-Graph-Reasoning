#!/bin/bash
echo "=== Start reasoningLLM testing ==="
echo "Start time: $(date)"

# test
echo "Testing..."
for dataset in logiqa_test aiw_easy aiw_hard lasr_ar medqa mathqa; do
    python test.py \
        --llm_model_name llama_8b \
        --project_name R_llama_8b\
        --dataset_name $dataset 
done


echo "Finished testing at: $(date)"