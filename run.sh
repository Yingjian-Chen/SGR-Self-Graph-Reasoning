echo "=== Start reasoningLLM running ==="
echo "Start time: $(date)"

echo "Python: $(which python)"
python --version
echo "Pip: $(which pip)"
pip --version

# training
echo "Running training..."
python train.py \
    --llm_model_name llama_8b\
    --project_name R_llama_8b\
    --dataset_name logiqa

echo "Finished training at: $(date)"
