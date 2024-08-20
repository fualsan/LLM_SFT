import os

from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
print(f'Downloading tokenizer and model: {model_id}')

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ['HF_TOKEN'] # required for llama 3.1
)

print('*** TOKENIZER IS DOWNLOADED ***')


model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    token=os.environ['HF_TOKEN'] # required for llama 3.1
)

print('*** MODEL IS DOWNLOADED ***')
