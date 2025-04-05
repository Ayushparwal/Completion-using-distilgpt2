from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Input prompt
prompt = "Once upon a time in a forest,"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")


outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Story:\n")
print(generated_text)