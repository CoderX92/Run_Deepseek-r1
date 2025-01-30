from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)

# Prepare the input
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

# Generate a response (on CPU)
outputs = model.generate(inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
