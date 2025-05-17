from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
from huggingface_hub import login
from safetensors.torch import load_file
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset
import gc
import time


hf_token = "<Enter hf-token>"
login(token=hf_token)

model_name = "<Enter model name>"

save_name = "<Enter path to save results>"
text_column = "text"

dataset = "<Enter dataset name>"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Ensure padding token exists (GPT-2 lacks a native one)
tokenizer.pad_token = tokenizer.eos_token


# Function to compute log probabilities for a sequence
def compute_log_probabilities_for_sequence(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # exclude last token's prediction
        target_ids = input_ids[:, 1:]       # shift input to match next-token prediction

        log_probs = F.log_softmax(logits, dim=-1)
        seq_token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # shape: [batch, seq_len-1]

    word_probabilities = []
    for i, token_id in enumerate(target_ids[0]):
        word = tokenizer.decode([token_id])
        log_prob = seq_token_logprobs[0, i].item()
        word_probabilities.append((word, log_prob))

    return word_probabilities


test_df = pd.DataFrame(load_dataset(dataset, split="general_test"))
# Compute log probabilities for each text
results = []
count = 0
exists_bool = False
for count, text in enumerate(tqdm(test_df[text_column], desc="Processing Texts")):
    if count > -1:
        word_probs = compute_log_probabilities_for_sequence(model, tokenizer, text)
        total_log_prob = sum(prob for _, prob in word_probs) 
        avg_log_prob = total_log_prob / len(word_probs) if word_probs else float("-inf")  #

        # Append the word-probability pairs along with total and average log probabilities
        results.append({
            "text_id": count,
            "total_log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "word_probabilities": str(word_probs),  # Store the word-probability pairs
        })

results.to_csv(f"{save_name}.csv")
