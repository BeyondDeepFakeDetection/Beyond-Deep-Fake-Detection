import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
from scipy.special import kl_div
import random
import matplotlib.pyplot as plt

# Login and setup
hf_token = "<Enter hf token>"
login(token=hf_token)

goal_kl_divergence = 0.6  # Enter desired kl_div between real anf general distribution
general_training_dataset_name = "<Enter original dataset name>" 
label_column_name = "label"
output_name = "<Enter desired output name>"

# Load the dataset
general_train_dataset = pd.DataFrame(load_dataset(general_training_dataset_name, split="general_train"))

label_id_counts = general_train_dataset[label_column_name].value_counts()
total_labels = general_train_dataset[label_column_name].nunique()

tries = 0
while True:
    tries += 1
    if tries > 24:
        print("error")
        exit()
        
    # Define target frequencies for label distribution
    target_frequencies = [15, 4, 1, 1]

    # target_frequencies = [freq for freq in target_frequencies for _ in range(10)]
    target_frequencies = [f / sum(target_frequencies) for f in target_frequencies]

    random.shuffle(target_frequencies)
    print(target_frequencies)

    assert abs(sum(target_frequencies) - 1.0) < 1e-6, "Target frequencies must sum to 1."
    assert len(target_frequencies) == total_labels, "Target frequency list must have one value per category."

    # Determine max feasible number of total samples (no replacement)
    scaling_factors = [
        (label_id_counts[label] / target_frequencies[i]) if target_frequencies[i] > 0 else np.inf
        for i, label in enumerate(label_id_counts.index)
    ]

    max_total_samples = int(min(scaling_factors))

    # Compute target number of samples per book_id
    target_counts = [int(freq * max_total_samples) for freq in target_frequencies]

    # Sample based on book_id
    distorted_samples = []
    for i, (label, n_samples) in enumerate(zip(label_id_counts.index, target_counts)):
        data = general_train_dataset[general_train_dataset["label"] == label]

        available = len(data)
        if n_samples > available:
            # print(f"⚠️ID {label} only has {available} samples, requested {n_samples}. Skipping this book.")
            continue

        sampled = data.sample(n=n_samples, replace=False, random_state=42)
        distorted_samples.append(sampled)

    distorted_dataset = pd.concat(distorted_samples).reset_index(drop=True)

    # Check distorted dataset
    print(f"\nNew data len / Original data len: {round(distorted_dataset.shape[0] / general_train_dataset.shape[0] * 100, 2)}%")

    # Visualizing book_id distribution
    label_id_dist = general_train_dataset[label_column_name].value_counts()
    distorted_label_id_dist = distorted_dataset[label_column_name].value_counts()

    original_distribution = general_train_dataset[label_column_name].value_counts(normalize=True).sort_index()
    distorted_distribution = distorted_dataset[label_column_name].value_counts(normalize=True).sort_index()

    # Ensure both distributions have the same labels (use union of the unique labels)
    labels = sorted(set(original_distribution.index).union(set(distorted_distribution.index)))

    # Normalize both distributions to ensure they sum to 1 (probabilities)
    original_probs = np.array([original_distribution.get(label, 0) for label in labels])
    distorted_probs = np.array([distorted_distribution.get(label, 0) for label in labels])

    # Compute the KL divergence (using scipy's kl_div which computes P * log(P/Q) for each element)
    kl_divergence = np.sum(kl_div(original_probs, distorted_probs))
    print(f"KL Divergence: {kl_divergence}")

    if goal_kl_divergence * 1.1 > kl_divergence > goal_kl_divergence:
        print(f"KL Divergence: {kl_divergence}")
        ratio = distorted_dataset.shape[0] / general_train_dataset.shape[0]
        break


# Plot original book_id distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
label_id_dist.plot(kind='bar', alpha=0.7, edgecolor='black')
plt.title("Original Label ID Distribution")
plt.xlabel("Book ID")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot distorted book_id distribution
plt.subplot(1, 2, 2)
distorted_label_id_dist.plot(kind='bar', alpha=0.7, edgecolor='black', color='orange')
plt.title("Distorted Label ID Distribution")
plt.xlabel("Book ID")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

distorted_dataset.to_csv(output_name)

