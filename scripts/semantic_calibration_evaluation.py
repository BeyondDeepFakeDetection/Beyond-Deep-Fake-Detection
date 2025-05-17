"""
Label Distribution Analysis and Rejection Sampling Visualization for CIFAR-100

This script loads datasets, computes rejection sampling, and visualizes the label distributions
for general, real, and filtered sets. It is designed for reproducible, publication-quality analysis.
"""

import ast
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
import seaborn as sns
from matplotlib import pyplot as plt

hf_token = "<Enter hf token>"
login(token=hf_token)

label_column_name = "label"
text_column_name = "florence_caption"
split_name = "severe"  # baseline/mild/moderate/severe
dataset_name = "BeyondDeepFakeDetection/CIFAR-100"
save_path = "<Enter Label Distribution Comparison output path>"
real_probs_csv = "<Enter path to real-split computed probabilities>"
general_probs_csv = "<Enter path to general computed probabilities>"


def load_datasets(hf_token):
    login(token=hf_token)
    dataset = load_dataset(dataset_name, split="train")
    return (
        pd.read_csv(general_probs_csv),
        pd.read_csv(real_probs_csv),
        pd.DataFrame(dataset)["general_test"],
        pd.DataFrame(dataset)[f"real_{split_name}"],
        pd.DataFrame(dataset)["general_train"],
    )


def prepare_dataframe(general_model_probs, real_model_probs, general_test_dataset):
    final_df = pd.merge(general_model_probs, real_model_probs, on="text_id", how="inner",
                        suffixes=("_general", "_real"))

    general_columns = [col for col in final_df.columns if col.endswith("_general")] + ["text_id"]
    real_columns = [col for col in final_df.columns if col.endswith("_real")] + ["text_id"]

    general_model_probs = final_df[general_columns].copy()
    real_model_probs = final_df[real_columns].copy()[:len(general_model_probs)]
    general_test_dataset = general_test_dataset[:len(general_model_probs)]

    df = pd.concat([
        real_model_probs.reset_index(drop=True),
        general_model_probs.reset_index(drop=True),
        general_test_dataset[[text_column_name, label_column_name]].reset_index(drop=True)
    ], axis=1)

    df["word_probabilities_real"] = df["word_probabilities_real"].apply(ast.literal_eval)
    df["word_probabilities_general"] = df["word_probabilities_general"].apply(ast.literal_eval)
    return df.drop(index=0).reset_index(drop=True)


def compute_clipped_diffs(df_arg, top_p=0.75):
    df_copy = df_arg.copy()

    def process_row(m1, m2):
        diffs = np.array([float(m_one[1]) - float(m_two[1]) for m_one, m_two in zip(m1, m2)])
        sorted_indices = np.argsort(-np.abs(diffs))
        sorted_abs_diffs = np.abs(diffs[sorted_indices])
        total_mass = sorted_abs_diffs.sum()

        if total_mass == 0:
            return 0.0

        normalized = sorted_abs_diffs / total_mass
        cumulative = np.cumsum(normalized)
        keep_mask = cumulative <= top_p
        keep_mask[0] = True
        return diffs[sorted_indices][keep_mask].sum()

    df_copy["clipped_diff"] = [process_row(m1, m2) for m1, m2 in
                               zip(df_copy["word_probabilities_real"], df_copy["word_probabilities_general"])]
    return df_copy


def normalized_freq(df_subset):
    return df_subset[label_column_name].value_counts(normalize=True).sort_index()


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0.0))


def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def compute_metrics(df_arg, real_train_dataset, general_test_dataset, log_m):
    rng = np.random.default_rng()
    df_copy = df_arg.copy()
    df_copy["log_accept_prob"] = df_copy["clipped_diff"] - log_m

    df_copy["accept_prob"] = np.exp(df_copy["log_accept_prob"]).clip(upper=1.0)
    df_copy["keep"] = rng.random(len(df_copy)) < df_copy["accept_prob"]
    filtered_df = df_copy[df_copy["keep"]]

    freqs_real = normalized_freq(real_train_dataset)
    freqs_filtered = normalized_freq(filtered_df)
    all_book_ids = sorted(set(freqs_real.index).union(freqs_filtered.index))

    aligned_real = freqs_real.reindex(all_book_ids, fill_value=0)
    aligned_filtered = freqs_filtered.reindex(all_book_ids, fill_value=0)

    metrics = {
        'kl_div': kl_divergence(aligned_filtered.values, aligned_real.values),
        'tvd': total_variation_distance(aligned_real.values, aligned_filtered.values),
        'pass_rate': filtered_df.shape[0] / df_copy.shape[0] * 100
    }

    return filtered_df, metrics


def compute_metrics_with_stats(df_arg, real_train_dataset, general_test_dataset, log_m, n_runs=1000):
    kl_divs = []
    tvds = []
    pass_rates = []
    freqs_filtered_list = []
    df_copy = df_arg.copy()
    for _ in range(n_runs):
        filtered_df, metrics = compute_metrics(df_copy, real_train_dataset, general_test_dataset, log_m)
        kl_divs.append(metrics['kl_div'])
        tvds.append(metrics['tvd'])
        pass_rates.append(metrics['pass_rate'])
        # Store normalized frequencies for this run
        freqs_filtered = normalized_freq(filtered_df)
        freqs_filtered_list.append(freqs_filtered)

    # Compute average freqs_filtered over all runs
    all_labels = sorted(set().union(*(f.index for f in freqs_filtered_list)))
    aligned_freqs = [f.reindex(all_labels, fill_value=0) for f in freqs_filtered_list]
    avg_freqs_filtered = sum(aligned_freqs) / n_runs

    return {
               'kl_div_mean': np.mean(kl_divs),
               'kl_div_std': np.std(kl_divs),
               'tvd_mean': np.mean(tvds),
               'tvd_std': np.std(tvds),
               'pass_rate_mean': np.mean(pass_rates),
               'pass_rate_std': np.std(pass_rates)
           }, avg_freqs_filtered


def plot_normalized_frequencies_one_graph_sorted(freqs_general, freqs_real, freqs_filtered):
    plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    all_labels = sorted(set().union(
        freqs_general.index,
        freqs_real.index,
        freqs_filtered.index
    ))

    # Align all frequencies
    aligned_general = freqs_general.reindex(all_labels, fill_value=0)
    aligned_real = freqs_real.reindex(all_labels, fill_value=0)
    aligned_filtered = freqs_filtered.reindex(all_labels, fill_value=0)

    # Sort by descending general frequency (highest first)
    sorted_labels = aligned_real.sort_values(ascending=False).index.tolist()
    aligned_general = aligned_general[sorted_labels]
    aligned_real = aligned_real[sorted_labels]
    aligned_filtered = aligned_filtered[sorted_labels]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    color_general = "#f2c45f"
    color_real = "#1a80bb"
    color_mean = "#7ec6e6"
    x = np.arange(len(sorted_labels))
    width = 0.22

    # Plot three bars for each label
    ax.bar(x - width, aligned_general, width,
           label='General Training Set',
           color=color_general, alpha=0.9,
           edgecolor='black', linewidth=0.5)
    ax.bar(x, aligned_real, width,
           label='Real Training Set',
           color=color_real, alpha=0.9,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width, aligned_filtered, width,
           label='Filtered Set (Ours)',
           color=color_mean, alpha=0.9,
           edgecolor='black', linewidth=0.5)

    # Axes styling
    ax.set_xlabel('Class Labels', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Frequency', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, ha='right', fontsize=6)

    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(False)
    # Title, legend, layout
    ax.set_title(f'Label Distribution {dataset_name}: General vs. Real vs. Filtered',
                 pad=20, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True,
              fancybox=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()

    return fig


def main():
    general_model_probs, real_model_probs, general_test_dataset, real_train_dataset, general_train_dataset = load_datasets(
        hf_token)

    log_m = np.log(1 / (1 - (1 - real_train_dataset.shape[0] / general_train_dataset.shape[0])))
    df = prepare_dataframe(general_model_probs, real_model_probs, general_test_dataset)
    df = compute_clipped_diffs(df)

    metrics_clipped, avg_freqs_filtered = compute_metrics_with_stats(df, real_train_dataset, general_test_dataset, log_m)

    freqs_real_train = normalized_freq(real_train_dataset)
    freqs_general_test = normalized_freq(general_test_dataset)
    freqs_general_train = normalized_freq(general_train_dataset)

    freqs_filtered = avg_freqs_filtered

    all_labels = sorted(set(freqs_real_train.index).union(freqs_general_train.index))

    aligned_real_train = freqs_real_train.reindex(all_labels, fill_value=0)
    aligned_general_test = freqs_general_test.reindex(all_labels, fill_value=0)

    kl_div_general_test_vs_real = kl_divergence(aligned_general_test.values, aligned_real_train.values)

    print("\nResults (mean ± standard error over 1000 runs):")
    print(
        f"KL-divergence(filtered||real):  {metrics_clipped['kl_div_mean']:.4f} ± {metrics_clipped['kl_div_std']:.4f}")
    print(f"KL-divergence(general ||real):  {kl_div_general_test_vs_real:.4f}")

    improvement_ratio = metrics_clipped['kl_div_mean'] / kl_div_general_test_vs_real
    gain = 1 - improvement_ratio
    print("Deception reduction: ", f"{gain * 100:.2f}%")

    plot_normalized_frequencies_one_graph_sorted(freqs_general_test, freqs_real_train, freqs_filtered)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()




