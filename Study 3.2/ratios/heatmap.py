import matplotlib.pyplot as plt

round_prompts = ["CCCC", "NNNN", "SSSS"]
round_models = ["No_N: GPT-4o (1st)", "No_N: Sonnet 4 (2nd)", "No_N: Llama 4 (3rd)", "No_N: Qwen3 (4th)", 
                "Name: GPT-4o (1st)", "Name: Sonnet 4 (2nd)", "Name: Llama 4 (3rd)", "Name: Qwen3 (4th)"]

spear_scores = [
    ["0.39-0.70", "0.33-0.78", "0.37-0.83", "0.48-0.72", "0.27-0.48", "0.41-0.88", "0.34-0.72", "0.44-0.59"],
    ["0.60-0.75", "0.61-0.87", "0.42-0.67", "0.50-0.76", "0.12-0.71", "0.63-0.86", "0.22-0.65", "0.58-0.64"],
    ["-0.12-0.46", "0.89-0.96", "0.19-0.52", "0.63-0.88", "0.74-0.82", "0.90-0.96", "0.87-0.91", "0.30-0.85"]
]

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

table = ax.table(cellText=spear_scores,
                 rowLabels=round_prompts,
                 colLabels=round_models,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2) 

plt.title("Range of Spearman Correlations of Avg. Sentiment Score & Avg. Contributions per Model Instantiation and Prompt Pair, Study 3.2")
plt.tight_layout()
plt.savefig("study32_spear", dpi=600)