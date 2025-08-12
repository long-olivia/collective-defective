import json
import numpy as np
import matplotlib.pyplot as plt

# Load files
with open("basic_round.json") as f:
    basic_round = json.load(f)
with open("basic_sentiment.json") as f:
    basic_sentiment = json.load(f)
with open("discrim_round.json") as f:
    discrim_round = json.load(f)
with open("discrim_sentiment.json") as f:
    discrim_sentiment = json.load(f)

round_idx = 0  # pick round (0 = first)
pairings = list(basic_round.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(pairings)))  # unique colors for pairings
markers = ['o', '*']  # Model 1 = circle, Model 2 = triangle

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

def plot_condition(ax, round_data, sent_data, title):
    for idx, key in enumerate(pairings):
        for model_idx in [0, 1]:
            contrib = round_data[key][model_idx][round_idx]
            senti = sent_data[key][model_idx][round_idx]
            ax.scatter(contrib, senti,
                       color=colors[idx],
                       marker=markers[model_idx],
                       s=300,
                       alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Round Contributions")
    ax.set_ylabel("Collective-Defective Scores: 0 = defective & 1 = cooperative")

# Plot Basic and Discrim
plot_condition(axes[0], basic_round, basic_sentiment, "No Name")
plot_condition(axes[1], discrim_round, discrim_sentiment, "Name")

# Legends
pairing_handles = [plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='', label=pairings[i], markersize=8) 
                   for i in range(len(pairings))]
model_handles = [
    plt.Line2D([0], [0], color='black', marker='o', linestyle='', label='GPT-4o', markersize=8),
    plt.Line2D([0], [0], color='black', marker='*', linestyle='', label='Sonnet 4', markersize=8)
]

legend1 = axes[1].legend(handles=pairing_handles, title="Prompt Pairing", bbox_to_anchor=(1.05, 1), loc='upper left')
legend2 = axes[1].legend(handles=model_handles, title="Model", bbox_to_anchor=(1.05, 0.4), loc='upper left')
axes[1].add_artist(legend1)

plt.suptitle(f"Round {round_idx + 1} — Average Contributions vs Collective-Defective Score", fontsize=14)
plt.tight_layout()
plt.show()
