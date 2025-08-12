import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

round_models=["No Name: GPT-4o (1st)", "No Name: Sonnet 4 (2nd)", "No Name: Llama 4 (1st)", "No Name: Qwen3 (2nd)", 
              "Name: GPT-4o (1st)", "Name: Sonnet 4 (2nd)", "Name: Llama 4 (1st)", "Name: Qwen3 (2nd)"]

round_prompts=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]

spear_scores=np.array([[],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       []])

spear_scores_trunc=np.trunc(spear_scores * 100)/100

fig, ax = plt.subplots()
im = ax.imshow(spear_scores)

ax.set_xticks(range(len(round_prompts)), labels=round_prompts)
ax.set_yticks(range(len(round_models)), labels=round_models)

for i in range(len(round_models)):
    for j in range(len(round_prompts)):
        text=ax.text(j,i,spear_scores_trunc[i][j], ha="center", va="center", color="w")

fig.suptitle("Spearman Correlations of Avg. Sentiment Score &\nAvg. Contributions per Model and Prompt Pair, Study 1", ha="center")
fig.tight_layout()
plt.savefig("study1.png", dpi=600)
