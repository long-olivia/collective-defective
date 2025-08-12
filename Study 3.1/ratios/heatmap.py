import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

round_prompts=["CCCC", "NNNN", "SSSS"]
round_models=["No Name: GPT-4o (1st)", "No Name: Sonnet 4 (2nd)", "No Name: Llama 4 (3rd)", "No Name: Qwen3 (4th)", 
              "Name: GPT-4o (1st)", "Name: Sonnet 4 (2nd)", "Name: Llama 4 (3rd)", "Name: Qwen3 (4th)"]

spear_scores=np.array([[0.6752618677, 0.2518893671, 0.8569205893, 0.864684109, 0.4628861842, 0.2644204315, 0.3467926603, 0.7216734485],
                       [0.7041098068, 0.7269316092, 0.6573465827, 0.6676703521, 0.7763556418, 0.1911439497, 0.7932208454, 0.7835910151],
                       [0.9375472932, 0.9394509878, 0.953724605, 0.8362815797, 0.9770591138, 0.8039142459, 0.9578947368, 0.9857035365]])

spear_scores_trunc=np.trunc(spear_scores * 100)/100

fig, ax = plt.subplots()
im = ax.imshow(spear_scores)

ax.set_xticks(range(len(round_models)), labels=round_models, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(round_prompts)), labels=round_prompts)

for i in range(len(round_prompts)):
    for j in range(len(round_models)):
        text=ax.text(j,i,spear_scores_trunc[i][j], ha="center", va="center", color="w")

ax.set_title("Spearman Correlations of Avg. Sentiment Score &\nAvg. Contributions per Model and Prompt Pair, Study 3.1", ha="center")
# plt.subplots_adjust(left=0.3)
fig.tight_layout()
plt.savefig("study3.png", dpi=600)
