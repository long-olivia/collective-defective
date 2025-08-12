import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

round_models=["No Name: GPT-4o (1st)", "No Name: Sonnet 4 (2nd)", "No Name: Llama 4 (1st)", "No Name: Qwen3 (2nd)", 
              "Name: GPT-4o (1st)", "Name: Sonnet 4 (2nd)", "Name: Llama 4 (1st)", "Name: Qwen3 (2nd)"]

round_prompts=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]

spear_scores=np.array([[-0.1163000547, 0.5664645726, 0.2713543029, -0.0030257402, 0.3068973926, 0.1791375148, 0.7046410119, 0.4416793413, 0.3498523173],
                       [-0.1348285663, 0.1347243658, 0.8334598765, -0.089467883, -0.5505967028, 0.4260884487, 0.5823929414, -0.2359597104, 0.4180798374],
                       [0.7278612113, 0.5372331086, 0.633790677, -0.2985798486, 0.2767527464, 0.6149213277, 0.7190207666, 0.7816624437, 0.0722238595],
                       [0.7159069955, 0.6309597309, 0.2316690536, -0.7675072819, 0.6138936388, 0.4992120549, -0.2266013919, 0.4820268628, -0.2366251667],
                       [0.2970133046, -0.0290459682, 0.5798983214, 0.4587969895, 0.4175522937, 0.3741150285, 0.7389085293, 0.253502747, 0.7625774831],
                       [0.24580851, 0.2285954928, 0.1188464067, -0.8502873399, 0.002451378, -0.0856549934, -0.7289381745, -0.7967067471, -0.1115461585],
                       [0.7594478629, 0.0936642166, 0.8743418577, 0.4598583666, 0.554349825, 0.6077641895, 0.7267957647, 0.9376931011, 0.9408268428],
                       [0.9456286668, 0.02939624, 0.9401586723, 0.8364935014, 0.7625545688, 0.8467225672, 0.8291187012, 0.898754909, 0.9882766341]])

spear_scores_trunc=np.trunc(spear_scores * 100)/100

fig, ax = plt.subplots()
im = ax.imshow(spear_scores)

ax.set_xticks(range(len(round_prompts)), labels=round_prompts)
ax.set_yticks(range(len(round_models)), labels=round_models)

for i in range(len(round_models)):
    for j in range(len(round_prompts)):
        text=ax.text(j,i,spear_scores_trunc[i][j], ha="center", va="center", color="w")

fig.suptitle("Spearman Correlations of Avg. Sentiment Score &\nAvg. Contributions per Model and Prompt Pair, Study 2", ha="center")
plt.subplots_adjust(left=0.3)
fig.tight_layout()
plt.savefig("study2.png", dpi=600)
