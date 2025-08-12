import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

round_models=["No Name: GPT-4o (1st)", "No Name: Sonnet 4 (2nd)", "No Name: Llama 4 (1st)", "No Name: Qwen3 (2nd)", 
              "Name: GPT-4o (1st)", "Name: Sonnet 4 (2nd)", "Name: Llama 4 (1st)", "Name: Qwen3 (2nd)"]

round_prompts=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]

spear_scores=np.array([[np.nan, 0.7589285714, 0.3318390597, -0.0598677818, -0.1843061856, 0.8372423489, 0.683530204, 0.9611256893, 0.9344388486], 
                       [0.727360788, 0.1459884416, 0.8552122401, 0.4375699917, -0.3392252966, 0.8230804121, 0.3097587401, 0.8231755528, 0.9856930505], 
                       [0.761357118, 0.8930325852, 0.9717520328, 0.4332456117, 0.8815344738, 0.8728369664, 0.8392319277, 0.6303946967, 0.7374769303], 
                       [0.8998456658, 0.7739270282, 0.8539652817, 0.6794584316, 0.7725912376, 0.8254053137, 0.5931502114, 0.5931502114, 0.6027914424],  
                       [0.6537786173, 0.0834543006, 0.5852478413, 0.2350825843, 0.5331929874, 0.8880528399, 0.7446255451, 0.9256832849, 0.7681883224], 
                       [0.3606361396, 0.3839554115, 0.8433744504, -0.4658930926, -0.3805586213, 0.661898341, -0.4071383122, 0.3363431151, 0.6694246431], 
                       [0.584557812, 0.6954068237, 0.7030601182, 0.7784882085, 0.8807823113, 0.7727616936, 0.8962406015, 0.7566754954, 0.7694622583], 
                       [0.1917862012, 0.5662657024, 0.8624717408, 0.8547782706, 0.7923250564, 0.6102334308, 0.8446785103, 0.7688255192, 0.6940158565]])

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
