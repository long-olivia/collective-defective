import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("llama_qwen_rounds.json") as file:
    no_name=json.load(file)

with open("self_llama_qwen_rounds.json") as file:
    name=json.load(file)

with open("lq_basic_rounds_SE.json") as file:
    no_ci=json.load(file)

with open("lq_self_rounds_SE.json") as file:
    name_ci=json.load(file)


all_data = []
rounds = list(range(1, 21))

for pairing, contributions in name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    ci_1 = name_ci[pairing][0]
    ci_2 = name_ci[pairing][1]
    
    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'CI': ci_1[round_num],
            'Model': 'Llama 4 Maverick',
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Model': 'Qwen3 235B A22B Instruct 2507',
            'Condition': 'Name'
        })


for pairing, contributions in no_name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    ci_1 = no_ci[pairing][0]
    ci_2 = no_ci[pairing][1]

    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'CI': ci_1[round_num],
            'Model': 'Llama 4 Maverick',
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Model': 'Qwen3 235B A22B Instruct 2507',
            'Condition': 'No-Name'
        })

df_all = pd.DataFrame(all_data)
markers = {'Name': 'D', 'No-Name': 'o'}
colors = {'Llama 4 Maverick': 'mediumslateblue', 'Qwen3 235B A22B Instruct 2507': 'indigo'}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
axes = axes.flatten()
sns.set_theme(style='white')
pairings_to_plot = ["CC", "CS", "NC", "NS", "SC", "SS"]
df_filtered=df_all[df_all['Prompt_Pairing'].isin(pairings_to_plot)]
titles = {
    "CC": "Collective - Llama 4, Collective - Qwen3",
    "CN": "Collective - Llama 4, Neutral - Qwen3",
    "CS": "Collective - Llama 4, Selfish - Qwen3",
    "NC": "Neutral - Llama 4, Collective - Qwen3",
    "NN": "Neutral - Llama 4, Neutral - Qwen3",
    "NS": "Neutral - Llama 4, Selfish - Qwen3",
    "SC": "Selfish - Llama 4, Collective - Qwen3",
    "SN": "Selfish - Llama 4, Neutral - Qwen3",
    "SS": "Selfish - Llama 4, Selfish - Qwen3",
}
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})
for i, pairing in enumerate(pairings_to_plot):
    if i == 0:
        legend = 'brief'
    else:
        legend= False
    ax = axes[i]
    df_pairing = df_all[df_all['Prompt_Pairing'] == pairing]
    
    sns.lineplot(data=df_pairing, x='Round', y='Contribution', hue='Model', palette=colors, style='Condition', ax=ax, markers=markers, legend=legend)
    hatches = {'Name': '.', 'No-Name': '/'}
    for (model, condition), subdf in df_pairing.groupby(['Model', 'Condition']):
        prompt_colors = {'C': 'whitesmoke', 'N': 'slategray', 'S': 'navy'}
        prompt = subdf['Prompt_Pairing'].iloc[0][0] if model == 'Llama 4 Maverick' else subdf['Prompt_Pairing'].iloc[0][1]
        ax.plot(subdf['Round'], subdf['Contribution'], 
                label=f"{model} - {condition}",
                color=colors[model],
                marker=markers[condition],
                markerfacecolor=prompt_colors[prompt])
        ax.fill_between(subdf['Round'],
                        subdf['Contribution'] - subdf['CI'],
                        subdf['Contribution'] + subdf['CI'],
                        color=colors[model],
                        alpha=0.15,
                        hatch=hatches[condition], linewidth=0)
    
    ax.set_title(f'Prompt Pairing: {titles[pairing]}')
    ax.set_xlabel('Round')
    ax.set_xticks(range(1, 21))
    ax.set_ylabel('Average Point Contribution (0-10)')
    ax.set_yticks(range(0,11))
    ax.tick_params(left=True, labelleft=True)
    ax.set_ylim(0, 10)
    ax = axes[1]
    prompt_handles = [ax.scatter([], [], color=color, label=f'Prompt: {p}', s=80)
                  for p, color in prompt_colors.items()]
    ax.legend(handles=prompt_handles, loc='lower left', title='Prompt Colors')


plt.tight_layout()
# plt.subplots_adjust(top=0.94)
# plt.suptitle('Study 1: Per Round Model Contributions by Prompt Pairing (Llama 4 Maverick - Qwen3 235B A22B Instruct 2507)')
plt.savefig('study1_lq_short', dpi=600)
