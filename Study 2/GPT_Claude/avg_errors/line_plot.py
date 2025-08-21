import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("gc_rounds.json") as file:
    no_name=json.load(file)

with open("self_gc_rounds.json") as file:
    name=json.load(file)

with open("basic_rounds_SE.json") as file:
    no_ci=json.load(file)

with open("self_rounds_SE.json") as file:
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
            'Model': 'GPT-4o',
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Model': 'Sonnet 4',
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
            'Model': 'GPT-4o',
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Model': 'Sonnet 4',
            'Condition': 'No-Name'
        })

df_all = pd.DataFrame(all_data)
markers = {'Name': 'D', 'No-Name': 'o'}
colors = {'GPT-4o': 'teal', 'Sonnet 4': 'lightcoral'}

fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axes = axes.flatten()
sns.set_theme(style='white')
pairings_to_plot = ["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
df_filtered=df_all[df_all['Prompt_Pairing'].isin(pairings_to_plot)]
titles = {
    "CC": "Collective - GPT-4o, Collective - Sonnet 4",
    "CN": "Collective - GPT-4o, Neutral - Sonnet 4",
    "CS": "Collective - GPT-4o, Selfish - Sonnet 4",
    "NC": "Neutral - GPT-4o, Collective - Sonnet 4",
    "NN": "Neutral - GPT-4o, Neutral - Sonnet 4",
    "NS": "Neutral - GPT-4o, Selfish - Sonnet 4",
    "SC": "Selfish - GPT-4o, Collective - Sonnet 4",
    "SN": "Selfish - GPT-4o, Neutral - Sonnet 4",
    "SS": "Selfish - GPT-4o, Selfish - Sonnet 4",
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
        prompt = subdf['Prompt_Pairing'].iloc[0][0] if model == 'GPT-4o' else subdf['Prompt_Pairing'].iloc[0][1]
        ax.plot(subdf['Round'], subdf['Contribution'], 
                label=f"{model} - {condition}",
                color=colors[model],
                marker=markers[condition])
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


plt.tight_layout()
# plt.subplots_adjust(top=0.94)
# plt.suptitle('Study 2: Per Round Model Contributions by Prompt Pairing (GPT-4o - Sonnet 4)')
plt.savefig('study2_gc', dpi=600)
