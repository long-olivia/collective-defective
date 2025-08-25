import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("ant_rounds.json") as file:
    no_name=json.load(file)

with open("self_ant_rounds.json") as file:
    name=json.load(file)

with open("ant_basic_round_SE.json") as file:
    no_ci=json.load(file)

with open("ant_self_rounds_SE.json") as file:
    name_ci=json.load(file)

all_data = []
rounds = list(range(1, 21))

for pairing, contributions in name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    model_3 = contributions[2]
    model_4 = contributions[3]
    ci_1 = name_ci[pairing][0]
    ci_2 = name_ci[pairing][1]
    ci_3 = name_ci[pairing][2]
    ci_4 = name_ci[pairing][3]
    
    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'CI': ci_1[round_num],
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_3[round_num],
            'CI': ci_3[round_num],
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_4[round_num],
            'CI': ci_4[round_num],
            'Condition': 'Name'
        })


for pairing, contributions in no_name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    model_3 = contributions[2]
    model_4 = contributions[3]
    ci_1 = no_ci[pairing][0]
    ci_2 = no_ci[pairing][1]
    ci_3 = no_ci[pairing][2]
    ci_4 = no_ci[pairing][3]

    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'CI': ci_1[round_num],
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'CI': ci_2[round_num],
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_3[round_num],
            'CI': ci_3[round_num],
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_4[round_num],
            'CI': ci_4[round_num],
            'Condition': 'No-Name'
        })

df_all = pd.DataFrame(all_data)
markers = {'Name': 'D', 'No-Name': 'o'}
colors = {'Name': 'teal', 'No-Name': 'salmon'}

fig, axes = plt.subplots(1,3, figsize=(18, 6), sharex=True, sharey=True)
axes = axes.flatten()
sns.set_theme(style='whitegrid')

titles={"CCCC": "All Collective", "NNNN": "All Neutral", "SSSS": "All Selfish"}

for i, pairing in enumerate(df_all['Prompt_Pairing'].unique()):
    if i == 0:
        legend = 'brief'
    else:
        legend= False
    ax = axes[i]
    df_pairing = df_all[df_all['Prompt_Pairing'] == pairing]
    
    sns.lineplot(data=df_pairing, x='Round', y='Contribution', hue='Condition', palette=colors, style='Condition', ax=ax, markers=markers, legend=legend)
    hatches = {'Name': '.', 'No-Name': '/'}
    for (condition,), subdf in df_pairing.groupby(['Condition']):
        ax.plot(subdf['Round'], subdf['Contribution'], 
                label=f"Sonnet 4 - {condition}",
                color=colors[condition],
                marker=markers[condition])
        ax.fill_between(subdf['Round'],
                        subdf['Contribution'] - subdf['CI'],
                        subdf['Contribution'] + subdf['CI'],
                        color=colors[condition],
                        alpha=0.15,
                        hatch=hatches[condition], linewidth=0)

    ax.set_title(f'Prompt Pairing: {titles[pairing]}')
    ax.set_xlabel('Round')
    ax.set_xticks(range(1, 21))
    ax.set_yticks(range(0,11))
    ax.tick_params(axis='both', left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_ylabel('Average Point Contribution (0-10)')
    ax.set_ylim(0, 10)


plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Study 3: Per Round Model Contributions for Sonnet 4, Four-Player Condition')
plt.savefig('study3_ant', dpi=600)