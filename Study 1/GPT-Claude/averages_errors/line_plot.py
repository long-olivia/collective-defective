import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("basic_round.json") as file:
    no_name=json.load(file)

with open("discrim_round.json") as file:
    name=json.load(file)


all_data = []
rounds = list(range(1, 21))

for pairing, contributions in name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    
    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'Model': 'GPT-4o',
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'Model': 'Sonnet 4',
            'Condition': 'Name'
        })


for pairing, contributions in no_name.items():
    model_1 = contributions[0]
    model_2 = contributions[1]
    
    for round_num in range(len(model_1)):
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_1[round_num],
            'Model': 'GPT-4o',
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_2[round_num],
            'Model': 'Sonnet 4',
            'Condition': 'No-Name'
        })

df_all = pd.DataFrame(all_data)
markers = {'Name': 'D', 'No-Name': 'o'}
colors = {'GPT-4o': 'teal', 'Sonnet 4': 'lightcoral'}

fig, axes = plt.subplots(3, 3, figsize=(20, 20), sharex=True, sharey=True)
axes = axes.flatten()
sns.set_theme(style='white')

for i, pairing in enumerate(df_all['Prompt_Pairing'].unique()):
    if i == 0:
        legend = 'brief'
    else:
        legend= False
    ax = axes[i]
    df_pairing = df_all[df_all['Prompt_Pairing'] == pairing]
    
    sns.lineplot(data=df_pairing, x='Round', y='Contribution', hue='Model', palette=colors, style='Condition', ax=ax, markers=markers, legend=legend)
    
    ax.set_title(f'Prompt Pairing: {pairing}')
    ax.set_xlabel('Round')
    ax.set_xticks(range(1, 21))
    ax.set_ylabel('Average Point Contribution (0-10)')
    ax.set_yticks(range(0,11))
    ax.tick_params(left=True, labelleft=True)
    ax.set_ylim(0, 10)


plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.suptitle('Study 1: Per Round Model Contributions by Prompt Pairing (GPT-4o - Sonnet 4)')
plt.savefig('study1_gpt_claude', dpi=600)
