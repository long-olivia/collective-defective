import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("four_rounds.json") as file:
    no_name=json.load(file)

with open("self_four_rounds.json") as file:
    name=json.load(file)

with open("four_basic_round_SE.json") as file:
    no_ci=json.load(file)

with open("four_self_rounds_SE.json") as file:
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
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_3[round_num],
            'CI': ci_3[round_num],
            'Model': 'Llama 4 Maverick',
            'Condition': 'Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_4[round_num],
            'CI': ci_4[round_num],
            'Model': 'Qwen3 235B A22B Instruct 2507',
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
            'Model': 'GPT-4o',
            'CI': ci_1[round_num],
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
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_3[round_num],
            'CI': ci_3[round_num],
            'Model': 'Llama 4 Maverick',
            'Condition': 'No-Name'
        })
        all_data.append({
            'Prompt_Pairing': pairing,
            'Round': round_num + 1,
            'Contribution': model_4[round_num],
            'CI': ci_4[round_num],
            'Model': 'Qwen3 235B A22B Instruct 2507',
            'Condition': 'No-Name'
        })

df_all = pd.DataFrame(all_data)
markers = {'Name': 'D', 'No-Name': 'o'}
colors = {'GPT-4o': 'teal', 'Sonnet 4': 'salmon','Llama 4 Maverick': 'darkgoldenrod', 'Qwen3 235B A22B Instruct 2507': 'indigo'}

fig, axes = plt.subplots(1,3, figsize=(18, 6), sharex=True, sharey=True)
axes = axes.flatten()
sns.set_theme(style='whitegrid')

for i, pairing in enumerate(df_all['Prompt_Pairing'].unique()):
    if i == 0:
        legend = 'brief'
    else:
        legend= False
    ax = axes[i]
    df_pairing = df_all[df_all['Prompt_Pairing'] == pairing]
    
    sns.lineplot(data=df_pairing, x='Round', y='Contribution', hue='Model', palette=colors, style='Condition', ax=ax, markers=markers, legend=legend)
    hatches = {'Name': '.', 'No-Name': '/'}
    for (model, condition), subdf in df_pairing.groupby(['Model', 'Condition']):
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

    ax.set_title(f'Prompt Pairing: {pairing}')
    ax.set_xlabel('Round')
    ax.set_xticks(range(1, 21))
    ax.set_yticks(range(0,11))
    ax.tick_params(axis='both', left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_ylabel('Average Point Contribution (0-10)')
    ax.set_ylim(0, 10)


plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Study 3.1: Per Round Model Contributions by Prompt Pairing (GPT - Sonnet - Llama - Qwen)')
plt.show()
# plt.savefig('study31_gslq', dpi=600)
