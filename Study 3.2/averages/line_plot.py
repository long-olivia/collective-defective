import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_files = {
    'GPT-4o': ('gpt_rounds.json', 'self_gpt_rounds.json'),
    'Sonnet 4': ('ant_rounds.json', 'self_ant_rounds.json'),
    'Llama 4 Maverick': ('meta_rounds.json', 'self_meta_rounds.json'),
    'Qwen3 235B A22B Instruct 2507': ('qwen_rounds.json', 'self_qwen_rounds.json')
}

palette = {
    'GPT-4o': ['teal', 'salmon', 'darkgoldenrod', 'indigo'],
    'Sonnet 4': ['teal', 'salmon', 'darkgoldenrod', 'indigo'],
    'Llama 4 Maverick': ['teal', 'salmon', 'darkgoldenrod', 'indigo'],
    'Qwen3 235B A22B Instruct 2507': ['teal', 'salmon', 'darkgoldenrod', 'indigo']
}

markers = {'Name': 'D', 'No-Name': 'o'}
sns.set_theme(style='white')
all_data = []

# Load and process data from all files
for model_name, (file_no_name, file_name) in model_files.items():
    # Process 'Name' condition file
    try:
        with open(file_name) as file:
            name_data = json.load(file)
        
        for pairing, contributions in name_data.items():
            for i, contrib_list in enumerate(contributions):
                for round_num, contribution in enumerate(contrib_list):
                    all_data.append({
                        'Prompt_Pairing': pairing,
                        'Round': round_num + 1,
                        'Contribution': contribution,
                        'Model_Instance': f'{model_name}_{i+1}',
                        'Model': model_name,
                        'Condition': 'Name'
                    })
    except FileNotFoundError:
        print(f"File not found: {file_name}. Skipping...")
        
    try:
        with open(file_no_name) as file:
            no_name_data = json.load(file)
            
        for pairing, contributions in no_name_data.items():
            for i, contrib_list in enumerate(contributions):
                for round_num, contribution in enumerate(contrib_list):
                    all_data.append({
                        'Prompt_Pairing': pairing,
                        'Round': round_num + 1,
                        'Contribution': contribution,
                        'Model_Instance': f'{model_name}_{i+1}',
                        'Model': model_name,
                        'Condition': 'No-Name'
                    })
    except FileNotFoundError:
        print(f"File not found: {file_no_name}. Skipping...")

if not all_data:
    print("No data was loaded. Please check your file paths.")
else:
    df = pd.DataFrame(all_data)
    
    unique_models = df['Model'].unique()
    unique_pairings = ['CCCC', 'NNNN', 'SSSS']
    
    fig, axes = plt.subplots(len(unique_models), len(unique_pairings), figsize=(20, 20), sharex=True, sharey=True)
    
    for row_idx, model in enumerate(unique_models):
        for col_idx, pairing in enumerate(unique_pairings):
            ax = axes[row_idx, col_idx]
            df_plot = df[(df['Model'] == model) & (df['Prompt_Pairing'] == pairing)]
            
            sns.lineplot(
                data=df_plot,
                x='Round',
                y='Contribution',
                hue='Model_Instance',
                palette=palette[model],
                style='Condition',
                markers=markers,
                ax=ax,
                legend=False
            )

            if row_idx == 0:
                ax.set_title(f'Prompt Pairing: {pairing}')
            if col_idx == 0:
                ax.set_ylabel(f'{model}\nAverage Point Contribution (0-10)', ha='center', rotation=90, labelpad=50)
            else:
                ax.set_ylabel('') # Clear y-label for other columns
            
            if row_idx == len(unique_models) - 1:
                ax.set_xlabel('Round')
                ax.set_xticks(range(1,21))
                ax.set_yticks(range(0, 11))
                ax.tick_params(axis='both', which='both', left=True, bottom=True, labelleft=True, labelbottom=True)
            else:
                ax.set_xlabel('')
                ax.set_yticks(range(0, 11))
                ax.set_xticks(range(1,21))
                ax.tick_params(axis='both', left=True, bottom=True, labelleft=True, labelbottom=False)
            
            ax.set_ylim(0, 10)
            # ax.tick_params(axis='both', which='both', left=True, bottom=True, labelleft=True, labelbottom=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.suptitle('Study 3.2: Per Round Contributions by Prompt Pairing and Model')
    plt.savefig('study32.png', dpi=600)
    plt.close()