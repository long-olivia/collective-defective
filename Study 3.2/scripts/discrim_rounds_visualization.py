import numpy as np
import matplotlib.pyplot as plt
import json

def plot(prompt_pair, arr, gpt_err, claude_err, llama_err, qwen_err):
    gpt=arr[0]
    claude=arr[1]
    llama=arr[2]
    qwen=arr[3]
    width=0.4
    x = np.arange(len(gpt)) * 2
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Average Contribution per Round, {prompt_pair}, Four Player - Name Condition")
    error_config = {'ecolor': 'black', 'elinewidth': 0.7, 'capsize': 3}
    gpt_bars=plt.bar(x-1.5*width, gpt, width=width, yerr=gpt_err, error_kw=error_config, color='powderblue', label='GPT-4o')
    claude_bars=plt.bar(x-0.5*width, claude, width=width, yerr=claude_err, error_kw=error_config, color='teal', label='Sonnet 4')
    llama_bars=plt.bar(x+0.5*width, llama, width=width, yerr=llama_err, error_kw=error_config, color='lightcoral', label='Llama 4 Maverick')
    qwen_bars=plt.bar(x+1.5*width, qwen, width=width, yerr= qwen_err, error_kw=error_config, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
    plt.bar_label(gpt_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(claude_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(llama_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(qwen_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.xticks(x, x_labels)
    plt.xlabel("Rounds")
    plt.ylabel("Points Contributed")
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    plt.ylim(top=12)
    plt.tight_layout()
    plt.show()

def prepare(prompt_pair):
    file=open("self_four_rounds.json")
    file2=open("four_self_rounds_SE.json")
    data=json.load(file)
    arr = data[prompt_pair]
    errors=json.load(file2)
    err=errors[prompt_pair]
    labels = {
        "CCCC": "Collective",
        "NNNN": "Neutral",
        "SSSS": "Selfish"
    }
    plot(labels[prompt_pair], arr, err[0], err[1], err[2], err[3])
    
if __name__ == "__main__":
    prepare("SSSS")