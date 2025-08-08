import numpy as np
import matplotlib.pyplot as plt
import json

import numpy as np
import matplotlib.pyplot as plt
import json

def plot(prompt_pair, arr):
    gpt=arr[0]
    claude=arr[1]
    llama=arr[2]
    qwen=arr[3]
    width=0.4
    x = np.arange(len(gpt)) * 2
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Collective-Defective Score, {prompt_pair}, Name Condition")
    gpt_bars=plt.bar(x-1.5*width, gpt, width=width, color='powderblue', label='GPT-4o')
    claude_bars=plt.bar(x-0.5*width, claude, width=width, color='teal', label='Sonnet 4')
    llama_bars=plt.bar(x+0.5*width, llama, width=width, color='lightcoral', label='Llama 4 Maverick')
    qwen_bars=plt.bar(x+1.5*width, qwen, width=width, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
    plt.bar_label(gpt_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(claude_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(llama_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.bar_label(qwen_bars, fmt='%.1f', padding=5, fontsize=6.5)
    plt.xticks(x, x_labels)
    plt.xlabel("Rounds")
    plt.ylabel("Collective-defective score, where 0 = most defective & 1 = most cooperative")
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    plt.ylim(top=1.2)
    plt.tight_layout()
    plt.savefig(f"{prompt_pair}_self")

def prepare(prompt_pair):
    file=open("self_four_agg.json")
    data=json.load(file)
    arr = data[prompt_pair]
    labels = {
        "CCCC": "Collective",
        "NNNN": "Neutral",
        "SSSS": "Selfish"
    }
    plot(labels[prompt_pair], arr)
    
if __name__ == "__main__":
    prepare("CCCC")