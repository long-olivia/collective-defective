import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot(prompt_pair, arr):
    gpt=arr[0]
    claude=arr[1]
    width=0.3
    x = np.arange(20)
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Average Contribution per Round, {prompt_pair}, Name Condition")
    gpt_bars=plt.bar(np.arange(len(gpt)), gpt, width=width, color='powderblue', label='GPT-4o')
    claude_bars=plt.bar(np.arange(len(claude)) + width, claude, width=width, color='teal', label='Sonnet 4')
    plt.bar_label(gpt_bars, fmt='%.1f')
    plt.bar_label(claude_bars, fmt='%.1f')
    plt.xticks(x, x_labels)
    plt.xlabel("Rounds")
    plt.ylabel("Points Contributed")
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    plt.ylim(top=12)
    plt.tight_layout()
    plt.show()

def prepare(prompt_pair):
    file=open("discrim_round.json")
    data=json.load(file)
    arr = data[prompt_pair]
    labels = {
        "CC": "Collective Collective",
        "CN": "Collective Neutral",
        "CS": "Collective Self",
        "NC": "Neutral Collective",
        "NN": "Neutral Neutral",
        "NS": "Neutral Self",
        "SC": "Self Collective",
        "SN": "Self Neutral",
        "SS": "Self Self"
    }
    plot(labels[prompt_pair], arr)
    
if __name__ == "__main__":
    prepare("SS")