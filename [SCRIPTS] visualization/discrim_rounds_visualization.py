import numpy as np
import matplotlib.pyplot as plt
import json

def plot(prompt_pair, arr, llama_err, qwen_err):
    llama=arr[0]
    qwen=arr[1]
    width=0.35
    x = np.arange(20)
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Average Contribution per Round, {prompt_pair}, Name Condition")
    llama_bars=plt.bar(np.arange(len(llama)), llama, width=width, yerr=llama_err, capsize=5, color='lightcoral', label='Llama 4 Maverick')
    qwen_bars=plt.bar(np.arange(len(qwen)) + width, qwen, width=width, yerr= qwen_err, capsize=5, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
    plt.bar_label(llama_bars, fmt='%.1f', padding=5)
    plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
    plt.xticks(x+width/2, x_labels)
    plt.xlabel("Rounds")
    plt.ylabel("Points Contributed")
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    plt.ylim(top=12)
    plt.tight_layout()
    plt.show()

def prepare(prompt_pair):
    file=open("self_llama_qwen_rounds.json")
    file2=open("self_llama4_qwen_rounds_SE.json")
    data=json.load(file)
    errors=json.load(file2)
    arr=data[prompt_pair]
    err=errors[prompt_pair]
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
    plot(labels[prompt_pair], arr, err[0], err[1])
    
if __name__ == "__main__":
    prepare("SS")