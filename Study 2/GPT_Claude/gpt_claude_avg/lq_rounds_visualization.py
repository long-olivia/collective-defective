import numpy as np
import matplotlib.pyplot as plt
import json

basic={}
basic_err={}
self={}
self_err={}

def plot_basic(prompt_pair, arr, llama_err, qwen_err):
    gpt=arr[0]
    qwen=arr[1]
    width=0.35
    x = np.arange(20)
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Study 2: Average Contribution per Round, {prompt_pair}, No Name Condition")
    llama_bars=plt.bar(np.arange(len(gpt)), gpt, width=width, yerr=llama_err, capsize=5, color='lightcoral', label='Llama 4 Maverick')
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
    plt.savefig(f"basic_{prompt_pair}")
    plt.show()

def plot_discrim(prompt_pair, arr, llama_err, qwen_err):
    gpt=arr[0]
    qwen=arr[1]
    width=0.35
    x = np.arange(20)
    x_labels=[str(i) for i in range(1, 21)]
    plt.figure(figsize=(15, 7))
    plt.title(f"Study 2: Average Contribution per Round, {prompt_pair}, Name Condition")
    llama_bars=plt.bar(np.arange(len(gpt)), gpt, width=width, yerr=llama_err, capsize=5, color='lightcoral', label='Llama 4 Maverick')
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
    plt.savefig(f"self_{prompt_pair}")
    plt.show()

def prepare(prompt_pair):
    global basic, basic_err, self, self_err
    basic_arr = basic[prompt_pair]
    basic_err=basic_err[prompt_pair]
    self_arr=self[prompt_pair]
    self_err=self_err[prompt_pair]
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
    plot_basic(labels[prompt_pair], basic_arr, basic_err[0], basic_err[1])
    plot_discrim(labels[prompt_pair], self_arr, self_err[0], self_err[1])

def load(basic_rounds, basic_SE, self_rounds, self_SE):
    file=open(basic_rounds)
    file1=open(basic_SE)
    file2=open(self_rounds)
    file3=open(self_SE)
    global basic, basic_err, self, self_err
    basic=json.load(file)
    basic_err=json.load(file1)
    self=json.load(file2)
    self_err=json.load(file3)

if __name__ == "__main__":
    prompts=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    load("basic_lq_rounds.json", "basic_lq_round_SE.json", "self_lq_rounds.json", "self_lq_rounds_SE.json")
    for prompt in prompts:
        prepare(prompt)