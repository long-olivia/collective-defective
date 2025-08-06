import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

def run_basic(final, errs):
    with open(final) as file:
        final_points=json.load(file)
    with open(errs) as file:
        errors=json.load(file)
    llama_basic_final=[value[0] for value in final_points.values()]
    qwen_basic_final=[value[1] for value in final_points.values()]
    llama_SE_final=[value[0] for value in errors.values()]
    qwen_SE_final=[value[1] for value in errors.values()]

    width=0.3
    labels = ['Collective Collective', 
            'Collective Neutral', 
            'Collective Self', 
            'Neutral Collective', 
            'Neutral Neutral', 
            'Neutral Self', 
            'Self Collective', 
            'Self Neutral', 
            'Self Self']

    x = np.arange(9)
    plt.figure(figsize=(15, 7))
    plt.title("Study 1: Average Final Points Accumulated Across Prompts, No Name Condition")
    #yerr=llama_SE_final, capsize=10, 
    llama_bars=plt.bar(np.arange(len(llama_basic_final)), llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Llama 4 Maverick')
    qwen_bars=plt.bar(np.arange(len(qwen_basic_final)) + width, qwen_basic_final, width=width, yerr=qwen_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
    plt.bar_label(llama_bars, fmt='%.1f', padding=5)
    plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
    plt.xticks(x+width/2, labels, rotation=-45, ha='left')
    plt.ylabel("Average Final Points Accumulated")
    plt.xlabel("Prompt Pairings: Llama 4 & Qwen 3")
    plt.ylim(bottom=180, top=350)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    ax = plt.gca()
    y_major_step=10
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
    plt.tight_layout()
    plt.savefig("final_lq")

def run_discrim(final, errs):
    with open(final) as file:
        final_points=json.load(file)
    with open(errs) as file:
        errors=json.load(file)
    llama_basic_final=[value[0] for value in final_points.values()]
    qwen_basic_final=[value[1] for value in final_points.values()]
    llama_SE_final=[value[0] for value in errors.values()]
    qwen_SE_final=[value[1] for value in errors.values()]

    width=0.3
    labels = ['Collective Collective', 
            'Collective Neutral', 
            'Collective Self', 
            'Neutral Collective', 
            'Neutral Neutral', 
            'Neutral Self', 
            'Self Collective', 
            'Self Neutral', 
            'Self Self']

    x = np.arange(9)
    plt.figure(figsize=(15, 7))
    plt.title("Study 1: Average Final Points Accumulated Across Prompts, Name Condition")
    #yerr=llama_SE_final, capsize=10, 
    llama_bars=plt.bar(np.arange(len(llama_basic_final)), llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Llama 4 Maverick')
    qwen_bars=plt.bar(np.arange(len(qwen_basic_final)) + width, qwen_basic_final, width=width, yerr=qwen_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
    plt.bar_label(llama_bars, fmt='%.1f', padding=5)
    plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
    plt.xticks(x+width/2, labels, rotation=-45, ha='left')
    plt.ylabel("Average Final Points Accumulated")
    plt.xlabel("Prompt Pairings: Llama 4 & Qwen 3")
    plt.ylim(bottom=180, top=350)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend()
    ax = plt.gca()
    y_major_step=10
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
    plt.tight_layout()
    plt.savefig("self_final_lq")


if __name__=="__main__":
    run_basic("./llama_qwen_final.json", "./lq_basic_final_SE.json")
    run_discrim("./self_llama_qwen_final.json", "./lq_self_final_SE.json")