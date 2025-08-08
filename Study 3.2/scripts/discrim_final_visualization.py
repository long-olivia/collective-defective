import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

with open("qwen_final.json") as f:
    a=json.load(f)

with open("qwen_basic_final_SE.json") as f:
    b=json.load(f)

gpt_basic_final=[]
claude_basic_final=[]
llama_basic_final=[]
qwen_basic_final=[]
gpt_SE_final=[]
claude_SE_final=[]
llama_SE_final=[]
qwen_SE_final=[]

for arr in a.values():
    gpt_basic_final.append(arr[0])
    claude_basic_final.append(arr[1])
    llama_basic_final.append(arr[2])
    qwen_basic_final.append(arr[3])

for arr in b.values():
    gpt_SE_final.append(arr[0])
    claude_SE_final.append(arr[1])
    llama_SE_final.append(arr[2])
    qwen_SE_final.append(arr[3])


width=0.2
labels = ['Collective',
          'Neutral',
          'Self']

x = np.arange(len(gpt_basic_final))
plt.figure(figsize=(15, 7))
plt.title("Average Final Points Accumulated Across Prompts, Qwen3 Four Player - No Name Condition")
gpt_bars=plt.bar(x - 1.5*width, gpt_basic_final, width=width, yerr=gpt_SE_final, capsize=10, color='powderblue', label='Qwen3 235B A22B Instruct 2507')
claude_bars=plt.bar(x - 0.5*width, claude_basic_final, width=width, yerr=claude_SE_final, capsize=10, color='teal', label='Qwen3 235B A22B Instruct 2507')
llama_bars=plt.bar(x + 0.5*width, llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Qwen3 235B A22B Instruct 2507')
qwen_bars=plt.bar(x + 1.5*width, qwen_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.bar_label(llama_bars, fmt='%.1f', padding=5)
plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
plt.xticks(x, labels)
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: All Qwen3")
plt.ylim(bottom=180,top=350)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.savefig("qwen_basic_final")

