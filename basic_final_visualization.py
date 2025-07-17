import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

gpt_basic_final = [322.872, 
             306.4719999999999, 
             191.40600000000003, 
             299.90200000000004, 
             273.70399999999995,
             206.74799999999996, 
             324.19000000000005,
             233.558,
             203.044]

claude_basic_final = [305.332,
                305.59199999999987,
                330.37600000000003,
                235.78200000000004,
                261.10400000000004,
                217.67799999999997,
                214.62,
                217.418,
                208.46400000000003]

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
plt.title("Average Final Points Accumulated Across Prompts, No Name Condition")
gpt_bars=plt.bar(np.arange(len(gpt_basic_final)), gpt_basic_final, width=width, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(np.arange(len(claude_basic_final)) + width, claude_basic_final, width=width, color='teal', label='Sonnet 4')
plt.bar_label(gpt_bars, fmt='%.1f')
plt.bar_label(claude_bars, fmt='%.1f')
plt.xticks(x, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: GPT-4o & Sonnet 4")
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()
