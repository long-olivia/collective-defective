import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

gpt_self_final=[324.922,
                288.752,
                261.18399999999997,
                325.87000000000006,
                279.18600000000004,
                251.84399999999994,
                324.54,
                261.0240000000001,
                243.122]

claude_self_final=[271.03200000000004,
                   280.682,
                   282.3239999999999,
                   254.19,
                   267.31600000000003,
                   266.644,
                   231.85000000000002,
                   242.79400000000012,
                   243.18200000000002]

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
plt.title("Average Final Points Accumulated Across Prompts, Name Condition")
gpt_bars=plt.bar(np.arange(len(gpt_self_final)), gpt_self_final, width=width, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(np.arange(len(claude_self_final)) + width, claude_self_final, width=width, color='teal', label='Sonnet 4')
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
