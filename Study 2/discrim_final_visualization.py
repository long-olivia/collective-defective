import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

llama_self_final=[293.728000000000074,
                292.73600000000005,
                287.976,
                280.16,
                276.768,
                278.136,
                279.24800000000005,
                271.856,
                265.38400000000007]

qwen_self_final=[284.80800000000005,
                294.29600000000005,
                288.376,
                266.48,
                271.528,
                274.81600000000003,
                265.56800000000004,
                266.216,
                264.26400000000007]


llama_SE_final=[1.3905811179429604,
              0.22958204670338125,
              0.5721092895224712,
              0.24347246994764418,
              1.2531850363463422,
              0.4683954508165934,
              3.2348037342709413,
              3.0628742850763735,
              0.8954760511828974]

qwen_SE_final=[2.2597041832147617,
              0.1972479207576071,
              0.6200427808832406,
              1.3919750851086707,
              0.9225911012634574,
              0.7266920926142679,
              3.1815639241839726,
              3.2484125125680587,
              0.5131887273626083]

width=0.35
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
gpt_bars=plt.bar(np.arange(len(llama_self_final)), llama_self_final, width=width, yerr=llama_SE_final, capsize=10, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(np.arange(len(qwen_self_final)) + width, qwen_self_final, width=width, yerr=qwen_SE_final, capsize=10, color='teal', label='Sonnet 4')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: GPT-4o & Sonnet 4")
plt.ylim(bottom=230)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()
