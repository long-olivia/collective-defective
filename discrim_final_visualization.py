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

claude_SE_final=[0.9598708088856314,
                 0.3859788105727871,
                 0.4377764733005109,
                 0.2785164758462859,
                 0.1477003014209899,
                 0.1381007545130626,
                 0.7928473697749705,
                 0.4040362438085194,
                 0.3298340560099799]

gpt_SE_final=[0.09759341326432391,
              0.6420843574644529,
              0.031896277444965745,
              0.27699382568826253,
              0.36049156034546415,
              0.19138112704461888,
              0.17400402795322537,
              0.8305719549178633,
              0.3309985694855612]

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
gpt_bars=plt.bar(np.arange(len(gpt_self_final)), gpt_self_final, width=width, yerr=gpt_SE_final, capsize=8, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(np.arange(len(claude_self_final)) + width, claude_self_final, width=width, yerr=claude_SE_final, capsize=8, color='teal', label='Sonnet 4')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: GPT-4o & Sonnet 4")
plt.ylim(bottom=220)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()
