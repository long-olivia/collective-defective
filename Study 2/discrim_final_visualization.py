import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

llama_self_final=[251.06800000000004,
                268.7980000000001,
                276.68600000000004,
                241.11800000000005,
                257.728,
                261.08,
                247.99199999999996,
                246.81399999999988,
                244.70200000000006]

qwen_self_final=[275.26800000000003,
                278.74800000000005,
                281.576,
                246.81800000000007,
                254.47800000000004,
                262.49,
                235.612,
                238.4039999999999,
                240.52200000000005]


llama_SE_final=[0.5242177296747138,
              0.3423038475065299,
              1.0659411874635047,
              2.2124556616140665,
              1.4894625709750706,
              0.18052761590039812,
              1.3005000994036366,
              0.061009564792436626,
              0.2918397408967686]

qwen_SE_final=[0.6109608821204975,
              0.6766712897137847,
              0.9707686482783796,
              0.9054529003975746,
              0.9236664962530007,
              0.20918755710819836,
              0.34012813759220295,
              0.20005729037503575,
              0.19268378565420222]

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
gpt_bars=plt.bar(np.arange(len(llama_self_final)), llama_self_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Llama 4 Maverick')
claude_bars=plt.bar(np.arange(len(qwen_self_final)) + width, qwen_self_final, width=width, yerr=qwen_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: Llama 4 & Qwen3")
plt.ylim(bottom=230)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()
