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

gpt_SE_final=[0.05648360771213828,
              0.21830095245034387,
              0.6156207351379002,
              0.7859874819801875,
              0.36257890961055617,
              0.1323007849940359,
              0.7019609165601141,
              0.20350673365892438,
              0.004202651247875926]

claude_SE_final=[0.2876849441663415,
                 0.2943489443808685,
                 0.5807006306160345,
                 0.08277048358559827,
                 0.10033653214758165,
                 0.3467292844583052,
                 1.0706413232620913,
                 0.10382861676632238,
                 0.18709374845521218]

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
gpt_bars=plt.bar(np.arange(len(gpt_basic_final)), gpt_basic_final, width=width, yerr=gpt_SE_final, capsize=10, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(np.arange(len(claude_basic_final)) + width, claude_basic_final, width=width, yerr=claude_SE_final, capsize=10, color='teal', label='Sonnet 4')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: GPT-4o & Sonnet 4")
plt.ylim(bottom=200)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()

