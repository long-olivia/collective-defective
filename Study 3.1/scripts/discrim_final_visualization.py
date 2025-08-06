import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

gpt_basic_final = [297.5759999999999,
                   281.7120000000001,
                   253.51999999999995]

claude_basic_final = [280.756,
                      257.4120000000001,
                      252.73999999999995]

llama_basic_final = [310.3159999999999,
                     281.3520000000001,
                     261.28]

qwen_basic_final = [299.6359999999999,
                    283.2320000000001,
                    255.35999999999996]

gpt_SE_final=[4.651672785823183,
              6.754242221670169,
              7.231600020797607]

claude_SE_final=[6.505077091783615,
                 5.665596599422873,
                 6.540160235835202]

llama_SE_final=[10.36695914797622,
                11.01696907530705,
                9.688533571433803]

qwen_SE_final=[4.5902472233342735,
               6.515585852464225,
               6.953173965779942]

width=0.2
labels = ['Collective',
          'Neutral',
          'Self']

x = np.arange(len(gpt_basic_final))
plt.figure(figsize=(15, 7))
plt.title("Average Final Points Accumulated Across Prompts, Four Player - Name Condition")
gpt_bars=plt.bar(x - 1.5*width, gpt_basic_final, width=width, yerr=gpt_SE_final, capsize=10, color='powderblue', label='GPT-4o')
claude_bars=plt.bar(x - 0.5*width, claude_basic_final, width=width, yerr=claude_SE_final, capsize=10, color='teal', label='Sonnet 4')
llama_bars=plt.bar(x + 0.5*width, llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Llama 4 Maverick')
qwen_bars=plt.bar(x + 1.5*width, qwen_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
plt.bar_label(gpt_bars, fmt='%.1f', padding=5)
plt.bar_label(claude_bars, fmt='%.1f', padding=5)
plt.bar_label(llama_bars, fmt='%.1f', padding=5)
plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
plt.xticks(x, labels)
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: GPT-4o, Sonnet 4, Llama 4, & Qwen 3")
plt.ylim(bottom=180,top=350)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.savefig("discrim")

