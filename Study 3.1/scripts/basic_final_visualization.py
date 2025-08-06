import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

gpt_basic_final = [288.75200000000007,
                   275.91600000000005,
                   254.496]

claude_basic_final = [263.07200000000006,
                      257.83600000000007,
                      256.17600000000004]

llama_basic_final = [291.59200000000004,
                     282.91600000000005,
                     257.81600000000003]

qwen_basic_final = [288.97200000000004,
                    276.99600000000004,
                    254.85600000000002]

gpt_SE_final=[3.8851857044769402,
              5.088420803177344,
              5.767824007256806]

claude_SE_final=[5.250573436160282,
                 5.9245681215872565,
                 7.723181095116694]

llama_SE_final=[7.951738773850154,
                7.070507292564512,
                8.720525003909334]

qwen_SE_final=[3.81929540901355,
               5.123024469147885,
               5.790168187854993]

width=0.2
labels = ['Collective',
          'Neutral',
          'Self']

x = np.arange(len(gpt_basic_final))
plt.figure(figsize=(15, 7))
plt.title("Average Final Points Accumulated Across Prompts, Four Player - No Name Condition")
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
plt.savefig("basic")

