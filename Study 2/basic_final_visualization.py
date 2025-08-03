import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

llama_basic_final = [291.68000000000006, 
             265.584, 
             263.36800000000005, 
             265.552, 
             263.28,
             264.168, 
             275.256,
             265.576,
             258.856]

qwen_basic_final = [261.68000000000006,
                259.024,
                259.008,
                255.072,
                256.6,
                262.528,
                255.37599999999998,
                253.75799999999995,
                253.89600000000002]

llama_SE_final=[4.730257250369447,
              0.36246757296646104,
              0.1900074413334101,
              0.3599633127097825,
              0.18185170376895268,
              0.25146368778734984,
              1.123680986509908,
              0.6136817512993032,
              0.17119097293598687]

qwen_SE_final=[1.7019918903475397,
                 0.24017741030137915,
                 0.2369543866758089,
                 0.0758933797026667,
                 0.05021626856647094,
                 0.5148776494754147,
                 0.061315564094617095,
                 0.49997649104258673,
                 0.16805550693041196]

# gpt_basic_final=[
#     220.0,
#     217.00000000000003,
#     142.70000000000002,
#     265.0,
#     214.95000000000002,
#     187.5666666666667,
#     271.0833333333333,
#     212.9,
#     198.15
# ]

# claude_basic_final=[
#     220.0,
#     210.33333333333334,
#     277.3666666666667,
#     165.0,
#     205.95000000000002,
#     222.9,
#     152.08333333333331,
#     201.5666666666667,
#     203.15
# ]

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
#yerr=llama_SE_final, capsize=10, 
llama_bars=plt.bar(np.arange(len(llama_basic_final)), llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='lightcoral', label='Llama 4 Maverick')
qwen_bars=plt.bar(np.arange(len(qwen_basic_final)) + width, qwen_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Instruct 2507')
plt.bar_label(llama_bars, fmt='%.1f', padding=5)
plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: Llama 4 & Qwen3")
plt.ylim(bottom=200)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()

