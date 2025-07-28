import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

llama_basic_final = [291.976, 
             269.756, 
             220.02800000000008, 
             287.33600000000007, 
             256.7899999999999,
             215.17, 
             278.05,
             259.018,
             210.22799999999998]

qwen_basic_final = [281.586,
                267.44599999999997,
                257.25800000000004,
                263.03600000000006,
                251.89999999999992,
                236.57999999999996,
                255.39000000000004,
                253.75799999999995,
                219.80800000000002]

llama_SE_final=[1.1538948171176695,
              0.040868505876644576,
              0.05094531794994587,
              0.37857486695477105,
              0.24941701220789986,
              0.2854014816237598,
              0.6075689369863724,
              0.47964474234192117,
              0.07947382982320789]

qwen_SE_final=[0.9081189880519411,
                 0.2677536250181409,
                 2.085148685288669,
                 0.623987823390412,
                 0.10079221398382683,
                 0.2657309115127556,
                 0.5802352533077515,
                 0.6978722992962543,
                 0.08761806806572034]

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
qwen_bars=plt.bar(np.arange(len(qwen_basic_final)) + width, qwen_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='firebrick', label='Qwen3 235B A22B Thinking 2507')
plt.bar_label(llama_bars, fmt='%.1f', padding=5)
plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
plt.xticks(x+width/2, labels, rotation=-45, ha='left')
plt.ylabel("Average Final Points Accumulated")
plt.xlabel("Prompt Pairings: Llama 4 & Qwen 3")
plt.ylim(bottom=200)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.legend()
ax = plt.gca()
y_major_step=10
ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_major_step))
plt.tight_layout()
plt.show()

