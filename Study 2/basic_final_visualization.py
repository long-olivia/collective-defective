import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

llama_basic_final = [283.2720000000001,
                279.616,
                266.328,
                271.84799999999996,
                273.896,
                266.896,
                260.184,
                265.576,
                263.048]

qwen_basic_final = [279.2320000000001,
                274.89599999999996,
                269.488,
                259.52799999999996,
                268.13599999999997,
                265.776,
                247.74400000000003,
                258.592,
                259.20799999999997]

llama_SE_final=[2.692383274514989,
                0.4175599781314503,
                0.31075797851406783,
                2.529167308669765,
                1.3582498925791118,
                1.9070919233757522,
                3.2918780510418846,
                1.395724269557886,
                1.6440933270245957]

qwen_SE_final=[3.2445121606546703,
                 0.5193763625482302,
                 0.7964646018739194,
                 2.5035622731566103,
                 1.2511913921987006,
                 2.4669870614825573,
                 2.865873384865216,
                 1.0344599229332814,
                 1.9452575058161357]

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
llama_bars=plt.bar(np.arange(len(llama_basic_final)), llama_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='powderblue', label='GPT-4o')
qwen_bars=plt.bar(np.arange(len(qwen_basic_final)) + width, qwen_basic_final, width=width, yerr=llama_SE_final, capsize=10, color='teal', label='Sonnet 4')
plt.bar_label(llama_bars, fmt='%.1f', padding=5)
plt.bar_label(qwen_bars, fmt='%.1f', padding=5)
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

