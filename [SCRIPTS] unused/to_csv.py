import pandas as pd

with open('corpus/new_basic_results_CC.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv('CC_basic.csv', encoding='utf-8', index=False)