import json
import pandas as pd
from scipy.stats import spearmanr

def load_and_correlate(basic_round, basic_sent, discrim_round, discrim_sent, study_models):
    with open(basic_round) as file:
        basic_r = json.load(file)
    with open(basic_sent) as file:
        basic_s = json.load(file)
    with open(discrim_round) as file:
        discrim_r = json.load(file)
    with open(discrim_sent) as file:
        discrim_s = json.load(file)

    data_rows = []

    for prompt_pair in basic_r:
        for player_idx in range(4):
            for round_idx in range(20):
                contr_basic = basic_r[prompt_pair][player_idx][round_idx]
                sent_basic = basic_s[prompt_pair][player_idx][round_idx]

                data_rows.append({
                    "prompt_pair": prompt_pair,
                    "player": player_idx,
                    "round": round_idx,
                    "condition": "no_name",
                    "contribution": contr_basic,
                    "avg_sentiment": sent_basic
                })

                contr_name = discrim_r[prompt_pair][player_idx][round_idx]
                sent_name = discrim_s[prompt_pair][player_idx][round_idx]

                data_rows.append({
                    "prompt_pair": prompt_pair,
                    "player": player_idx,
                    "round": round_idx,
                    "condition": "name",
                    "contribution": contr_name,
                    "avg_sentiment": sent_name
                })

    df = pd.DataFrame(data_rows)

    def safe_spearman_corr(g):
        if len(g) < 2:
            return None
        if g['avg_sentiment'].nunique() < 2 or g['contribution'].nunique() < 2:
            return None
        corr, _ = spearmanr(g['avg_sentiment'], g['contribution'])
        return corr

    corr_results = df.groupby(['player', 'prompt_pair', 'condition']).apply(safe_spearman_corr).reset_index(name='spearman_corr')
    corr_results.to_json(f"{study_models}_correlations.json", orient='records', indent=2)

if __name__=="__main__":
    load_and_correlate(
        "four_rounds.json", "four_agg.json",
        "self_four_rounds.json", "self_four_agg.json",
        "study31"
    )