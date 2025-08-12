import json
import pandas as pd

def load_and_calculate_correlation(basic_round, basic_sent, discrim_round, discrim_sent, study_models):
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
        for model_idx in range(2):
            for round_idx in range(20):
                contribution_basic = basic_r[prompt_pair][model_idx][round_idx]
                sentiment_basic = basic_s[prompt_pair][model_idx][round_idx]

                data_rows.append({
                    "prompt_pair": prompt_pair,
                    "condition": "no_name",
                    "player": model_idx,
                    "round": round_idx,
                    "contribution": contribution_basic,
                    "avg_sentiment": sentiment_basic
                })

                contribution_discrim = discrim_r[prompt_pair][model_idx][round_idx]
                sentiment_discrim = discrim_s[prompt_pair][model_idx][round_idx]

                data_rows.append({
                    "prompt_pair": prompt_pair,
                    "condition": "name",
                    "player": model_idx,
                    "round": round_idx,
                    "contribution": contribution_discrim,
                    "avg_sentiment": sentiment_discrim
                })

    df = pd.DataFrame(data_rows)

    group_pp = df.groupby(['player', 'prompt_pair', 'condition'])
    corr_pp = group_pp.apply(
        lambda g: g['avg_sentiment'].corr(g['contribution'], method='spearman') if len(g) > 1 else None
    ).reset_index(name='spearman_corr')

    corr_pp.to_json(f"{study_models}.json", orient='records', indent=2)

if __name__=="__main__":
    load_and_calculate_correlation(
        "basic_lq_rounds.json", "basic_LQ_sentiment.json", "self_lq_rounds.json", "self_LQ_sentiment.json", "study2_lq")