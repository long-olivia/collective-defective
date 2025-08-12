import json

def load(basic_round, basic_sent, discrim_round, discrim_sent, study_models):
    with open(basic_round) as file:
        basic_r=json.load(file)

    with open(basic_sent) as file:
        basic_s=json.load(file)

    with open(discrim_round) as file:
        discrim_r=json.load(file)

    with open(discrim_sent) as file:
        discrim_s=json.load(file)

    no_name_ratio={
        "CC": [[0]*20, [0]*20],
        "CN": [[0]*20, [0]*20],
        "CS": [[0]*20, [0]*20],
        "NC": [[0]*20, [0]*20],
        "NN": [[0]*20, [0]*20],
        "NS": [[0]*20, [0]*20],
        "SC": [[0]*20, [0]*20],
        "SN": [[0]*20, [0]*20],
        "SS": [[0]*20, [0]*20]
    }

    name_ratio={
        "CC": [[0]*20, [0]*20],
        "CN": [[0]*20, [0]*20],
        "CS": [[0]*20, [0]*20],
        "NC": [[0]*20, [0]*20],
        "NN": [[0]*20, [0]*20],
        "NS": [[0]*20, [0]*20],
        "SC": [[0]*20, [0]*20],
        "SN": [[0]*20, [0]*20],
        "SS": [[0]*20, [0]*20]
    }

    for prompt_pair in basic_r:
        for i in range(0,2):
            for j in range(0,20):
                if basic_r[prompt_pair][i][j] == 0:
                    no_name_ratio[prompt_pair][i][j] = 0
                else:
                    no_name_ratio[prompt_pair][i][j] = basic_s[prompt_pair][i][j] / basic_r[prompt_pair][i][j]
                if discrim_r[prompt_pair][i][j] == 0:
                    name_ratio[prompt_pair][i][j] = 0
                else:
                    name_ratio[prompt_pair][i][j] = discrim_s[prompt_pair][i][j] / discrim_r[prompt_pair][i][j]
        
    with open(f"no_name_{study_models}.json", 'w') as f:
        json.dump(no_name_ratio, f)
    with open(f"name_{study_models}.json", 'w') as f:
        json.dump(name_ratio, f)

if __name__=="__main__":
    load("basic_lq_rounds.json", "basic_LQ_sentiment.json", "self_lq_rounds.json", "self_LQ_sentiment.json", "study2_llama_qwen")
