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
        "CCCC": [[0]*20, [0]*20, [0]*20, [0]*20],
        "NNNN": [[0]*20, [0]*20, [0]*20, [0]*20],
        "SSSS": [[0]*20, [0]*20, [0]*20, [0]*20]
    }

    name_ratio={
        "CCCC": [[0]*20, [0]*20, [0]*20, [0]*20],
        "NNNN": [[0]*20, [0]*20, [0]*20, [0]*20],
        "SSSS": [[0]*20, [0]*20, [0]*20, [0]*20]
    }

    for prompt_pair in basic_r:
        for i in range(0,4):
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
    load("qwen_rounds.json", "qwen_agg.json", "self_qwen_rounds.json", "self_qwen_agg.json", "study32_qwen")
