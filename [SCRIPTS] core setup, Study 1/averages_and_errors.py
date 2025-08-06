import os
import json
import numpy as np

#final average dictionary. the key is the prompt pair, and the value is the array
basic_final_avg={}
discrim_final_avg={}
base_dir="."
#round average dictionary. the key is the prompt pair, and the value is an array of averages
basic_round_avg={}
discrim_round_avg={}

basic_round_SE={}
discrim_round_SE={}

basic_final_SE={}
discrim_final_SE={}

"""
The final_average function takes a path string. It goes into the specified directory, sums up
total_points_after_round for each model's 20th round, and averages those points. It returns an array with these two averages.
"""

def final_average(path):
    a_sum=0
    b_sum=0
    average=[0]*2
    files=os.listdir(path)
    for file_name in files:
        file_path = os.path.join(base_dir, path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round in data:
                if round["round"] == 20:
                    a_sum+=round["a_total_points_after_round"]
                    b_sum+=round["b_total_points_after_round"]
    a_sum/=100
    b_sum/=100
    average[0]=a_sum
    average[1]=b_sum
    return average

"""
The per_round_avg function takes a path string. It goes into the specified directory, and for each round, it 
sums up the contribution for each model. When the sums are done, everything is averaged.
"""

def per_round_avg(path):
    a_round_avg=[0]*20
    b_round_avg=[0]*20
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(base_dir, path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                a_round_avg[index-1]+=round_data["a_contribution"]
                b_round_avg[index-1]+=round_data["b_contribution"]
    a_round_avg[:] = [x / 100 for x in a_round_avg]
    b_round_avg[:] = [x / 100 for x in b_round_avg]
    average=[a_round_avg, b_round_avg]
    return average

"""
The error function calculates the standard for each round for each prompt pairing.
"""

def error(directory, pair):
    a_err=[0]*20
    b_err=[0]*20
    path=os.path.join(base_dir, directory, pair)
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                if directory=="basic_results":
                    a_result=(round_data["a_contribution"]-basic_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-basic_means[pair][1][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
                else:
                    a_result=(round_data["a_contribution"]-discrim_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-discrim_means[pair][1][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
    a_err[:] = [((x / 100) ** 0.5) * (1.960/10) for x in a_err]
    b_err[:] = [((x / 100) ** 0.5) * (1.960/10) for x in b_err]
    error=[a_err, b_err]
    return error

"""
The error_final function calculates the standard error for the final points accumulated per prompt pairing.
"""
def error_final(directory, pair):
    a_fin=0
    b_fin=0
    path=os.path.join(base_dir, directory, pair)
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                if index==20:
                     if directory=="basic_results":
                        a_result=(round_data["a_total_points_after_round"]-basic_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-basic_final[pair][1])**2
                        a_fin+=a_result
                        b_fin+=b_result
                     else:
                        a_result=(round_data["a_total_points_after_round"]-discrim_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-discrim_final[pair][1])**2
                        a_fin+=a_result
                        b_fin+=b_result
    a_fin = ((a_fin / 100) ** 0.5) * (1.960/10)
    b_fin = ((b_fin / 100) ** 0.5) * (1.960/10)
    error_final=[a_fin, b_fin]
    return error_final

"""
The run function takes a directory (basic_results or self_results) and runs final_average & per_round_avg for each prompt pair.
It adds the return value of each function to the global dictionaries defined above. After it's done, it dumps the global dictionaries
into a json file within the upper level directory.
"""

def run(directory_name):
    prompt_pairs=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    for name in prompt_pairs:
        path=f"{directory_name}/{name}"
        if (directory_name == "basic_results"):
            basic_final_avg[name] = final_average(path)
            basic_round_avg[name] = per_round_avg(path)
        else:
            discrim_final_avg[name] = final_average(path)
            discrim_round_avg[name] = per_round_avg(path)

"""
The run_error_final function runs standard error calculations for final averages.
"""
def run_error_final(directory_name):
    prompt_pairs=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    for name in prompt_pairs:
        result=error_final(directory_name, name)
        if (directory_name == "basic_results"):
            basic_final_SE[name] = result
        else:
            discrim_final_SE[name] = result

"""
The run_error function runs standard error calculations for round averages.
"""
def run_error(directory_name):
    prompt_pairs=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    for name in prompt_pairs:
        result=error(directory_name, name)
        if (directory_name == "basic_results"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

def load_files(basic_final, basic_rounds, discrim_final, discrim_rounds):
    bf_path=os.path.join(base_dir, basic_final)
    with open(bf_path) as file:
        basic_final=json.load(file)
        print(basic_final)
    br_path=os.path.join(base_dir, basic_rounds)
    with open(br_path) as file:
        basic_means=json.load(file)
        print(basic_means)
    df_path=os.path.join(base_dir, discrim_final)
    with open(df_path) as file:
        discrim_final=json.load(file)
        print(discrim_final)
    dr_path=os.path.join(base_dir, discrim_rounds)
    with open(dr_path) as file:
        discrim_means=json.load(file)
        print(discrim_means)

if __name__ == "__main__":
    load_files("basic_lq_final.json", "basic_lq_rounds.json", "self_lq_final.json", "self_lq_rounds.json")
    # run("llama4_qwen")
    # with open("llama_qwen_final.json", 'w') as b:
    #     json.dump(basic_final_avg, b)
    # with open("llama_qwen_rounds.json", 'w') as f:
    #     json.dump(basic_round_avg, f)
    # run("self_llama4_qwen")
    # with open("self_llama_qwen_final.json", 'w') as c:
    #     json.dump(discrim_final_avg, c)
    # with open("self_llama_qwen_rounds.json", 'w') as g:
    #     json.dump(discrim_round_avg, g)

    run_error("llama4_qwen_results")
    with open("lq_basic_rounds_SE.json", 'w') as b:
        json.dump(basic_round_SE, b)
    run_error_final("llama4_qwen_results")
    with open("lq_basic_final_SE.json", 'w') as b:
        json.dump(basic_final_SE, b)
    run_error("self_llama4_qwen_results")
    with open("lq_self_rounds_SE.json", 'w') as s:
        json.dump(discrim_round_SE, s)
    run_error_final("self_llama4_qwen_results")
    with open("lq_self_final_SE.json", 'w') as s:
        json.dump(discrim_final_SE, s)