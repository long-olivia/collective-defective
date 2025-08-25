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

discrim_means={}
basic_means={}
basic_final={}
discrim_final={}
"""
The final_average function takes a path string. It goes into the specified directory, sums up
total_points_after_round for each model's 20th round, and averages those points. It returns an array with these two averages.
"""

def final_average(path):
    a_sum=0
    b_sum=0
    c_sum=0
    d_sum=0
    average=[0]*4
    files=os.listdir(path)
    for file_name in files:
        file_path = os.path.join(base_dir, path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round in data:
                if round["round"] == 20:
                    a_sum+=round["a_total_points_after_round"]
                    b_sum+=round["b_total_points_after_round"]
                    c_sum+=round["c_total_points_after_round"]
                    d_sum+=round["d_total_points_after_round"]
    a_sum/=50
    b_sum/=50
    c_sum/=50
    d_sum/=50
    average[0]=a_sum
    average[1]=b_sum
    average[2]=c_sum
    average[3]=d_sum
    return average

"""
The per_round_avg function takes a path string. It goes into the specified directory, and for each round, it 
sums up the contribution for each model. When the sums are done, everything is averaged.
"""

def per_round_avg(path):
    a_round_avg=[0]*20
    b_round_avg=[0]*20
    c_round_avg=[0]*20
    d_round_avg=[0]*20
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(base_dir, path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                a_round_avg[index-1]+=round_data["a_contribution"]
                b_round_avg[index-1]+=round_data["b_contribution"]
                c_round_avg[index-1]+=round_data["c_contribution"]
                d_round_avg[index-1]+=round_data["d_contribution"]
    a_round_avg[:] = [x / 50 for x in a_round_avg]
    b_round_avg[:] = [x / 50 for x in b_round_avg]
    c_round_avg[:] = [x / 50 for x in c_round_avg]
    d_round_avg[:] = [x / 50 for x in d_round_avg]
    average=[a_round_avg, b_round_avg, c_round_avg, d_round_avg]
    return average

"""
The error function calculates the standard for each round for each prompt pairing.
"""

def error(directory, pair):
    a_err=[0]*20
    b_err=[0]*20
    c_err=[0]*20
    d_err=[0]*20
    path=os.path.join(base_dir, directory, pair)
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                if directory=="meta-llama":
                    a_result=(round_data["a_contribution"]-basic_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-basic_means[pair][1][index-1])**2
                    c_result=(round_data["c_contribution"]-basic_means[pair][2][index-1])**2
                    d_result=(round_data["d_contribution"]-basic_means[pair][3][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
                    c_err[index-1]+=c_result
                    d_err[index-1]+=d_result
                else:
                    a_result=(round_data["a_contribution"]-discrim_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-discrim_means[pair][1][index-1])**2
                    c_result=(round_data["c_contribution"]-discrim_means[pair][2][index-1])**2
                    d_result=(round_data["d_contribution"]-discrim_means[pair][3][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
                    c_err[index-1]+=c_result
                    d_err[index-1]+=d_result
    a_err[:] = [((x / 50) ** 0.5) * (1.960/(50 ** 0.5)) for x in a_err]
    b_err[:] = [((x / 50) ** 0.5) * (1.960/(50 ** 0.5)) for x in b_err]
    c_err[:] = [((x / 50) ** 0.5) * (1.960/(50 ** 0.5)) for x in c_err]
    d_err[:] = [((x / 50) ** 0.5) * (1.960/(50 ** 0.5)) for x in d_err]
    error=[a_err, b_err, c_err, d_err]
    return error

"""
The error_final function calculates the standard error for the final points accumulated per prompt pairing.
"""
def error_final(directory, pair):
    a_fin=0
    b_fin=0
    c_fin=0
    d_fin=0
    path=os.path.join(base_dir, directory, pair)
    files=os.listdir(path)
    for file_name in files:
        file_path=os.path.join(path, file_name)
        with open(file_path) as file:
            data=json.load(file)
            for round_data in data:
                index=round_data["round"]
                if index==20:
                     if directory=="meta-llama":
                        a_result=(round_data["a_total_points_after_round"]-basic_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-basic_final[pair][1])**2
                        c_result=(round_data["c_total_points_after_round"]-basic_final[pair][2])**2
                        d_result=(round_data["d_total_points_after_round"]-basic_final[pair][3])**2
                        a_fin+=a_result
                        b_fin+=b_result
                        c_fin+=c_result
                        d_fin+=d_result
                     else:
                        a_result=(round_data["a_total_points_after_round"]-discrim_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-discrim_final[pair][1])**2
                        c_result=(round_data["c_total_points_after_round"]-discrim_final[pair][2])**2
                        d_result=(round_data["d_total_points_after_round"]-discrim_final[pair][3])**2
                        a_fin+=a_result
                        b_fin+=b_result
                        c_fin+=c_result
                        d_fin+=d_result
    a_fin = ((a_fin / 50) ** 0.5) * (1.960/(50 ** 0.5))
    b_fin = ((b_fin / 50) ** 0.5) * (1.960/(50 ** 0.5))
    c_fin = ((c_fin / 50) ** 0.5) * (1.960/(50 ** 0.5))
    d_fin = ((d_fin / 50) ** 0.5) * (1.960/(50 ** 0.5))
    error_final=[a_fin, b_fin, c_fin, d_fin]
    return error_final

"""
The run function takes a directory (basic_results or self_results) and runs final_average & per_round_avg for each prompt pair.
It adds the return value of each function to the global dictionaries defined above. After it's done, it dumps the global dictionaries
into a json file within the upper level directory.
"""

def run(directory_name):
    prompt_pairs=["CCCC", "NNNN", "SSSS"]
    for name in prompt_pairs:
        path=f"{directory_name}/{name}"
        if (directory_name == "meta-llama"):
            basic_final_avg[name] = final_average(path)
            basic_round_avg[name] = per_round_avg(path)
        else:
            discrim_final_avg[name] = final_average(path)
            discrim_round_avg[name] = per_round_avg(path)

"""
The run_error_final function runs standard error calculations for final averages.
"""
def run_error_final(directory_name):
    prompt_pairs=["CCCC", "NNNN", "SSSS"]
    for name in prompt_pairs:
        result=error_final(directory_name, name)
        if (directory_name == "meta-llama"):
            basic_final_SE[name] = result
        else:
            discrim_final_SE[name] = result

"""
The run_error function runs standard error calculations for round averages.
"""
def run_error(directory_name):
    prompt_pairs=["CCCC", "NNNN", "SSSS"]
    for name in prompt_pairs:
        result=error(directory_name, name)
        if (directory_name == "meta-llama"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

def load_files(basic_fin, basic_r, discrim_fin, discrim_r):
    global basic_final, discrim_final, basic_means, discrim_means
    bf_path=os.path.join(base_dir, basic_fin)
    with open(bf_path) as file:
        basic_final=json.load(file)
    br_path=os.path.join(base_dir, basic_r)
    with open(br_path) as file:
        basic_means=json.load(file)
    df_path=os.path.join(base_dir, discrim_fin)
    with open(df_path) as file:
        discrim_final=json.load(file)
    dr_path=os.path.join(base_dir, discrim_r)
    with open(dr_path) as file:
        discrim_means=json.load(file)

if __name__ == "__main__":
    # run("meta-llama")
    # with open("meta_final.json", 'w') as b:
    #     json.dump(basic_final_avg, b)
    # with open("meta_rounds.json", 'w') as f:
    #     json.dump(basic_round_avg, f)
    # run("self_meta-llama")
    # with open("self_meta_final.json", 'w') as c:
    #     json.dump(discrim_final_avg, c)
    # with open("self_meta_rounds.json", 'w') as g:
    #     json.dump(discrim_round_avg, g)

    load_files("meta_final.json", "meta_rounds.json", "self_meta_final.json", "self_meta_rounds.json")
    run_error("meta-llama")
    with open("meta_basic_round_SE.json", 'w') as b:
        json.dump(basic_round_SE, b)
    run_error_final("meta-llama")
    with open("meta_basic_final_SE.json", 'w') as b:
        json.dump(basic_final_SE, b)
    run_error("self_meta-llama")
    with open("meta_self_rounds_SE.json", 'w') as s:
        json.dump(discrim_round_SE, s)
    run_error_final("self_meta-llama")
    with open("meta_self_final_SE.json", 'w') as b:
        json.dump(discrim_final_SE, b)