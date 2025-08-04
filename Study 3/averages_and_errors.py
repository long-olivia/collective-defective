import os
import json
import numpy as np

discrim_means={
    "CC": [[5.0, 5.36, 5.48, 5.6, 5.72, 5.72, 5.76, 5.72, 5.76, 4.92, 5.56, 5.52, 5.56, 5.52, 5.52, 5.52, 5.52, 5.52, 5.52, 5.12], 
           [5.36, 5.32, 5.52, 5.64, 5.72, 5.76, 5.72, 5.76, 5.72, 5.76, 4.92, 5.24, 5.28, 5.44, 5.52, 5.52, 5.52, 5.52, 5.52, 5.52]], 
           
    "CN": [[5.08, 5.28, 5.16, 5.2, 5.32, 5.28, 5.36, 5.32, 5.4, 5.04, 5.36, 5.16, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.04], 
           [5.16, 5.4, 5.44, 5.32, 5.36, 5.44, 5.4, 5.48, 5.44, 5.52, 5.04, 5.4, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32]], 
    
    "CS": [[5.08, 5.32, 5.64, 5.76, 5.88, 5.92, 5.96, 5.96, 6.04, 5.12, 5.76, 5.84, 4.96, 5.28, 5.16, 5.48, 5.2, 5.6, 5.36, 5.4], 
           [5.0, 5.6, 5.52, 5.68, 5.88, 6.04, 6.0, 6.08, 6.0, 5.6, 4.76, 4.4, 5.04, 4.92, 5.16, 4.96, 5.12, 5.0, 5.2, 4.68]], 
    
    "NC": [[5.0, 5.36, 5.4, 5.52, 5.6, 5.12, 5.44, 5.52, 5.56, 4.64, 5.28, 5.44, 5.24, 5.28, 5.32, 5.36, 5.08, 5.28, 5.2, 4.88], 
           [5.4, 5.48, 5.68, 5.64, 5.72, 5.8, 5.4, 5.68, 5.8, 5.8, 5.0, 4.92, 5.52, 5.56, 5.6, 5.64, 5.68, 5.28, 5.52, 5.52]], 
           
    "NN": [[5.0, 4.96, 5.0, 5.04, 5.04, 4.88, 4.92, 5.0, 5.12, 2.92, 4.36, 4.2, 4.0, 4.2, 4.0, 4.0, 4.2, 4.0, 4.08, 3.4], 
           [5.0, 5.16, 5.12, 5.16, 5.04, 5.04, 4.88, 4.92, 5.0, 5.12, 2.92, 3.76, 3.8, 4.0, 4.0, 4.4, 4.0, 4.2, 3.8, 4.12]], 
           
    "NS": [[5.0, 5.12, 5.16, 5.2, 5.2, 5.0, 5.0, 5.2, 5.0, 4.0, 4.8, 4.6, 4.4, 4.6, 4.0, 4.2, 4.2, 4.4, 4.2, 4.0], 
           [5.0, 5.08, 5.2, 5.2, 5.24, 5.24, 5.04, 4.8, 5.0, 5.0, 4.0, 4.12, 4.4, 4.2, 4.48, 4.2, 4.4, 4.4, 4.6, 4.4]], 
           
    "SC": [[5.0, 5.2, 5.4, 5.56, 5.64, 5.6, 5.44, 5.56, 5.48, 3.6, 4.72, 3.96, 4.2, 4.24, 4.24, 4.32, 4.24, 4.28, 4.28, 4.28], 
           [5.6, 5.76, 5.68, 5.8, 5.8, 5.84, 5.8, 5.64, 5.8, 5.68, 3.8, 4.52, 4.16, 4.28, 4.52, 4.2, 4.24, 4.36, 4.2, 3.96]], 
           
    "SN": [[5.0, 5.04, 5.12, 5.2, 5.24, 5.12, 5.12, 5.32, 5.32, 2.32, 4.12, 3.52, 3.72, 2.92, 3.2, 2.92, 3.4, 2.92, 3.2, 2.52], 
           [5.04, 5.08, 5.12, 5.16, 5.24, 5.28, 5.16, 5.16, 5.36, 5.36, 2.36, 2.88, 3.16, 3.24, 3.36, 3.04, 3.16, 3.44, 2.96, 3.16]], 
    
    "SS": [[5.0, 5.12, 5.28, 5.28, 5.36, 5.12, 5.12, 5.12, 5.12, 1.84, 3.6, 2.0, 2.84, 2.2, 2.24, 2.6, 2.4, 2.4, 2.76, 2.0], 
           [5.0, 5.12, 5.04, 5.24, 5.24, 5.32, 5.12, 5.12, 5.12, 5.08, 1.88, 2.56, 2.0, 2.96, 2.2, 2.6, 2.4, 2.2, 2.0, 2.12]]
}

basic_means={
    "CC": [[5.0, 5.2, 5.36, 5.48, 5.6, 5.68, 5.68, 5.72, 5.68, 5.72, 5.68, 5.72, 5.68, 5.72, 5.68, 5.72, 5.68, 5.68, 5.68, 2.44], 
           [6.36, 6.08, 6.12, 6.2, 6.28, 6.36, 6.44, 6.4, 6.44, 6.4, 6.44, 6.4, 6.44, 6.4, 6.4, 6.36, 6.4, 6.4, 6.4, 6.08]], 
    
    "CN": [[5.0, 5.04, 5.12, 5.16, 5.08, 5.24, 5.2, 5.28, 5.24, 5.32, 5.28, 5.32, 5.28, 5.32, 5.28, 5.32, 5.28, 5.32, 5.28, 1.2], 
           [5.28, 5.24, 5.24, 5.04, 5.28, 5.28, 5.44, 5.4, 5.48, 5.44, 5.48, 5.44, 5.48, 5.44, 5.48, 5.44, 5.48, 5.44, 5.16, 5.16]], 
    
    "CS": [[5.0, 5.04, 5.08, 5.12, 5.16, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 1.6], 
           [5.2, 5.12, 5.12, 5.16, 5.2, 5.24, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 5.28, 4.48]], 
           
    "NC": [[5.0, 5.04, 5.0, 5.0, 4.96, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 1.4], 
           [5.68, 5.44, 5.4, 5.32, 5.28, 5.28, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24, 5.24]], 
    
    "NN": [[5.0, 5.0, 5.12, 5.08, 5.16, 5.12, 5.16, 5.12, 5.16, 5.12, 5.12, 4.8, 5.0, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 1.8], 
           [5.24, 5.28, 5.24, 5.32, 5.28, 5.32, 5.28, 5.32, 5.28, 5.28, 5.24, 5.28, 4.96, 5.16, 4.96, 4.96, 4.96, 4.96, 4.96, 4.96]], 
           
    "NS": [[5.0, 5.0, 5.04, 5.08, 5.4, 5.52, 5.52, 5.6, 5.48, 5.52, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 1.56], 
           [5.08, 5.0, 4.96, 5.28, 5.24, 5.36, 5.44, 5.4, 5.44, 5.36, 5.4, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 5.44, 4.92]], 
           
    "SC": [[4.6, 5.12, 5.2, 5.44, 5.56, 5.56, 5.4, 5.08, 5.32, 5.36, 5.16, 5.12, 5.24, 5.04, 4.88, 4.96, 5.0, 5.04, 4.84, 1.0], 
           [6.28, 6.0, 6.28, 6.2, 6.28, 5.88, 5.8, 6.08, 6.08, 5.76, 5.64, 5.92, 5.72, 5.6, 5.92, 5.96, 5.96, 5.56, 5.96, 5.92]], 
           
    "SN": [[4.8, 5.0, 5.28, 5.2, 5.24, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.32, 5.12, 0.8], 
           [5.12, 5.08, 5.24, 5.48, 5.44, 5.36, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.2]], 
    
    "SS": [[4.8, 5.0, 4.84, 4.8, 4.84, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 0.0], 
           [5.04, 4.92, 5.04, 4.88, 4.8, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84, 4.0]]
}

discrim_final={
    "CC": [266.24, 265.88], 
    "CN": [264.632, 262.35200000000003], 
    "CS": [263.168, 267.248], 
    "NC": [267.408, 262.288], 
    "NN": [253.888, 252.768], 
    "NS": [256.54400000000004, 255.824], 
    "SC": [260.66400000000004, 256.26400000000007], 
    "SN": [249.92800000000003, 248.44800000000004], 
    "SS": [244.776, 243.856]
}

basic_final={
    "CC": [279.68000000000006, 261.68000000000006], 
    "CN": [265.584, 259.024], 
    "CS": [263.36800000000005, 259.008], 
    "NC": [265.552, 255.072], 
    "NN": [263.28, 256.6], 
    "NS": [264.168, 262.528], 
    "SC": [275.256, 255.37599999999998], 
    "SN": [265.576, 259.05600000000004], 
    "SS": [258.856, 253.89600000000002]
}

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
                if directory=="basic_llama_qwen_results":
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
                     if directory=="four_player_results":
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
        if (directory_name == "four_player_results"):
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
        if (directory_name == "four_players_results"):
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
        if (directory_name == "four_players_results"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

if __name__ == "__main__":
    run("four_player_results")
    with open("four_final.json", 'w') as b:
        json.dump(basic_final_avg, b)
    with open("four_rounds.json", 'w') as f:
        json.dump(basic_round_avg, f)
    run("four_players_discrim_results")
    with open("self_four_final.json", 'w') as c:
        json.dump(discrim_final_avg, c)
    with open("self_four_rounds.json", 'w') as g:
        json.dump(discrim_round_avg, g)

    # run_error("basic_llama_qwen_results")
    # with open("basic_lq_round_SE.json", 'w') as b:
    #     json.dump(basic_round_SE, b)
    # run_error_final("basic_llama_qwen_results")
    # with open("basic_lq_final_SE.json", 'w') as b:
    #     json.dump(basic_final_SE, b)
    # run_error("self_llama_qwen_results")
    # with open("self_lq_rounds_SE.json", 'w') as s:
    #     json.dump(discrim_round_SE, s)
    # run_error_final("basic_llama_qwen_results")
    # with open("self_lq_final_SE.json", 'w') as b:
    #     json.dump(discrim_final_SE, b)