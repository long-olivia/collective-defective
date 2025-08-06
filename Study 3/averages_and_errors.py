import os
import json
import numpy as np

discrim_means={
    "CCCC": [[7.9, 7.88, 7.94, 8.02, 8.06, 8.08, 8.08, 8.08, 8.1, 8.1, 8.1, 8.1, 8.12, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.12], 
             [8.24, 8.44, 8.84, 8.96, 8.98, 8.98, 8.98, 8.98, 9.0, 9.04, 9.0, 9.0, 8.98, 8.98, 8.96, 8.96, 8.96, 8.94, 8.94, 8.94], 
             [7.1, 7.2, 7.34, 7.44, 7.48, 7.52, 7.48, 7.5, 7.5, 7.46, 7.46, 7.44, 7.44, 7.44, 7.46, 7.46, 7.46, 7.46, 7.44, 7.46], 
             [5.74, 7.9, 7.96, 7.98, 8.08, 8.08, 8.12, 8.12, 8.12, 8.14, 8.1, 8.1, 8.1, 8.08, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1]], 
             
    "NNNN": [[6.06, 6.18, 6.32, 6.46, 6.46, 6.5, 6.36, 6.3, 6.16, 6.06, 6.04, 5.9, 5.82, 5.84, 5.76, 5.76, 5.72, 5.74, 5.7, 5.62], 
             [6.9, 7.34, 8.04, 8.3, 7.96, 7.96, 7.76, 7.66, 7.56, 7.34, 7.12, 6.96, 6.9, 6.82, 6.8, 6.76, 6.72, 6.72, 6.72, 6.72], 
             [6.16, 6.22, 6.36, 6.46, 6.44, 6.42, 6.44, 6.34, 6.26, 6.0, 5.96, 5.86, 5.84, 5.82, 5.76, 5.76, 5.74, 5.72, 5.72, 5.84], 
             [5.08, 6.22, 6.44, 6.36, 6.48, 6.42, 6.32, 6.3, 6.24, 5.88, 5.82, 5.88, 5.78, 5.78, 5.74, 5.74, 5.7, 5.68, 5.66, 5.72]], 
             
    "SSSS": [[5.58, 5.74, 6.02, 6.0, 6.1, 5.92, 5.88, 5.62, 5.28, 4.82, 4.7, 4.36, 4.08, 3.86, 3.68, 3.6, 3.5, 3.5, 3.44, 3.4], 
             [6.32, 6.4, 6.74, 6.7, 6.68, 6.3, 5.94, 5.56, 5.32, 4.86, 4.18, 3.88, 3.6, 3.46, 3.4, 3.32, 3.3, 3.32, 3.3, 3.28], 
             [5.3, 5.68, 5.76, 5.78, 5.52, 5.44, 5.36, 5.28, 5.04, 4.08, 4.16, 3.86, 3.62, 3.5, 3.32, 3.22, 3.2, 3.12, 3.1, 2.98], 
             [5.02, 6.0, 6.04, 6.24, 6.02, 5.9, 5.58, 5.44, 5.2, 4.36, 4.36, 4.14, 4.12, 3.8, 3.56, 3.6, 3.54, 3.52, 3.42, 3.38]]
}

basic_means={
    "CCCC": [[6.14, 6.26, 6.36, 6.52, 6.6, 6.64, 6.7, 6.7, 6.74, 6.74, 6.78, 6.76, 6.76, 6.6, 6.76, 6.76, 6.78, 6.76, 6.76, 6.72], 
             [6.86, 7.48, 7.8, 8.0, 8.02, 8.08, 8.02, 8.1, 8.14, 8.18, 8.18, 8.14, 8.14, 8.1, 8.04, 8.02, 7.94, 7.86, 7.76, 7.66], 
             [6.14, 6.22, 6.2, 6.32, 6.4, 6.5, 6.54, 6.56, 6.58, 6.58, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.62, 6.56, 6.62, 6.56], 
             [5.86, 6.28, 6.44, 6.4, 6.56, 6.62, 6.7, 6.74, 6.74, 6.78, 6.74, 6.74, 6.76, 6.76, 6.76, 6.76, 6.76, 6.74, 6.72, 6.76]], 
    
    "NNNN": [[5.72, 6.04, 6.14, 6.26, 6.26, 6.24, 6.26, 6.24, 6.16, 6.16, 6.12, 6.06, 6.0, 5.94, 5.9, 5.82, 5.8, 5.68, 5.64, 5.42], 
             [6.44, 6.82, 7.32, 7.44, 7.44, 7.5, 7.44, 7.38, 7.32, 7.2, 7.14, 7.02, 6.92, 6.78, 6.62, 6.52, 6.36, 6.22, 6.08, 5.98], 
             [5.46, 5.68, 5.74, 5.84, 5.92, 5.88, 5.9, 5.88, 5.88, 5.82, 5.76, 5.68, 5.66, 5.54, 5.5, 5.46, 5.46, 5.46, 5.28, 5.06], 
             [5.12, 5.94, 6.24, 6.26, 6.28, 6.26, 6.22, 6.24, 6.16, 6.1, 6.08, 6.06, 5.98, 5.9, 5.84, 5.82, 5.76, 5.66, 5.6, 5.26]], 
        
    "SSSS": [[5.48, 5.48, 5.6, 5.58, 5.56, 5.38, 5.22, 5.04, 4.86, 4.62, 4.58, 4.4, 4.32, 4.34, 4.24, 4.24, 4.2, 4.18, 4.04, 3.04], 
             [5.42, 5.96, 6.3, 6.02, 5.92, 5.58, 5.22, 4.98, 4.8, 4.68, 4.62, 4.5, 4.26, 4.18, 4.0, 3.86, 3.66, 3.3, 2.94, 2.52], 
             [5.14, 5.36, 5.48, 5.6, 5.46, 5.32, 5.2, 5.04, 4.72, 4.5, 4.3, 4.12, 4.08, 4.08, 3.98, 3.96, 3.94, 3.86, 3.78, 3.16], 
             [5.08, 5.92, 5.78, 5.6, 5.48, 5.34, 5.08, 4.92, 4.86, 4.62, 4.4, 4.34, 4.56, 4.28, 4.2, 4.2, 4.12, 4.14, 3.96, 3.16]]
}

discrim_final={
    "CCCC": [297.5759999999999, 280.756, 310.3159999999999, 299.6359999999999], 
    "NNNN": [281.7120000000001, 257.4120000000001, 281.3520000000001, 283.2320000000001], 
    "SSSS": [253.51999999999995, 252.73999999999995, 261.28, 255.35999999999996]
}

basic_final={
    "CCCC": [288.75200000000007, 263.07200000000006, 291.59200000000004, 288.97200000000004], 
    "NNNN": [275.91600000000005, 257.83600000000007, 282.91600000000005, 276.99600000000004], 
    "SSSS": [254.496, 256.17600000000004, 257.81600000000003, 254.85600000000002]
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
        if (directory_name == "four_player_results"):
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
        if (directory_name == "four_player_results"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

if __name__ == "__main__":
    # run("four_player_results")
    # with open("four_final.json", 'w') as b:
    #     json.dump(basic_final_avg, b)
    # with open("four_rounds.json", 'w') as f:
    #     json.dump(basic_round_avg, f)
    # run("four_players_discrim_results")
    # with open("self_four_final.json", 'w') as c:
    #     json.dump(discrim_final_avg, c)
    # with open("self_four_rounds.json", 'w') as g:
    #     json.dump(discrim_round_avg, g)

    run_error("four_player_results")
    with open("four_basic_round_SE.json", 'w') as b:
        json.dump(basic_round_SE, b)
    run_error_final("four_player_results")
    with open("four_basic_final_SE.json", 'w') as b:
        json.dump(basic_final_SE, b)
    run_error("four_players_discrim_results")
    with open("four_self_rounds_SE.json", 'w') as s:
        json.dump(discrim_round_SE, s)
    run_error_final("four_players_discrim_results")
    with open("four_self_final_SE.json", 'w') as b:
        json.dump(discrim_final_SE, b)