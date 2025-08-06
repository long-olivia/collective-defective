import os
import json
import numpy as np

discrim_means={"CC": [[7.08, 7.04, 7.12, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24, 7.24], [7.24, 7.68, 7.68, 7.56, 7.68, 7.68, 7.68, 7.68, 7.72, 7.76, 7.68, 7.68, 7.68, 7.68, 7.68, 7.68, 7.68, 7.68, 7.68, 7.76]], "CN": [[7.84, 7.76, 7.76, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84], [6.44, 7.36, 7.84, 7.84, 7.92, 7.92, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84, 7.84]], "CS": [[7.16, 7.08, 7.24, 7.24, 7.36, 7.36, 7.4, 7.32, 7.4, 7.4, 7.44, 7.4, 7.44, 7.4, 7.44, 7.4, 7.44, 7.4, 7.44, 7.4], [6.36, 7.16, 7.68, 7.56, 7.4, 7.4, 7.36, 7.36, 7.28, 7.36, 7.36, 7.4, 7.36, 7.4, 7.4, 7.4, 7.36, 7.4, 7.36, 7.4]], "NC": [[5.92, 5.92, 5.92, 5.84, 5.92, 5.92, 5.84, 5.8, 5.72, 5.76, 5.68, 5.68, 5.68, 5.68, 5.68, 5.68, 5.68, 5.68, 5.68, 5.68], [7.48, 7.12, 7.12, 6.76, 6.48, 6.44, 6.44, 6.36, 6.36, 6.28, 6.32, 6.2, 6.24, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.24]], "NN": [[5.72, 5.76, 5.88, 5.96, 6.08, 6.08, 6.08, 6.04, 6.08, 6.08, 6.12, 6.12, 6.12, 6.12, 6.12, 6.12, 6.12, 6.12, 6.12, 6.12], [6.4, 6.32, 6.28, 6.4, 6.24, 6.36, 6.28, 6.32, 6.28, 6.28, 6.28, 6.32, 6.28, 6.32, 6.28, 6.32, 6.28, 6.32, 6.32, 6.32]], "NS": [[6.0, 6.2, 6.32, 6.36, 6.44, 6.44, 6.4, 6.36, 6.48, 6.28, 6.32, 6.32, 6.28, 6.24, 6.2, 6.2, 6.24, 6.24, 6.24, 6.24], [6.52, 7.16, 7.04, 7.16, 6.88, 6.64, 6.48, 6.28, 6.16, 6.28, 6.16, 6.2, 6.16, 6.16, 6.24, 6.32, 6.32, 6.32, 6.32, 6.32]], "SC": [[5.88, 6.0, 5.88, 5.92, 5.8, 5.76, 5.68, 5.6, 5.56, 5.6, 5.64, 5.64, 5.6, 5.6, 5.64, 5.6, 5.6, 5.64, 5.6, 5.6], [7.12, 7.32, 7.16, 6.6, 6.6, 6.36, 6.4, 6.24, 6.16, 6.2, 6.12, 6.16, 6.16, 6.12, 6.12, 6.16, 6.12, 6.12, 6.16, 6.12]], "SN": [[5.6, 5.76, 5.84, 5.76, 5.72, 5.64, 5.6, 5.6, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56, 5.56], [6.44, 6.2, 6.16, 6.08, 6.08, 5.96, 5.88, 5.84, 5.8, 5.76, 5.76, 5.76, 5.76, 5.76, 5.76, 5.84, 5.76, 5.76, 5.76, 5.76]], "SS": [[5.6, 5.76, 5.76, 5.88, 5.84, 5.8, 5.56, 5.48, 5.48, 5.32, 5.24, 5.28, 5.2, 5.2, 5.08, 5.04, 5.04, 5.04, 5.04, 4.84], [6.36, 6.48, 6.56, 6.16, 6.04, 5.8, 5.6, 5.48, 5.44, 5.08, 5.28, 4.92, 4.96, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92, 4.92]]}

basic_means={"CC": [[6.36, 6.6, 6.52, 6.68, 6.8, 6.76, 6.76, 6.76, 6.8, 6.72, 6.76, 6.72, 6.68, 6.64, 6.64, 6.64, 6.64, 6.64, 6.64, 6.64], [6.92, 6.96, 6.96, 6.96, 6.88, 7.0, 6.92, 6.92, 6.92, 7.04, 6.84, 6.84, 6.84, 6.84, 6.8, 6.76, 6.76, 6.76, 6.76, 6.76]], "CN": [[6.0, 6.16, 6.24, 6.32, 6.32, 6.32, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36, 6.36], [6.16, 6.48, 6.6, 6.56, 6.68, 6.68, 6.6, 6.6, 6.64, 6.6, 6.56, 6.6, 6.6, 6.56, 6.56, 6.56, 6.56, 6.52, 6.52, 6.48]], "CS": [[5.96, 6.0, 5.96, 5.96, 5.96, 5.92, 5.88, 5.76, 5.84, 5.72, 5.72, 5.68, 5.68, 5.68, 5.64, 5.64, 5.68, 5.56, 5.4, 5.12], [5.32, 5.84, 5.92, 6.12, 6.04, 5.96, 5.84, 6.0, 5.8, 5.84, 5.72, 5.64, 5.68, 5.64, 5.6, 5.44, 5.36, 4.96, 4.72, 4.16]], "NC": [[5.4, 5.6, 5.52, 5.64, 5.48, 5.48, 5.24, 5.2, 5.0, 4.96, 4.88, 4.88, 4.92, 4.96, 4.96, 4.96, 5.0, 5.04, 5.08, 5.12], [6.8, 6.44, 6.4, 6.32, 6.32, 5.92, 6.08, 5.68, 5.8, 5.44, 5.48, 5.4, 5.36, 5.36, 5.44, 5.4, 5.44, 5.48, 5.52, 5.56]], "NN": [[5.48, 5.72, 5.72, 5.72, 5.8, 5.88, 5.84, 5.84, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.76, 5.72], [6.08, 6.08, 6.2, 6.04, 6.16, 6.08, 6.2, 6.12, 6.08, 6.04, 6.08, 6.04, 6.04, 6.04, 6.04, 6.04, 6.04, 6.0, 5.96, 5.88]], "NS": [[5.76, 5.72, 5.88, 5.84, 5.8, 5.6, 5.56, 5.56, 5.44, 5.32, 5.36, 5.36, 5.36, 5.4, 5.36, 5.36, 5.4, 5.36, 5.36, 5.2], [5.28, 5.88, 5.92, 6.2, 6.04, 5.88, 5.84, 5.68, 5.68, 5.52, 5.36, 5.44, 5.44, 5.4, 5.44, 5.4, 5.36, 5.36, 5.12, 4.88]], "SC": [[5.24, 5.28, 5.08, 5.08, 4.68, 4.48, 4.28, 4.16, 4.04, 4.0, 3.88, 4.04, 3.8, 3.76, 3.76, 3.68, 3.64, 3.64, 3.64, 3.56], [6.88, 6.36, 6.12, 5.88, 5.72, 5.28, 5.12, 4.64, 4.6, 4.48, 4.4, 4.24, 4.44, 4.12, 4.08, 4.04, 4.0, 3.96, 3.92, 3.88]], "SN": [[5.44, 5.48, 5.44, 5.36, 5.24, 5.24, 5.2, 5.0, 4.92, 4.92, 4.88, 4.92, 4.88, 4.84, 4.84, 4.84, 4.84, 4.8, 4.88, 4.56], [6.2, 6.16, 6.16, 6.0, 5.8, 5.64, 5.68, 5.52, 5.36, 5.28, 5.36, 5.16, 5.32, 5.16, 5.08, 5.12, 5.12, 5.08, 4.96, 4.96]], "SS": [[5.08, 5.28, 5.32, 5.44, 5.16, 5.08, 5.0, 5.04, 4.96, 5.04, 4.92, 4.92, 4.88, 4.88, 4.84, 4.84, 4.92, 4.92, 4.8, 4.64], [5.68, 5.72, 5.76, 5.6, 5.6, 5.32, 5.36, 5.24, 5.32, 5.16, 5.24, 5.12, 5.2, 5.08, 5.16, 5.04, 4.84, 4.84, 4.68, 3.84]]}

discrim_final={
    "CC": [293.72800000000007, 284.80800000000005], 
    "CN": [292.73600000000005, 294.29600000000005], 
    "CS": [287.976, 288.376], 
    "NC": [280.16, 266.48], 
    "NN": [276.768, 271.528], 
    "NS": [278.136, 274.81600000000003], 
    "SC": [279.24800000000005, 265.56800000000004], 
    "SN": [271.856, 266.216], 
    "SS": [265.38400000000007, 264.26400000000007]
}

basic_final={
    "CC": [283.2720000000001, 279.2320000000001], 
    "CN": [279.616, 274.89599999999996], 
    "CS": [266.328, 269.488], 
    "NC": [271.84799999999996, 259.52799999999996], 
    "NN": [273.896, 268.13599999999997], 
    "NS": [266.896, 265.776], 
    "SC": [260.184, 247.74400000000003], 
    "SN": [265.576, 258.592], 
    "SS": [263.048, 259.20799999999997]
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
    a_sum/=25
    b_sum/=25
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
    a_round_avg[:] = [x / 25 for x in a_round_avg]
    b_round_avg[:] = [x / 25 for x in b_round_avg]
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
                if directory=="basic_llama_qwen_results":
                    a_result=(round_data["a_contribution"]-basic_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-basic_means[pair][1][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
                else:
                    a_result=(round_data["a_contribution"]-discrim_means[pair][0][index-1])**2
                    b_result=(round_data["b_contribution"]-discrim_means[pair][1][index-1])**2
                    a_err[index-1]+=a_result
                    b_err[index-1]+=b_result
        a_err[:] = [((x / 25) ** 0.5) * (1.960/5) for x in a_err]
        b_err[:] = [((x / 25) ** 0.5) * (1.960/5) for x in b_err]
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
                     if directory=="basic_llama_qwen_results":
                        a_result=(round_data["a_total_points_after_round"]-basic_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-basic_final[pair][1])**2
                        a_fin+=a_result
                        b_fin+=b_result
                     else:
                        a_result=(round_data["a_total_points_after_round"]-discrim_final[pair][0])**2
                        b_result=(round_data["b_total_points_after_round"]-discrim_final[pair][1])**2
                        a_fin+=a_result
                        b_fin+=b_result
        a_fin = ((a_fin / 25) ** 0.5) * (1.960/5)
        b_fin = ((b_fin / 25) ** 0.5) * (1.960/5)
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
        if (directory_name == "basic_llama_qwen_results"):
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
        if (directory_name == "basic_llama_qwen_results"):
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
        if (directory_name == "basic_llama_qwen_results"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

if __name__ == "__main__":
    run_error("basic_llama_qwen_results")
    with open("basic_lq_round_SE.json", 'w') as b:
        json.dump(basic_round_SE, b)
    run_error_final("basic_llama_qwen_results")
    with open("basic_lq_final_SE.json", 'w') as b:
        json.dump(basic_final_SE, b)
    run_error("self_llama_qwen_results")
    with open("self_lq_rounds_SE.json", 'w') as s:
        json.dump(discrim_round_SE, s)
    run_error_final("self_llama_qwen_results")
    with open("self_lq_final_SE.json", 'w') as b:
        json.dump(discrim_final_SE, b)