import os
import json
import numpy as np

discrim_means={
    "CC": [[6.73, 6.79, 6.79, 6.81, 6.8, 6.82, 6.81, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.82, 6.96], 
            [9.52, 9.55, 9.45, 9.45, 9.51, 9.49, 9.51, 9.51, 9.52, 9.53, 9.52, 9.52, 9.52, 9.52, 9.52, 9.52, 9.52, 9.52, 9.52, 9.52]], 
    
    "CN": [[6.73, 6.81, 6.78, 6.83, 6.82, 6.86, 6.88, 6.87, 6.87, 6.88, 6.87, 6.87, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.89, 6.92], 
            [6.56, 7.32, 7.53, 7.34, 7.3, 7.31, 7.32, 7.35, 7.33, 7.35, 7.35, 7.33, 7.33, 7.34, 7.34, 7.34, 7.23, 7.18, 7.07, 7.01]], 
    
    "CS": [[6.63, 6.7, 6.66, 6.66, 6.59, 6.57, 6.54, 6.53, 6.49, 6.46, 6.45, 6.45, 6.46, 6.45, 6.45, 6.45, 6.46, 6.46, 6.46, 6.24], 
            [4.69, 5.73, 5.91, 5.91, 5.82, 5.78, 5.74, 5.73, 5.74, 5.72, 5.71, 5.65, 5.69, 5.7, 5.69, 5.56, 5.34, 4.86, 4.38, 3.67]], 
            
    "NC": [[5.53, 5.68, 5.71, 5.73, 5.72, 5.72, 5.71, 5.72, 5.72, 5.72, 5.72, 5.72, 5.72, 5.72, 5.72, 5.72, 5.73, 5.73, 5.73, 5.74], 
            [9.48, 9.33, 9.31, 9.31, 9.31, 9.29, 9.29, 9.29, 9.29, 9.29, 9.29, 9.29, 9.29, 9.29, 9.29, 9.27, 9.25, 9.25, 9.25, 9.23]], 
            
    "NN": [[5.71, 5.91, 5.84, 5.87, 5.84, 5.8, 5.81, 5.79, 5.79, 5.79, 5.8, 5.79, 5.79, 5.79, 5.79, 5.79, 5.79, 5.78, 5.8, 5.88], 
            [6.53, 6.7, 6.86, 6.72, 6.54, 6.44, 6.44, 6.41, 6.38, 6.4, 6.39, 6.39, 6.38, 6.38, 6.37, 6.32, 6.29, 6.18, 6.07, 5.83]], 
            
    "NS": [[5.42, 5.58, 5.57, 5.57, 5.5, 5.46, 5.41, 5.37, 5.31, 5.29, 5.2, 5.21, 5.2, 5.2, 5.19, 5.18, 5.17, 5.17, 5.25, 4.89], 
            [5.1, 5.2, 5.24, 5.04, 4.97, 4.78, 4.78, 4.71, 4.71, 4.63, 4.6, 4.66, 4.66, 4.63, 4.58, 4.5, 4.24, 3.9, 3.57, 2.84]], 
            
    "SC": [[4.11, 4.29, 4.29, 4.29, 4.28, 4.29, 4.27, 4.27, 4.27, 4.27, 4.25, 4.23, 4.22, 4.21, 4.2, 4.21, 4.2, 4.2, 4.18, 3.45], 
            [9.58, 9.24, 9.12, 8.98, 8.9, 8.87, 8.81, 8.85, 8.81, 8.79, 8.75, 8.74, 8.71, 8.7, 8.67, 8.63, 8.63, 8.63, 8.63, 8.63]], 
            
    "SN": [[4.3, 4.53, 4.41, 4.29, 4.13, 4.01, 3.85, 3.82, 3.78, 3.77, 3.77, 3.77, 3.76, 3.76, 3.76, 3.76, 3.75, 3.76, 3.73, 2.69], 
            [6.36, 5.85, 5.62, 5.36, 5.16, 4.89, 4.72, 4.62, 4.56, 4.58, 4.5, 4.5, 4.48, 4.47, 4.45, 4.42, 4.39, 4.32, 4.24, 4.14]], 
            
    "SS": [[4.5, 4.5, 4.28, 4.16, 3.9, 3.73, 3.55, 3.52, 3.48, 3.48, 3.5, 3.5, 3.48, 3.46, 3.44, 3.42, 3.4, 3.29, 3.18, 2.18], 
            [4.9, 4.81, 4.6, 4.22, 3.95, 3.68, 3.57, 3.6, 3.61, 3.58, 3.58, 3.48, 3.49, 3.44, 3.39, 3.28, 2.96, 2.83, 2.68, 2.24]]
}

basic_means={
    "CC": [[9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07, 9.07], 
            [9.93, 9.92, 9.94, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95, 9.95]], 
    
    "CN": [[8.78, 8.78, 8.8, 8.81, 8.82, 8.81, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82, 8.82], 
            [6.76, 8.81, 8.99, 8.99, 9.0, 8.99, 8.97, 8.98, 8.98, 8.98, 8.98, 8.98, 8.98, 8.98, 8.98, 8.98, 8.98, 8.95, 8.96, 8.94]], 
            
    "CS": [[8.93, 8.75, 8.71, 8.7, 8.6, 8.62, 8.55, 8.53, 8.53, 8.42, 8.47, 8.45, 8.45, 8.45, 8.45, 8.45, 8.5, 8.5, 8.46, 8.45], 
            [2.13, 2.27, 1.97, 1.68, 1.51, 1.59, 1.47, 1.55, 1.55, 1.45, 1.65, 1.55, 1.65, 1.55, 1.65, 1.45, 1.5, 1.35, 1.4, 1.08]], 
            
    "NC": [[3.95, 4.02, 4.07, 4.07, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.15], 
            [7.26, 7.23, 7.21, 7.23, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.35, 7.35]], 
            
    "NN": [[5.22, 5.46, 5.51, 5.54, 5.47, 5.44, 5.38, 5.31, 5.27, 5.26, 5.29, 5.25, 5.25, 5.22, 5.22, 5.22, 5.22, 5.21, 5.21, 5.09], 
            [6.71, 6.41, 6.34, 6.23, 6.17, 6.04, 5.98, 5.93, 5.86, 5.88, 5.8, 5.8, 5.75, 5.76, 5.73, 5.71, 5.69, 5.66, 5.63, 5.56]], 
            
    "NS": [[2.2, 2.0, 1.83, 1.57, 1.4, 1.32, 1.26, 1.18, 1.14, 1.11, 1.09, 1.08, 1.07, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.09], 
            [1.06, 1.13, 1.01, 0.9, 0.86, 0.79, 0.74, 0.72, 0.69, 0.67, 0.66, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.58, 0.53]], 
            
    "SC": [[2.45, 3.08, 3.09, 3.16, 3.25, 3.13, 3.11, 3.08, 3.08, 3.07, 3.08, 3.07, 3.07, 3.08, 3.07, 3.07, 3.07, 3.07, 3.07, 2.74], 
            [9.83, 9.52, 9.08, 8.49, 8.46, 8.44, 8.45, 8.41, 8.39, 8.38, 8.33, 8.29, 8.31, 8.28, 8.3, 8.31, 8.29, 8.32, 8.29, 8.29]], 
            
    "SN": [[2.27, 2.65, 2.53, 2.41, 2.15, 1.82, 1.69, 1.56, 1.55, 1.5, 1.49, 1.47, 1.46, 1.47, 1.46, 1.46, 1.48, 1.46, 1.42, 1.11], 
            [6.81, 4.65, 3.64, 3.51, 3.07, 2.77, 2.29, 2.28, 2.01, 1.92, 1.87, 1.84, 1.83, 1.8, 1.81, 1.77, 1.72, 1.69, 1.67, 1.6]], 
            
    "SS": [[2.55, 1.74, 1.66, 1.01, 0.72, 0.58, 0.47, 0.41, 0.33, 0.31, 0.29, 0.28, 0.27, 0.27, 0.26, 0.25, 0.25, 0.24, 0.23, 0.18], 
            [2.23, 1.08, 0.76, 0.59, 0.41, 0.31, 0.22, 0.18, 0.18, 0.13, 0.13, 0.12, 0.09, 0.1, 0.08, 0.07, 0.06, 0.06, 0.05, 0.03]]
}

discrim_final={
    "CC": [324.922, 271.03200000000004], 
    "CN": [288.752, 280.682], 
    "CS": [261.18399999999997, 282.3239999999999], 
    "NC": [325.87000000000006, 254.19], 
    "NN": [279.18600000000004, 267.31600000000003], 
    "NS": [251.84399999999994, 266.644], 
    "SC": [324.54, 231.85000000000002], 
    "SN": [261.0240000000001, 242.79400000000012], 
    "SS": [243.122, 243.18200000000002]
}

basic_final={
    "CC": [322.872, 305.332], 
    "CN": [306.4719999999999, 305.59199999999987], 
    "CS": [191.40600000000003, 330.37600000000003], 
    "NC": [299.90200000000004, 235.78200000000004], 
    "NN": [273.70399999999995, 261.10400000000004], 
    "NS": [206.74799999999996, 217.67799999999997], 
    "SC": [324.19000000000005, 214.62], 
    "SN": [233.558, 217.418], 
    "SS": [203.044, 208.46400000000003]
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
                if directory=="llama4_qwen":
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
                     if directory=="llama4_qwen":
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
        if (directory_name == "llama4_qwen"):
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
        if (directory_name == "llama4_qwen"):
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
        if (directory_name == "llama4_qwen"):
            basic_round_SE[name] = result
        else:
            discrim_round_SE[name] = result

if __name__ == "__main__":
    run("llama4_qwen")
    with open("llama_qwen_final.json", 'w') as b:
        json.dump(basic_final_avg, b)
    with open("llama_qwen_rounds.json", 'w') as f:
        json.dump(basic_round_avg, f)
    run("self_llama4_qwen")
    with open("self_llama_qwen_final.json", 'w') as c:
        json.dump(discrim_final_avg, c)
    with open("self_llama_qwen_rounds.json", 'w') as g:
        json.dump(discrim_round_avg, g)

    run_error_final("llama4_qwen")
    with open("llama4_qwen_final_SE.json", 'w') as b:
        json.dump(basic_final_SE, b)
    run_error_final("self_llama4_qwen")
    with open("self_llama4_qwen_final_SE.json", 'w') as s:
        json.dump(discrim_final_SE, s)