import os
import json

def extract(directory, subdirectory):
    all=[]
    base_dir="."
    path=os.path.join(base_dir, directory, subdirectory)
    files=os.listdir(path)
    for name in files:
        file_path=os.path.join(path, name)
        with open(file_path) as file:
            data=json.load(file)
            for reasoning in data:
                a=[
                    {"text": reasoning["a_reasoning"],
                     "cooperative_score": 0,
                     "round": reasoning["round"]
                    }
                ]
                b=[
                    {"text": reasoning["b_reasoning"],
                     "cooperative_score": 0,
                     "round": reasoning["round"]
                    }
                ]
                all+=a
                all+=b
    with open(f'{directory}_{subdirectory}_reasoning.json', 'w') as f:
        json.dump(all, f)

if __name__=="__main__":
    prompt_pairs=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    for pair in prompt_pairs:
        extract("self_results", pair)
        extract("basic_results", pair)