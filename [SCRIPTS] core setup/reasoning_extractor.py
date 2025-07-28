import os
import json

def extract(directory, subdirectory):
    base_dir="."
    path=os.path.join(base_dir, directory, subdirectory)
    files=os.listdir(path)
    a_reasoning={i: "" for i in range(1, 21)}
    for name in files:
        file_path=os.path.join(path, name)
        with open(file_path) as file:
            data=json.load(file)
            for reasoning in data:
                key=reasoning["round"]
                a=reasoning["a_reasoning"]+"\n\n"
                b=reasoning["b_reasoning"]+"\n\n"
                a_reasoning[key]+=a
                a_reasoning[key]+=b
    with open(f'{directory}_{subdirectory}.json', 'w') as f:
        json.dump(a_reasoning, f)

if __name__=="__main__":
    prompt_pairs=["CC", "CN", "CS", "NC", "NN", "NS", "SC", "SN", "SS"]
    for pair in prompt_pairs:
        extract("self_llama4_qwen", pair)
        extract("llama4_qwen", pair)