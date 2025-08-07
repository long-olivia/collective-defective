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
                c=reasoning["c_reasoning"]+"\n\n"
                d=reasoning["d_reasoning"]+"\n\n"
                a_reasoning[key]+=a
                a_reasoning[key]+=b
                a_reasoning[key]+=c
                a_reasoning[key]+=d
    with open(f'{directory}_{subdirectory}.json', 'w') as f:
        json.dump(a_reasoning, f)

if __name__=="__main__":
    prompt_pairs=["CCCC", "NNNN", "SSSS"]
    for pair in prompt_pairs:
        extract("anthropic/claude-sonnet-4_results", pair)
        extract("meta-llama/llama-4-maverick_results", pair)
        extract("openai/gpt-4o_results", pair)
        extract("qwen/qwen3-235b-a22b-2507_results", pair)
        extract("self_anthropic/claude-sonnet-4_results", pair)
        extract("self_meta-llama/llama-4-maverick_results", pair)
        extract("self_openai/gpt-4o_results", pair)
        extract("self_qwen/qwen3-235b-a22b-2507_results", pair)