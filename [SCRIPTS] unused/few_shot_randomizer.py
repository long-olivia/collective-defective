from random import sample
import os
import json

def extract(directory):
    base_dir="."
    path=os.path.join(base_dir, directory)
    files=os.listdir(path)
    for name in files:
        file_path=os.path.join(path, name)
        with open(file_path, 'r') as file:
            data=json.load(file)
            dict={}
            for i in range(1,21):
                index=f"{i}"
                dict[index]=sample(data[index],3)
            with open(f"random_{name}", 'w') as f:
                json.dump(dict, f)

if __name__=="__main__":
    extract("corpus")