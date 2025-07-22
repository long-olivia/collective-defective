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
                dict[index]=data[index].split('\n\n')                
            with open(f"new_{name}", 'w') as f:
                json.dump(dict, f)

if __name__=="__main__":
    extract("corpus")