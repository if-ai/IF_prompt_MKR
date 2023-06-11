import glob
import os

def get_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return f"{name}"

def get_negative(path):
    txt_files = glob.glob(os.path.join(path, "**/*.txt"), recursive=True)
    neg_prompt = [[get_name(f), f] for f in txt_files]
    neg_prompt = sorted(neg_prompt, key=lambda p: p[0])
    return neg_prompt