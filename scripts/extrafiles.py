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


def get_excluded_words_from_file(excluded_path):
    with open(excluded_path, 'r') as f:
        words = f.read().split(',')
        return [word.strip() for word in words]
    

def get_excluded_words(dynamic_excluded_words, excluded_path):
    file_excluded_words = get_excluded_words_from_file(excluded_path)

    dynamic_excluded_words = dynamic_excluded_words.split(',')

    file_excluded_words = [word.lower().strip() for word in file_excluded_words]
    dynamic_excluded_words = [word.lower().strip() for word in dynamic_excluded_words]

    excluded_words = dynamic_excluded_words + file_excluded_words

    return excluded_words





