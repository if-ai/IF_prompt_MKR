import modules.shared as shared

import os
import glob
import fnmatch
import requests
import json
import time
import hashlib
import io
import re


info_ext = ".info"
suffix = ".civitai"
def_headers = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
proxies = None

root_path = os.getcwd()

def print_msg(msg):
    print(f"iF_prompt_MKR: {msg}")

def get_embeddings_dir():
    if shared.cmd_opts.embeddings_dir and os.path.isdir(shared.cmd_opts.embeddings_dir):
        return shared.cmd_opts.embeddings_dir
    return os.path.join(root_path, "embeddings")

embeddings_dir = get_embeddings_dir()                  

def get_loras_dir():
    if shared.cmd_opts.lora_dir and os.path.isdir(shared.cmd_opts.lora_dir):
        return shared.cmd_opts.lora_dir
    return os.path.join(root_path, "models", "Lora")

loras_dir = get_loras_dir()


def read_chunks(file, size=io.DEFAULT_BUFFER_SIZE):
    while True:
        chunk = file.read(size)
        if not chunk:
            break
        yield chunk


def gen_file_sha256(filname):
    print_msg("Use Memory Optimized SHA256")
    blocksize=1 << 20
    h = hashlib.sha256()
    length = 0
    with open(os.path.realpath(filname), 'rb') as f:
        for block in read_chunks(f, size=blocksize):
            length += len(block)
            h.update(block)

    hash_value =  h.hexdigest()
    print_msg("sha256: " + hash_value)
    print_msg("length: " + str(length))
    return hash_value


def find_files(directory, exts):
    for root, dirs, files in os.walk(directory):
        for ext in exts:
            pattern = f'*.{ext}'
            for filename in fnmatch.filter(files, pattern):
                yield os.path.relpath(os.path.join(root, filename), directory)


def get_loras():
    return sorted(list(find_files(loras_dir, ['safetensors', 'ckpt','pt'])), key=str.casefold)


def get_embeddings():
    return [os.path.basename(x) for x in glob.glob(f'{embeddings_dir}/*.pt')]


def write_model_info(path, model_info):
    print_msg("Write model info to file: " + path)
    with open(os.path.realpath(path), 'w') as f:
        f.write(json.dumps(model_info, indent=4))


def load_model_info(path):
    print_msg("Load model info from file: " + path)
    model_info = None
    with open(os.path.realpath(path), 'r') as f:
        try:
            model_info = json.load(f)
        except Exception as e:
            print_msg("Selected file is not json: " + path)
            print_msg(e)
            return
        
    return model_info


def get_model_info_by_hash(hash:str):
    print_msg("Request model info from civitai")

    if not hash:
        print_msg("hash is empty")
        return

    r = requests.get("https://civitai.com/api/v1/model-versions/by-hash/"+hash, headers=def_headers, proxies=proxies)
    if not r.ok:
        if r.status_code == 404:
            # this is not a civitai model
            print_msg("Civitai does not have this model")
            return {}
        else:
            print_msg("Get error code: " + str(r.status_code))
            print_msg(r.text)
            return

    # try to get content
    content = None
    try:
        content = r.json()
    except Exception as e:
        print_msg("Parse response json failed")
        print_msg(str(e))
        print_msg("response:")
        print_msg(r.text)
        return
    
    if not content:
        print_msg("error, content from civitai is None")
        return
    
    return content


def load_model_info_by_lora_name(lora_name):
    print_msg(f"Load model info of {lora_name} from file: {loras_dir}")
    
    base, ext = os.path.splitext(lora_name)
    model_info_base = base
    if base[:1] == "/":
        model_info_base = base[1:]

    model_folder = loras_dir
    model_info_filename = model_info_base + suffix + info_ext
    model_info_filepath = os.path.join(model_folder, model_info_filename)

    if not os.path.isfile(model_info_filepath):
        print_msg("Can not find model info file: " + model_info_filepath)
        return
    
    return load_model_info(model_info_filepath)


def get_lora_trigger_words(lora_name):
    model_info = load_model_info_by_lora_name(lora_name)
    if not model_info or not model_info.get("trainedWords"):
        lora_rename, ext = os.path.splitext(lora_name)
        print_msg(f"{lora_rename}")
        return f'{lora_rename}'
    
    trigger_words = ", ".join(model_info["trainedWords"])
    print_msg("trigger_words: " + trigger_words)

    return trigger_words


def load_model_info_by_embedding_name(ti_name):
    print_msg(f"Load model info of {ti_name} from file: {embeddings_dir}")
    
   
    base, ext = os.path.splitext(ti_name)
    model_info_base = base
    if base[:1] == "/":
        model_info_base = base[1:]

    model_folder = embeddings_dir
    model_info_filename = model_info_base + suffix + info_ext
    model_info_filepath = os.path.join(model_folder, model_info_filename)

    if not os.path.isfile(model_info_filepath):
        print_msg("Can not find model info file: " + model_info_filepath)
        return
    
    return load_model_info(model_info_filepath)


def get_embedding_trigger_words(ti_name):
    model_info = load_model_info_by_embedding_name(ti_name)
    if not model_info or not model_info.get("trainedWords"):
        print_msg(f"No info found for {ti_name}")
        return 'trainedWords is empty or not exist in model info file'
    
    trigger_words = ", ".join(model_info["trainedWords"])
    print_msg("trigger_words: " + trigger_words)

    return trigger_words



def get_trigger_files():
    print_msg("Getting tags")
    output = ""
    loras_count = 0
    embeddings_count = 0

    # Get all lora files
    lora_files = get_loras()
    loras_dir = get_loras_dir()  
    
    for lora_file in lora_files:
        lora_file_path = os.path.join(loras_dir, lora_file)
        base, ext = os.path.splitext(lora_file_path)
        info_file = base + suffix + info_ext

        # Check if info file exists
        if os.path.isfile(info_file):
            continue

        print_msg("Creating model info for: " + lora_file)
        hash = gen_file_sha256(lora_file_path)

        
        if not hash:
            print_msg("Failed generating SHA256 for model:" + lora_file)
            continue
        model_info = get_model_info_by_hash(hash)

        if model_info is None:
            print_msg(f"Failed to connect to civitai for model: {lora_file}")
            continue

        write_model_info(info_file, model_info)
        loras_count += 1

    # Get all embedding files
    embedding_files = get_embeddings()
    embeddings_dir = get_embeddings_dir()
    
    
    for embedding_file in embedding_files:
        embedding_file_path = os.path.join(embeddings_dir, embedding_file)
        base, ext = os.path.splitext(embedding_file_path)
        info_file = base + suffix + info_ext

        # Check if info file exists
        if os.path.isfile(info_file):
            continue

        print_msg("Creating model info for: " + embedding_file)
        hash = gen_file_sha256(embedding_file_path)

        if not hash:
            print_msg("Failed generating SHA256 for model:" + embedding_file)
            continue

        model_info = get_model_info_by_hash(hash)

        time.sleep(1)

        if model_info is None:
            print_msg(f"Failed to connect to civitai for model: {embedding_file}")
            continue

        write_model_info(info_file, model_info)
        embeddings_count += 1

        time.sleep(1)

    model_count = embeddings_count + loras_count

    output = f"Done. Triggers were created for {model_count} models"
    print_msg(output)

    return output
import modules.shared as shared
import os
import glob
import fnmatch
import requests
import json
import time
import hashlib
import io
import re


info_ext = ".info"
suffix = ".civitai"
embeddings_dir = shared.cmd_opts.embeddings_dir
loras_dir = shared.cmd_opts.lora_dir
def_headers = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}


proxies = None

def print_msg(msg):
    print(f"iF_prompt_MKR: {msg}")


def read_chunks(file, size=io.DEFAULT_BUFFER_SIZE):
    while True:
        chunk = file.read(size)
        if not chunk:
            break
        yield chunk


def gen_file_sha256(filname):
    print_msg("Use Memory Optimized SHA256")
    blocksize=1 << 20
    h = hashlib.sha256()
    length = 0
    with open(os.path.realpath(filname), 'rb') as f:
        for block in read_chunks(f, size=blocksize):
            length += len(block)
            h.update(block)

    hash_value =  h.hexdigest()
    print_msg("sha256: " + hash_value)
    print_msg("length: " + str(length))
    return hash_value


def find_files(directory, exts):
    for root, dirs, files in os.walk(directory):
        for ext in exts:
            pattern = f'*.{ext}'
            for filename in fnmatch.filter(files, pattern):
                yield os.path.relpath(os.path.join(root, filename), directory)


def get_loras():
    return sorted(list(find_files(shared.cmd_opts.lora_dir, ['safetensors', 'ckpt','pt'])), key=str.casefold)


def get_embeddings():
    return [os.path.basename(x) for x in glob.glob(f'{shared.cmd_opts.embeddings_dir}/*.pt')]


def write_model_info(path, model_info):
    print_msg("Write model info to file: " + path)
    with open(os.path.realpath(path), 'w') as f:
        f.write(json.dumps(model_info, indent=4))


def load_model_info(path):
    print_msg("Load model info from file: " + path)
    model_info = None
    with open(os.path.realpath(path), 'r') as f:
        try:
            model_info = json.load(f)
        except Exception as e:
            print_msg("Selected file is not json: " + path)
            print_msg(e)
            return
        
    return model_info


def get_model_info_by_hash(hash:str):
    print_msg("Request model info from civitai")

    if not hash:
        print_msg("hash is empty")
        return

    r = requests.get("https://civitai.com/api/v1/model-versions/by-hash/"+hash, headers=def_headers, proxies=proxies)
    if not r.ok:
        if r.status_code == 404:
            # this is not a civitai model
            print_msg("Civitai does not have this model")
            return {}
        else:
            print_msg("Get error code: " + str(r.status_code))
            print_msg(r.text)
            return

    # try to get content
    content = None
    try:
        content = r.json()
    except Exception as e:
        print_msg("Parse response json failed")
        print_msg(str(e))
        print_msg("response:")
        print_msg(r.text)
        return
    
    if not content:
        print_msg("error, content from civitai is None")
        return
    
    return content


def load_model_info_by_lora_name(lora_name):
    print_msg(f"Load model info of {lora_name} from file: {loras_dir}")
    
    base, ext = os.path.splitext(lora_name)
    model_info_base = base
    if base[:1] == "/":
        model_info_base = base[1:]

    model_folder = loras_dir
    model_info_filename = model_info_base + suffix + info_ext
    model_info_filepath = os.path.join(model_folder, model_info_filename)

    if not os.path.isfile(model_info_filepath):
        print_msg("Can not find model info file: " + model_info_filepath)
        return
    
    return load_model_info(model_info_filepath)


def get_lora_trigger_words(lora_name):
    model_info = load_model_info_by_lora_name(lora_name)
    if not model_info or not model_info.get("trainedWords"):
        lora_rename, ext = os.path.splitext(lora_name)
        print_msg(f"{lora_rename}")
        return f'{lora_rename}'
    
    trigger_words = ", ".join(model_info["trainedWords"])
    print_msg("trigger_words: " + trigger_words)

    return trigger_words


def load_model_info_by_embedding_name(ti_name):
    print_msg(f"Load model info of {ti_name} from file: {embeddings_dir}")
    
    # ti_name = subfolderpath + model name + ext. And it always start with a / even there is no sub folder
    base, ext = os.path.splitext(ti_name)
    model_info_base = base
    if base[:1] == "/":
        model_info_base = base[1:]

    model_folder = embeddings_dir
    model_info_filename = model_info_base + suffix + info_ext
    model_info_filepath = os.path.join(model_folder, model_info_filename)

    if not os.path.isfile(model_info_filepath):
        print_msg("Can not find model info file: " + model_info_filepath)
        return
    
    return load_model_info(model_info_filepath)


def get_embedding_trigger_words(ti_name):
    model_info = load_model_info_by_embedding_name(ti_name)
    if not model_info or not model_info.get("trainedWords"):
        print_msg(f"No info found for {ti_name}")
        return 'trainedWords is empty or not exist in model info file'
    
    trigger_words = ", ".join(model_info["trainedWords"])
    print_msg("trigger_words: " + trigger_words)

    return trigger_words



def get_trigger_files():
    print_msg("Getting tags")
    output = ""
    loras_count = 0
    embeddings_count = 0

    # Get all lora files
    lora_files = get_loras()
    loras_dir = shared.cmd_opts.lora_dir  # Assuming shared.cmd_opts.lora_dir is defined
    
    for lora_file in lora_files:
        lora_file_path = os.path.join(loras_dir, lora_file)
        base, ext = os.path.splitext(lora_file_path)
        info_file = base + suffix + info_ext

        # Check if info file exists
        if os.path.isfile(info_file):
            continue

        print_msg("Creating model info for: " + lora_file)
        hash = gen_file_sha256(lora_file_path)

        
        if not hash:
            print_msg("Failed generating SHA256 for model:" + lora_file)
            continue
        model_info = get_model_info_by_hash(hash)

        if model_info is None:
            print_msg(f"Failed to connect to civitai for model: {lora_file}")
            continue

        write_model_info(info_file, model_info)
        loras_count += 1

    # Get all embedding files
    embedding_files = get_embeddings()
    embeddings_dir = shared.cmd_opts.embeddings_dir  # Assuming shared.cmd_opts.embeddings_dir is defined
    
    for embedding_file in embedding_files:
        embedding_file_path = os.path.join(embeddings_dir, embedding_file)
        base, ext = os.path.splitext(embedding_file_path)
        info_file = base + suffix + info_ext

        # Check if info file exists
        if os.path.isfile(info_file):
            continue

        print_msg("Creating model info for: " + embedding_file)
        hash = gen_file_sha256(embedding_file_path)

        if not hash:
            print_msg("Failed generating SHA256 for model:" + embedding_file)
            continue

        model_info = get_model_info_by_hash(hash)

        time.sleep(1)

        if model_info is None:
            print_msg(f"Failed to connect to civitai for model: {embedding_file}")
            continue

        write_model_info(info_file, model_info)
        embeddings_count += 1

        time.sleep(1)

    model_count = embeddings_count + loras_count

    output = f"Done. Triggers were created for {model_count} models"
    print_msg(output)

    return output

#might add an individual model trigger file generator later

'''def get_ti_trigger_file(ti_name):
    print_msg("Getting tags")
    output = ""
    base, ext = os.path.splitext(ti_name)
    info_file = base + suffix + info_ext
    # check info file
    if not os.path.isfile(info_file):
        print_msg("Creating model info for: " + ti_name)
        # get model's sha256
        hash = gen_file_sha256(ti_name)

        if not hash:
            output = "failed generating SHA256 for model:" + ti_name
            print_msg(output)
            return output
                        
        # get model info from civitai
        model_info = get_model_info_by_hash(hash)
        
        time.sleep(1)

        if model_info is None:
            output = "Connect to Civitai API service failed. Wait a while and try again"
            print_msg(output)
            return output+", check console log for detail"

        # write model info to file
        write_model_info(info_file, model_info)



    # scan_log = "Done"

    output = f"Trigger_words: {get_embedding_trigger_words(ti_name)}"

    print_msg(output)'''
