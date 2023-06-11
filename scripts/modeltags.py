import modules.shared as shared
import modules.scripts as scripts

import csv
import os
import difflib
import random
import glob
import hashlib
import shutil
import fnmatch

from collections import defaultdict


def find_files(directory, exts):
    for root, dirs, files in os.walk(directory):
        for ext in exts:
            pattern = f'*.{ext}'
            for filename in fnmatch.filter(files, pattern):
                yield os.path.relpath(os.path.join(root, filename), directory)

def get_embeddings():
    return [os.path.basename(x) for x in glob.glob(f'{shared.cmd_opts.embeddings_dir}/*.pt')]

def get_loras():
    return sorted(list(find_files(shared.cmd_opts.lora_dir, ['safetensors', 'ckpt','pt'])), key=str.casefold)


