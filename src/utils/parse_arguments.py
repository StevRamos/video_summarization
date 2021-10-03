import argparse
import random
import os
import sys

import torch
import numpy as np

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def parse_arguments_generate_dataset():
    ap = argparse.ArgumentParser()
    ap.add_argument('-vp', '--videospath', required=True, type=str, 
                    help="path where videos are located")
    ap.add_argument('-gtp', '--groundtruthpath', required=True, type=str,
                    help="path where ground truth annotations are located")
    ap.add_argument('-ds', '--dataset', required=True, type=str,
                    help="dataset name: summe, tvsum, youtube, ovp or cosum")

    args = ap.parse_args()

    return args