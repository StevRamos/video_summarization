import argparse
import random
import os
import sys
import json

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

def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file

def parse_arguments_train():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-nw  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-n', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-nt', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")
    ap.add_argument('-p', '--params', required=True, type=str, default=None,
                    help="path of json file")

    args = ap.parse_args()

    return args

def configure_model(config_file, use_wandb):
    config_file = parse_configuration(config_file)
    config = dict(
        learning_rate = config_file["hparams"]["learning_rate"],
        weight_decay = config_file["hparams"]["weight_decay"],
        epochs_max = config_file["hparams"]["epochs_max"],
        feature_1 = config_file["hparams"]["feature_1"],
        feature_2 = config_file["hparams"]["feature_2"],
        type_dataset = config_file["hparams"]["type_dataset"],
        type_setting = config_file["hparams"]["type_setting"],
        sameAccStopThres = config_file["hparams"]["sameAccStopThres"],

        path_tvsum = config_file["datasets"]["path_tvsum"],
        path_summe = config_file["datasets"]["path_summe"],
        path_ovp = config_file["datasets"]["path_ovp"],
        path_youtube = config_file["datasets"]["path_youtube"],
        path_cosum = config_file["datasets"]["path_cosum"],

        path_split_summe_canonical = config_file["splits"]["path_split_summe_canonical"],
        path_split_tvsum_canonical = config_file["splits"]["path_split_tvsum_canonical"],
        path_split_summe_aug = config_file["splits"]["path_split_summe_aug"],
        path_split_tvsum_aug = config_file["splits"]["path_split_tvsum_aug"],
        path_split_summe_non_overlap_ord = config_file["splits"]["path_split_summe_non_overlap_ord"],
        path_split_summe_non_overlap_rand = config_file["splits"]["path_split_summe_non_overlap_rand"],
        path_split_tvsum_non_overlap_ord = config_file["splits"]["path_split_tvsum_non_overlap_ord"],
        path_split_tvsum_non_overlap_rand = config_file["splits"]["path_split_tvsum_non_overlap_rand"],

        save_weights = config_file["save_weights"],
        num_backups = config_file["num_backups"],
        path_saved_weights = config_file["path_saved_weights"]
    )

    if not use_wandb:
        config = type("configuration", (object,), config)

    return config