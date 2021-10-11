import os
import sys

import wandb

from utils import parse_arguments_train, set_seed, configure_model
from src.models import VideoSumarizer

PROJECT_WANDB = "sports_video_summarization"

def train(config_file, use_wandb, run_name, run_notes):
    set_seed(12345)
    config = configure_model(config_file, use_wandb)
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, 
                    name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False

    vsm = VideoSumarizer(config, use_wandb)
    vsm.train()

if __name__ == '__main__':
    args = parse_arguments_train()
    use_wandb = args.wandb
    run_name = args.run_name
    run_notes = args.run_notes
    config_file = args.params

    train(config_file=config_file, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)