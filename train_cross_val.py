import os
import sys

import wandb

from src.utils import parse_arguments_train, set_seed, configure_model
from src.models import VideoSumarizer

PROJECT_WANDB = "sports_video_summarization"

def parse_arguments(args, master_arg):
    try:
        if(master_arg in args and args.index(master_arg)+1<len(args)):
            json_file  = args[args.index(master_arg)+1]
#            json_params = json.loads(open(json_file,"r").read())
            #json_params = eval(open(json_file,"r").read())
            return json_file
    except:
        print("Something went wrong when loading the parameters, Kindly check input carefully!!!")

def sweep_config(args, master_arg):
    if(master_arg in args):
        return True
    else:
        return False

def train(config_file, use_wandb=True, run_name=None, run_notes=None, pretrained_model=None):
    set_seed(12345)
    config = configure_model(config_file, use_wandb)
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, 
                    name=run_name, notes=run_notes)
        config = wandb.config
        wandb.run_name = wandb.run_name if run_name is not None else f'Exp. {config["feature_1"]} - {config["feature_2"]}'
        wandb.run_notes = wandb.run_notes if run_notes is not None else f'Exp. {config["type_dataset"]} - {config["type_setting"]}'
        wandb.run.save()
        wandb.watch_called = False

    vsm = VideoSumarizer(config, use_wandb)
    vsm.train_cross_validation(pretrained_model)

if __name__ == '__main__':
    use_sweep = sweep_config(sys.argv, '--use_sweep')
    if not use_sweep:
        args = parse_arguments_train()
        use_wandb = args.wandb
        run_name = args.run_name
        run_notes = args.run_notes
        config_file = args.params
        pretrained_model = args.pretrained_model
    else:
        use_wandb = True
        run_name = None
        run_notes = None
        config_file = parse_arguments(sys.argv, '--params')
        pretrained_model = parse_arguments(sys.argv, '--pretrained_model') if sweep_config(sys.argv, '--pretrained_model') else None

    train(config_file=config_file, use_wandb=use_wandb, run_name=run_name, 
            run_notes=run_notes, pretrained_model=pretrained_model)