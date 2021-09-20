import torch
import torchvision


class VideoSumarizer():

    def __init__(self, config, use_wandb):
        self.use_wandb = use_wandb
        self.config = config
        self.device = torch.device("cuda:" + (os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0") if self.config.use_gpu and torch.cuda.is_available() else "cpu")
