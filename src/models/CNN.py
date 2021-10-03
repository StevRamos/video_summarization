import torch.nn as nn
import torch
from torchvision import transforms, models
from torch.autograd import Variable
from PIL import Image
from numpy import linalg
import numpy as np
from .transforms import Transform_models_cnn
"""
pre-trained ResNet
"""

class ResNet(nn.Module):
    def __init__(self, device, transform="Transform_models_cnn"):
        super(ResNet, self).__init__()
        self.preprocess = eval(transform)()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.eval()
        self.device = device

    def forward(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)

        if torch.cuda.is_available():
            batch = batch.to(self.device)
            self.model = self.model.to(self.device)
            
        with torch.no_grad():
            feat = self.model(batch)
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (2048,), f'Invalid feature shape {feat.shape}: expected 2048'
        # normalize frame features
        #feat /= linalg.norm(feat) + 1e-10
        return feat


'''
class GoogleNet(nn.Module):
    def __init__(self, device):
        super(GoogleNet, self).__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.device = device

    def forward(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)

        if torch.cuda.is_available():
            batch = batch.to(self.device)
            self.model = self.model.to(self.device)
            
        with torch.no_grad():
            feat = self.model(batch)
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        #feat /= linalg.norm(feat) + 1e-10
        return feat
'''


class GoogleNet(nn.Module):
    def __init__(self, device, transform="Transform_models_cnn"):
        super(GoogleNet, self).__init__()
        self.preprocess = eval(transform)()
        self.model = models.googlenet(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.eval()
        self.device = device

    def forward(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)

        if torch.cuda.is_available():
            batch = batch.to(self.device)
            self.model = self.model.to(self.device)
            
        with torch.no_grad():
            feat = self.model(batch)
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        #feat /= linalg.norm(feat) + 1e-10
        return feat


class Inception(nn.Module):
    def __init__(self, device, transform="Transform_models_cnn"):
        super(Inception, self).__init__()
        self.preprocess = eval(transform)(resized=299, centercrop=299)
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.eval()
        self.device = device

    def forward(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        
        if torch.cuda.is_available():
            batch = batch.to(self.device)
            self.model = self.model.to(self.device)
            
        with torch.no_grad():
            feat = self.model(batch)
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (2048,), f'Invalid feature shape {feat.shape}: expected 2048'
        # normalize frame features
        #feat /= linalg.norm(feat) + 1e-10
        return feat