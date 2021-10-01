import cv2
import torch.nn as nn
import torch
from torchvision import transforms, models
from torch.autograd import Variable
from PIL import Image
from numpy import linalg
import numpy as np
from .pytorch_i3d import InceptionI3d
from .transforms import CenterCrop

class I3D(nn.Module):
    def __init__(self, device, path_weights_flow=None, path_weights_rgb=None):
        super(I3D, self).__init__()
        self.model_flow = InceptionI3d(400, in_channels=2)
        self.model_rgb = InceptionI3d(400, in_channels=3)

        if path_weights_flow is not None: 
            self.model_flow.load_state_dict(torch.load(path_weights_flow))
        if path_weights_rgb is not None:
            self.model_rgb.load_state_dict(torch.load(path_weights_rgb))
        
        self.transforms = transforms.Compose([CenterCrop(224)])
        self.device = device

    def forward(self, frame_list, flow_frames):
        #img = Image.fromarray(img)
        imgs_flow = self.load_flow_frames(flow_frames, 1, len(flow_frames))
        imgs_rgb = self.load_rgb_frames(frame_list, 1, len(frame_list))

        imgs_flow = self.video_to_tensor(self.transforms(imgs_flow))
        imgs_rgb = self.video_to_tensor(self.transforms(imgs_rgb))

        if torch.cuda.is_available():
            self.model_flow = self.model_flow.to(self.device)
            self.model_flow.train(False)
            self.model_rgb = self.model_rgb.to(self.device)
            self.model_rgb.train(False)

        with torch.no_grad():
            
            imgs_rgb = imgs_rgb.unsqueeze(0)
            b,c,t,h,w = imgs_rgb.shape
            features_rgb = []
            for start in range(t-16):
                ip = Variable(torch.from_numpy(imgs_rgb.numpy()[:,:,start:start+16]).to(self.device))
                features_rgb.append(self.model_rgb.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            
            imgs_flow = imgs_flow.unsqueeze(0)
            b,c,t,h,w = imgs_flow.shape
            features_flow = []
            for start in range(t-16):
                ip = Variable(torch.from_numpy(imgs_flow.numpy()[:,:,start:start+16]).to(self.device))
                features_flow.append(self.model_flow.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())

        return np.concatenate(features_rgb, axis=0).squeeze(), np.concatenate(features_flow, axis=0).squeeze()


    def load_rgb_frames(self, frame_list, start, num):
        frames = []
        for i in range(start, start+num):
            img = cv2.cvtColor(frame_list[i-1], cv2.COLOR_BGR2RGB)
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)


    def load_flow_frames(self, flow_frames, start, num):
        frames = []
        for i in range(start, start+num):
            imgx = flow_frames[i-1,:,:,0] 
            imgy = flow_frames[i-1,:,:,1]

            w,h = imgx.shape
            if w < 224 or h < 224:
                d = 224.-min(w,h)
                sc = 1+d/min(w,h)
                imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
                imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
                
            imgx = (imgx/255.)*2 - 1
            imgy = (imgy/255.)*2 - 1
            img = np.asarray([imgx, imgy]).transpose([1,2,0])
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)

    def video_to_tensor(self, pic):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)
        
        Args:
            pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return torch.from_numpy(pic.transpose([3,0,1,2]))