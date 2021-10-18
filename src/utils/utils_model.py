import os

import torch.nn.init as init
import torch
import wandb
import numpy as np

from .vsm_dataset import VSMDataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def save_weights(model, path, use_wandb):
    torch.save(model.state_dict(), os.path.join(path, 'vsm.pth'))
    if use_wandb:
        wandb.save(os.path.join(path, 'vsm.pth'),
                    base_path='/'.join(path.split('/')[:-2]))   


def get_flags_features(feature1, feature2):
    dict_use_feature = {
        "googlenet": False,
        "resnext": False,
        "inceptionv3": False,
        "i3d_rgb": False,
        "i3d_flow": False,
        "resnet3d": False
    }

    if feature1=="i3d":
        dict_use_feature["i3d_rgb"] = True
        dict_use_feature["i3d_flow"] = True 
    else:
        dict_use_feature[feature1] = True

    if feature2=="i3d":
        dict_use_feature["i3d_rgb"] = True
        dict_use_feature["i3d_flow"] = True 
    else:
        dict_use_feature[feature2] = True

    return dict_use_feature     


def get_dataloaders(dataset_paths, split, dict_use_feature, params, transformations_path=None):
    training_set = VSMDataset(dataset_paths, split=split, key_split="train_keys",
                                googlenet=dict_use_feature["googlenet"],
                                resnext=dict_use_feature["resnext"],
                                inceptionv3=dict_use_feature["inceptionv3"],
                                i3d_rgb=dict_use_feature["i3d_rgb"],
                                i3d_flow=dict_use_feature["i3d_flow"],
                                resnet3d=dict_use_feature["resnet3d"],
                                transformations_path=transformations_path
                                )
    test_set = VSMDataset(dataset_paths, split=split, key_split="test_keys",
                                googlenet=dict_use_feature["googlenet"],
                                resnext=dict_use_feature["resnext"],
                                inceptionv3=dict_use_feature["inceptionv3"],
                                i3d_rgb=dict_use_feature["i3d_rgb"],
                                i3d_flow=dict_use_feature["i3d_flow"],
                                resnet3d=dict_use_feature["resnet3d"],
                                transformations_path=transformations_path
                                )
    training_generator = torch.utils.data.DataLoader(training_set, **params, shuffle=True)
    test_generator = torch.utils.data.DataLoader(test_set, **params)
    return training_generator, test_generator


def init_optimizer(model, learning_rate, weight_decay):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                                weight_decay=weight_decay)
    return optimizer

def get_paths(type_dataset,type_setting, path_tvsum, 
            path_summe, path_ovp, path_youtube, path_cosum):

    if type_setting in ("transfer", "aug", "non_overlap_ord_aug", "non_overlap_rand_aug"):
        paths = [
                path_tvsum,
                path_summe,
                path_ovp,
                path_youtube
                ]
    elif type_setting in ("transfer_vsm", "aug_vsm", "all"):
        paths = [
                path_tvsum,
                path_summe,
                path_ovp,
                path_youtube,
                path_cosum
                ]
    elif type_setting in ("canonical", "non_overlap_ord", "non_overlap_rand"):
        if type_dataset=="summe":
            paths = path_summe
        elif type_dataset=="tvsum":
            paths = path_tvsum
    else:
        paths = []
    
    assert len(paths) > 0, f'Invalid len paths {len(paths)}: expected type_setting transfer,aug,canonical,non_overlap_ord,non_overlap_rand'

    return paths