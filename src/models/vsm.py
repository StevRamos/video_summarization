import os
import math
from itertools import groupby

import cv2
import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr, rankdata
from sklearn import preprocessing
import pickle

from .msva import MSVA
from src.utils import weights_init, generate_summary, evaluate_summary, save_weights 
from src.utils import get_flags_features, get_dataloaders, get_paths, init_optimizer, parse_configuration
from src.KTS.cpd_auto import cpd_auto
from .CNN import ResNet, GoogleNet, Inception
from .CNN3D import I3D, ResNet3D

class VideoSumarizer():
    def __init__(self, config, use_wandb):
        self.use_wandb = use_wandb
        self.config = config
        self.device = torch.device("cuda:" + (os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0") if torch.cuda.is_available() else "cpu")
        print(f'Using device {self.device}')
        self.msva = self.init_model()

    def init_model(self):
        msva = MSVA(feature_len=self.config.feature_len)
        msva.eval()
        msva.apply(weights_init)
        msva.to(self.device)
        if self.use_wandb:
            wandb.watch(msva, log="all")
        msva.train()
        return msva

    def load_weights(self, weights_path):
        self.msva.load_state_dict(torch.load(weights_path, 
                                            map_location=torch.device(self.device)))

    def _get_model_frame_feature(self, resnet=True, inception=True, googlenet=True):
        image_models = {}
        if resnet:
            resnet = ResNet(self.device)
            image_models["resnet"] = resnet.eval()
        if inception:
            inception = Inception(self.device)
            image_models["inception"] = inception.eval()
        if googlenet:
            googlenet = GoogleNet(self.device)
            image_models["googlenet"] = googlenet.eval()
        return image_models

    def _get_model_video(self, path_weights_flow, path_weights_rgb, paht_weights_r3d101_KM):
        i3d = I3D(self.device, path_weights_flow, path_weights_rgb)
        i3d = i3d.eval()
        resnet3D = ResNet3D(device=self.device, path_weights=paht_weights_r3d101_KM)
        resnet3D = resnet3D.eval()
        video_models = {
            "i3d": i3d,
            "resnet3D": resnet3D,
        }
        return video_models

    def load_weights_descriptor_models(self, 
                                       weights_path="/home/shuaman/video_sm/video_summarization/pretrained_models/tvsum_random_non_overlap_0.6271.tar.pth", 
                                       path_weights_flow="/data/shuaman/video_summarization/datasets/pytorch-i3d/models/flow_imagenet.pt",
                                       paht_weights_r3d101_KM="/data/shuaman/video_summarization/datasets/3D-ResNets-PyTorch/weights/r3d101_KM_200ep.pth",
                                       transformations_path="/data/shuaman/video_summarization/datasets/processed_datasets/transformations.pk"
                                       ):
        self.load_weights(weights_path)
        self.image_models = self._get_model_frame_feature(resnet=self.config.resnext, inception=self.config.inceptionv3, googlenet=self.config.googlenet)
        self.video_models = self._get_model_video(path_weights_flow=path_weights_flow, path_weights_rgb=None,
                                                paht_weights_r3d101_KM=paht_weights_r3d101_KM)
        transformations_path = transformations_path if self.config.feature_len==1024 else None
        self.transformations = pickle.load(open(transformations_path, 'rb'))

    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_feat = {}
        for model in self.image_models.keys():
            frame_feat[model] = self.image_models[model](frame) 
        return frame_feat

    def _extract_video_feature(self, frame_resized, flow_frames):
        _, features_flow = self.video_models["i3d"](frame_resized, flow_frames)
        features_3D = self.video_models["resnet3D"](frame_resized)
        return _, features_flow, features_3D

    def _get_change_points(self, video_feat, n_frame, fps):
        video_feat = video_feat.astype(np.float32)
        seq_len = len(video_feat)
        n_frames = n_frame
        m = int(np.ceil(seq_len/10 - 1))
        kernel = np.matmul(video_feat, video_feat.T)
        change_points, _ = cpd_auto(kernel, m, 1, verbose=False)
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg

    def _process_video(self, video_source):
        video_capture = cv2.VideoCapture(video_source)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_list = []
        picks = []
        video_feat_for_train = []
        n_frames = 0

        while True:
            success, frame = video_capture.read()
            if not success:
                break
            if n_frames % 15 == 0:
                frame_feat = self._extract_feature(frame)
                picks.append(n_frames)
                video_feat_for_train.append(frame_feat)
            frame_list.append(frame)
            n_frames += 1
        
        video_capture.release()
        rate = math.ceil(n_frames/8500)
        frame_list = frame_list[::rate]
        frame_resized = [cv2.resize(frame, (224, 224)) for frame in frame_list]
        flow_frames = [cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in frame_resized]
        print("flow optical")
        flow_frames = np.array([cv2.calcOpticalFlowFarneback(flow_frames[i],flow_frames[i+15], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(flow_frames)) if i+16<=len(flow_frames) ])

        rate = 15
        frame_resized = frame_resized[::rate]
        flow_frames = flow_frames[::rate]
        _, features_flow, features_3D = self._extract_video_feature(frame_resized, flow_frames)
        features_3D = self.transformations["pca_3D"].transform(self.transformations["normalizer_3D"].transform(features_3D))
        features_3D = preprocessing.normalize(features_3D, norm='l2')

        video_feat_for_train_googlenet = np.array([feature["googlenet"] for feature in video_feat_for_train])

        video_feat_for_train_resnet = np.array([feature["resnet"] for feature in video_feat_for_train])
        video_feat_for_train_resnet = self.transformations["pca_rn"].transform(self.transformations["normalizer_rn"].transform(video_feat_for_train_resnet))
        video_feat_for_train_resnet = preprocessing.normalize(video_feat_for_train_resnet, norm='l2')
        #n_frames = user_score.shape[1]
        print("calculatin change points")
        change_points, n_frame_per_seg = self._get_change_points(video_feat_for_train_googlenet, n_frames, fps)
        
        return fps, width, height, n_frames, video_feat_for_train_googlenet, video_feat_for_train_resnet, features_flow, features_3D, np.array(change_points), n_frame_per_seg, np.array(picks)


    def summarize_video(self, video_source):
        self.msva.eval()
        print("processing video")
        fps, width, height, n_frames, video_feat_for_train_googlenet, video_feat_for_train_resnet, features_flow, features_3D, change_points, n_frame_per_seg, picks = self._process_video(video_source)
        
        video_name = video_source.split('/')[-1]
        tam = os.path.getsize(video_source)

        print("forward prop")
        with torch.no_grad():
            features = [video_feat_for_train_googlenet, video_feat_for_train_resnet, features_flow, features_3D]
            shape_desire = video_feat_for_train_googlenet.shape[0]
            features = [cv2.resize(feature, (feature.shape[1],shape_desire), interpolation = cv2.INTER_AREA) for feature in features]
            features = [torch.from_numpy(feature).unsqueeze(0) for feature in features]
            features = [feature.float().to(self.device) for feature in features]
            y, _ = self.msva(features, shape_desire)
            summary = y[0].detach().cpu().numpy()

        return video_name, tam, width, height, fps, n_frames/fps, summary, change_points, n_frames, n_frame_per_seg, picks

    def generate_summary_proportion(self, video_source, summary, change_points, 
                                    n_frames, n_frame_per_seg, picks, proportion, video_saved="output.mp4"):
        machine_summary = generate_summary(summary, change_points, n_frames, 
                                            n_frame_per_seg, picks, proportion)
        print("generating summary")
        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_saved, fourcc, fps, (width, height))

        frame_idx = 0
        n_frames_spotlight = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if machine_summary[frame_idx]:
                out.write(frame)
                n_frames_spotlight += 1
            frame_idx += 1
        
        out.release()
        cap.release()

        n_segments = len([sum(g) for i, g in groupby(machine_summary) if i == 1])

        return n_frames_spotlight/fps, n_segments


    def infer(self, video_source, video_saved="output.mp4", proportion=0.15):

        video_name, tam, res_w, res_h, fps, dur_orig, summary, change_points, n_frames, n_frame_per_seg, picks = self.summarize_video(video_source)
        dur_spotlight, n_segments = self.generate_summary_proportion(video_source, summary, change_points, 
                                                                        n_frames, n_frame_per_seg, picks, proportion, video_saved)

        return video_name, tam, res_w, res_h, fps, dur_orig, dur_spotlight, n_segments
        
    def train_step(self, training_generator, criterion, optimizer):
        self.msva.train()

        avg_loss = []

        for video_info, label in training_generator:
            target = (label['gtscore'].squeeze(0)).cpu().numpy()
            features = [(video_info[key].squeeze(0)).cpu().numpy() for key in video_info.keys() if 'features' in  key]
        
            shape_desire = target.shape[0]
            features = [cv2.resize(feature, (feature.shape[1],shape_desire), interpolation = cv2.INTER_AREA) for feature in features]
        
            features = [torch.from_numpy(feature).unsqueeze(0) for feature in features]
            target =  torch.from_numpy(target).unsqueeze(0)

            target -= target.min()
            target = np.true_divide(target, target.max())

            target = target.float().to(self.device)
            features = [feature.float().to(self.device) for feature in features]

            y, _ = self.msva(features, shape_desire)

            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.item())

        avg_loss = np.mean(np.array(avg_loss))

        return avg_loss

    def eval_function(self, test_generator):
        self.msva.eval()

        avg_loss = []
        fms = []
        kts = []
        sps = []

        with torch.no_grad():
            for video_info, label in test_generator:

                target = (label['gtscore'].squeeze(0)).cpu().numpy()
                features = [(video_info[key].squeeze(0)).cpu().numpy() for key in video_info.keys() if 'features' in  key]
            
                shape_desire = target.shape[0]
                features = [cv2.resize(feature, (feature.shape[1],shape_desire), interpolation = cv2.INTER_AREA) for feature in features]
            
                features = [torch.from_numpy(feature).unsqueeze(0) for feature in features]
                target =  torch.from_numpy(target).unsqueeze(0)

                target -= target.min()
                target = np.true_divide(target, target.max())

                target = target.float().to(self.device)
                features = [feature.float().to(self.device) for feature in features]

                y, _ = self.msva(features, shape_desire)
                
                criterion = torch.nn.MSELoss()
                criterion.to(self.device)

                test_loss = criterion(y, target)

                avg_loss.append(test_loss.item())
                summary = y[0].detach().cpu().numpy()

                machine_summary = generate_summary(summary, (video_info["change_points"].squeeze(0)).cpu().numpy(),
                                                  (video_info["n_frames"].squeeze(0)).cpu().numpy(), (video_info["n_frame_per_seg"].squeeze(0)).cpu().numpy(),
                                                    (video_info["picks"].squeeze(0)).cpu().numpy())

                eval_metric = 'avg' if video_info["name_dataset"][0] == "tvsum" else 'max'
                fm, _, _ = evaluate_summary(machine_summary, (label["user_summary"].squeeze(0)).cpu().numpy(),
                                                eval_metric)
                
                fms.append(fm)
                y_pred2 = machine_summary
                y_true2 = (label["user_summary"].squeeze(0)).cpu().numpy().mean(axis=0)
                pS = spearmanr(y_pred2, y_true2)[0]
                kT = kendalltau(rankdata(-np.array(y_true2)), rankdata(-np.array(y_pred2)))[0]
                kts.append(kT)
                sps.append(pS)
        

        f_score = np.mean(fms)
        kt = np.mean(kts)
        sp = np.mean(sps)
        avg_loss = np.mean(np.array(avg_loss))

        return f_score, kt, sp, avg_loss


    def train(self, split=None, n_split=None, pretrained_model=None):
        if torch.cuda.is_available():
            print(f'Training in {torch.cuda.get_device_name(0)}')
        else:
            print('Training in CPU')
        
        if pretrained_model:
            self.load_weights(pretrained_model)

        if self.config.save_weights:
            if self.use_wandb:
                path_saved_weights = os.path.join(self.config.path_saved_weights, wandb.run.id)
            else:
                if split is None:
                    path_saved_weights = os.path.join(self.config.path_saved_weights, f'{self.config.weights_default}_{self.config.feature_1}_{self.config.feature_2}')
                else:
                    path_saved_weights = os.path.join(self.config.path_saved_weights, 
                                        f'{str(n_split)}_{self.config.type_dataset}_{self.config.type_setting}_{self.config.feature_1}_{self.config.feature_2}')
            try:
                os.mkdir(path_saved_weights)
            except OSError:
                pass

        dict_use_feature = {
            "googlenet": self.config.googlenet,
            "resnext": self.config.resnext,
            "inceptionv3": self.config.inceptionv3,
            "i3d_rgb": self.config.i3d_rgb,
            "i3d_flow": self.config.i3d_flow,
            "resnet3d": self.config.resnet3d
        }

        #dict_use_feature = get_flags_features(self.config.feature_1, self.config.feature_2)

        dict_paths = {
            'path_tvsum': self.config.path_tvsum,
            'path_summe': self.config.path_summe,
            'path_ovp': self.config.path_ovp,
            'path_youtube': self.config.path_youtube,
            'path_cosum': self.config.path_cosum,
        }

        dataset_paths = get_paths(self.config.type_dataset, self.config.type_setting, **dict_paths)
        params = {
                'batch_size': 1,
                'num_workers': 4
                }

        transformations_path = self.config.transformations_path if self.config.feature_len==1024 else None
        training_generator, test_generator = get_dataloaders(dataset_paths, split,
                                                            dict_use_feature, params,
                                                            transformations_path
                                                            )
                                                            
        optimizer = init_optimizer(self.msva, self.config.learning_rate, self.config.weight_decay)
        criterion = torch.nn.MSELoss()
        criterion.to(self.device)

        sameCount = 0
        max_val_fscore = 0
        maxkt = 0
        maxsp = 0
        max_val_fscoreLs=[]

        for epoch in tqdm(range(self.config.epochs_max)):
            train_loss = self.train_step(training_generator, criterion, optimizer)
            f_score, kt, sp, test_loss = self.eval_function(test_generator)

            metrics_log = {
                "epoch" + f'_split_{n_split}' if split else "epoch": epoch + 1,
                "train_loss" + f'_split_{n_split}' if split else "epoch": train_loss,
                "f_score" + f'_split_{n_split}' if split else "epoch": f_score,
                "kt" + f'_split_{n_split}' if split else "epoch": kt,
                "sp" + f'_split_{n_split}' if split else "epoch": sp,
                "test_loss" + f'_split_{n_split}' if split else "epoch": test_loss
            }

            if self.config.save_weights and ((epoch+1) % int(self.config.epochs_max/self.config.num_backups)) == 0:
                path_save_epoch = os.path.join(path_saved_weights, f'epoch_{epoch+1}')
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass
                save_weights(self.msva, path_save_epoch, self.use_wandb)

            if self.use_wandb:
                wandb.log(metrics_log)

            print("Losses/Metrics")
            print('Epoch [{}/{}], Train loss: {:.4f}'.format(epoch +1, 
                                    self.config.epochs_max, train_loss))
            print('Epoch [{}/{}], Test loss: {:.4f}'.format(epoch +1, 
                                    self.config.epochs_max, test_loss))
            print('Epoch [{}/{}], F1 score: {:.4f}'.format(epoch +1, 
                                    self.config.epochs_max, f_score))
            print('Epoch [{}/{}], Spearman s correlation: {:.4f}'.format(epoch +1, 
                                    self.config.epochs_max, sp))
            print('Epoch [{}/{}], Kendall s correlation: {:.4f}'.format(epoch +1, 
                                    self.config.epochs_max, kt))

            if max_val_fscore < f_score:
                max_val_fscore = f_score
                maxkt = kt
                maxsp = sp
            max_val_fscoreLs.append(max_val_fscore)

            if(len(max_val_fscoreLs)>2) and (max_val_fscoreLs[-2]>=max_val_fscoreLs[-1]):
                sameCount+=1
            else:
                sameCount=0

            if(sameCount>=self.config.sameAccStopThres):
                if self.config.save_weights:
                    path_save_epoch = os.path.join(path_saved_weights, f'epoch_stopthreshold')
                    try:
                        os.mkdir(path_save_epoch)
                    except OSError:
                        pass
                    save_weights(self.msva, path_save_epoch, self.use_wandb)
                break

        if self.use_wandb and (split is None):
            wandb.finish()

        return max_val_fscore, maxkt, maxsp, train_loss, test_loss


    def train_cross_validation(self, pretrained_model=None):
        f_avg = 0
        kt_avg = []
        sp_avg = []
        trl_avg = 0
        tsl_avg = 0
        
        split_name = f'path_split_{self.config.type_dataset}_{self.config.type_setting}'
        path_split = vars(self.config)[split_name] if not self.use_wandb else vars(self.config)["_items"][split_name] 
        splits = parse_configuration(path_split)

        for n_split in range(len(splits)):
            self.msva = self.init_model()
            print(f'Split number {n_split+1}')
            max_val_fscore, maxkt, maxsp, maxtrl, maxtsl = self.train(splits[n_split], n_split+1, pretrained_model)
            f_avg += max_val_fscore
            if (maxkt>=-1) and (maxkt<=1):
                 kt_avg.append(maxkt)
            if (maxsp>=-1) and (maxsp<=1):
                 sp_avg.append(maxsp)
            trl_avg += maxtrl
            tsl_avg += maxtsl

        f_avg = f_avg/len(splits)
        kt_avg = sum(kt_avg)/len(kt_avg)
        sp_avg = sum(sp_avg)/len(sp_avg)
        trl_avg = trl_avg/len(splits)
        tsl_avg = tsl_avg/len(splits)   

        if self.use_wandb:
            wandb.log({
                "train_loss": trl_avg,
                "f_score": f_avg,
                "kt": kt_avg,
                "sp": sp_avg,
                "test_loss": tsl_avg 
            })

        print("Metrics - cross validation")
        print('Train loss: {:.4f}'.format(trl_avg))
        print('Test loss: {:.4f}'.format(tsl_avg))
        print('F1 score: {:.4f}'.format(f_avg))
        print('Spearman s correlation: {:.4f}'.format(sp_avg))
        print('Kendall s correlation: {:.4f}'.format(kt_avg))    

        if self.use_wandb:
            wandb.finish()