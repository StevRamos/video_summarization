"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import os, sys
import math

from tqdm import tqdm
import cv2
import numpy as np
import h5py
import scipy.io
import pandas as pd
import torch

from models.CNN import ResNet, GoogleNet
from models.I3D import I3D
from KTS.cpd_auto import cpd_auto


def array_to_id(number_array):
    number_array = number_array.squeeze()
    chr_array = [chr(x) for x in number_array] 
    string_array = ''.join(chr_array)
    return string_array

def get_field_by_idx(all_mat, field, idx):
    key = all_mat['tvsum50'][field][idx][0]
    return np.array(all_mat[key])

def get_video_ids(all_mat):
    video_ids = []
    for video_idx in range(len(all_mat['tvsum50']['video'])):
        video_id_na = get_field_by_idx(all_mat, 'video', video_idx)
        video_id = array_to_id(video_id_na)
        video_ids.append(video_id)
    return video_ids

class Generate_Dataset:
    def __init__(self, video_path, path_ground_truth, save_path, dataset='summe', 
                path_weights_flow = "/data/shuaman/video_summarization/datasets/pytorch-i3d/models/flow_imagenet.pt", 
                path_weights_rgb = "/data/shuaman/video_summarization/datasets/pytorch-i3d/models/rgb_imagenet.pt"):
        #self.model = ResNet()
        self.device = torch.device("cuda:" + (os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0") if torch.cuda.is_available() else "cpu")
        print(f'Using device {self.device}')
        if torch.cuda.is_available():
            print(f'Using {torch.cuda.get_device_name(0)}')

        self.model = GoogleNet(self.device)
        self.video_model = I3D(self.device, path_weights_flow, path_weights_rgb)
        self.dataset = dataset
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = './frames'
        self.h5_file = h5py.File(save_path, 'w')
        self.gt_list = []
        self.gt_path = ''
        
        self._set_gt_lista(path_ground_truth, self.dataset)
        self._set_video_list(video_path, self.dataset)
    
    def _extract_video_feature(self, frame_resized, flow_frames):
        features_rgb, features_flow = self.video_model(frame_resized, flow_frames)
        return features_rgb, features_flow

    def _set_video_list(self, video_path, dataset='summe'):
        if os.path.isdir(video_path):
            self.video_path = video_path
            if dataset in ('summe', 'cosum'):
                self.video_list = [videoname for videoname in os.listdir(video_path) if videoname.lower().endswith(".mp4")]
                self.video_list.sort()
            elif dataset=='tvsum':
                self.video_list = [videoname + ".mp4" for videoname in self.gt_list]
            elif dataset in ('ovp', 'youtube'):
                self.video_list = [videoname for videoname in os.listdir(video_path) if videoname.lower().endswith((".mpg",".avi",".flv"))]
                self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.h5_file.create_group('video_{}'.format(idx+1))

    def _set_gt_lista(self, path_ground_truth, dataset='summe'):
        if os.path.isdir(path_ground_truth):
            if dataset=='summe':
                self.gt_path = path_ground_truth
                self.gt_list = [gtvideo for gtvideo in os.listdir(path_ground_truth) if gtvideo.lower().endswith(".mat")]
                self.gt_list.sort()
            elif dataset in ('ovp', 'youtube'):
                self.gt_path = path_ground_truth
                self.gt_list = [gtvideo for gtvideo in os.listdir(path_ground_truth) if os.path.isdir(os.path.join(path_ground_truth, gtvideo))]
                self.gt_list.sort()
            elif dataset=="cosum":
                self.gt_path = path_ground_truth
                df = pd.read_excel(self.gt_path + "/dataset.xlsx",engine='openpyxl')
                df = df.loc[df["DOWNLOADED"]==1].reset_index(drop=True)
                self.gt_list = sorted(list(df[["VIDEO_CATEGORY","SHORT CATEGORY",
                                                "VIDEO_ID_IN_CATEGORY","VIDEO"]].drop_duplicates().values),
                                        key=lambda x: x[-1] + ".mp4", reverse=False)
        else:
            if dataset=='summe':
                self.gt_path = ''
                self.gt_list.append(path_ground_truth)
            elif dataset=='tvsum':
                self.gt_path = h5py.File(path_ground_truth, 'r')
                self.gt_list = get_video_ids(self.gt_path)



    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224))
        #res_pool5 = self.model(frame)
        frame_feat = self.model(frame)
        #frame_feat = res_pool5.cpu().data.numpy().flatten()
        return frame_feat

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

    # TODO : save dataset
    def _save_dataset(self):
        pass

    def _get_ground_truth(self, dataset, gt_path, gt_filename, video_basename, video_feat_for_train, n_frames, fps, gt_info):
        if dataset=='summe':
            if os.path.isdir(self.gt_path):
                    gt_path = os.path.join(self.gt_path, gt_filename)
            gt_video = scipy.io.loadmat(gt_path) 
        elif dataset=='tvsum':
            gt_idx = self.gt_list.index(video_basename)
            annotations = get_field_by_idx(self.gt_path, 'user_anno', gt_idx).T
            annotations = np.where(annotations<=1,0,1)
            gt_video = {'user_score': annotations}
        elif dataset in ('ovp', 'youtube'):
            #youtube gt has less frames than the real ones, we found that 
            #was resampled to 1.03 fps
            factor = 1 if dataset=='ovp' else fps/1.03 
            gt_idx = []
            for user_summ in np.sort([folder for folder in os.listdir(os.path.join(self.gt_path, video_basename)) if os.path.isdir(os.path.join(self.gt_path, video_basename, folder))]):
                list_summ = [int(np.ceil(int(frame.split('.')[0][5:])*factor)) for frame in os.listdir(os.path.join(self.gt_path, video_basename, user_summ)) if frame.lower().endswith(("png","jpeg","jpg"))]
                list_summ.sort()
                gt_idx.append(list_summ)
            m = int(np.ceil(n_frames/(4.5*fps)))
            kernel = np.matmul(video_feat_for_train.astype(np.float32), video_feat_for_train.astype(np.float32).T)
            change_points, _ = cpd_auto(kernel, m, 1, verbose=False)
            change_points *= 15
            change_points = np.hstack((0, change_points, n_frames))
            begin_frames = change_points[:-1]
            end_frames = change_points[1:]
            change_points = np.vstack((begin_frames, end_frames - 1)).T

            annotations = []
            for user in gt_idx:
                new_gt_user = np.zeros(n_frames)
                segments = [segment for segment in change_points for frame in user if (frame>=segment[0]) and (frame<=segment[1])]
                for segment in segments:
                    new_gt_user[int(segment[0]):int(segment[1]+1)] = 1
                annotations.append(new_gt_user)
            annotations = np.array(annotations).T
            gt_video = {'user_score': annotations}

        elif dataset=="cosum":
            category, short_name, video_id, video_name = gt_info
            if video_basename==video_name:
                path_ant_category = os.path.join(self.gt_path, "annotation", category)
                path_shots_category = os.path.join(self.gt_path, "shots", category)
                
                shots = open(os.path.join(path_shots_category, f'{short_name}{video_id}_shots.txt'),'r')
                shots = shots.read().splitlines()
                shots = [int(np.ceil(int(nframe)*n_frames/int(shots[-1]))) for nframe in shots]

                user1 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__dualplus.mat'))["labels"][:,0] 
                user2 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__kk.mat'))["labels"][:,0]
                user3 = scipy.io.loadmat(os.path.join(path_ant_category, f'{video_id}__vv.mat'))["labels"][:,0]

                annotations = []
                for user in (user1, user2, user3):
                    new_gt_user = np.zeros(n_frames)
                    for indexshot in user:
                        if indexshot >= len(shots):
                            indexshot = len(shots) - 1
                    new_gt_user[int(shots[int(indexshot)-1])-1:int(shots[int(indexshot)])-1] = 1
                    annotations.append(new_gt_user)

                annotations = np.array(annotations).T
                gt_video = {'user_score': annotations}
            
            else:
                print("you are no getting the same files")
                sys.exit(0)

        return gt_video 

        

    def generate_dataset(self):
        for video_idx, video_gt in enumerate(tqdm(zip(self.video_list, self.gt_list), total=len(self.video_list))):
            video_filename, gt_info = video_gt
            gt_filename = video_filename
            video_path = video_filename
            gt_path = gt_filename

            if os.path.isdir(self.video_path):
                    video_path = os.path.join(self.video_path, video_filename)
            
            video_basename = ".".join(os.path.basename(video_path).split('.')[:-1])
                
            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
                os.mkdir(os.path.join(self.frame_root_path, video_basename))

            video_capture = cv2.VideoCapture(video_path)
           

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
                    #img_filename = "{}.jpg".format(str(frame_idx).zfill(5))
                    #cv2.imwrite(os.path.join(self.frame_root_path, video_basename, img_filename), frame)
                frame_list.append(frame)
                n_frames += 1
            
            video_capture.release()
            print(f'feature images extraction done: in total of {n_frames} frames')

            frame_resized = [cv2.resize(frame, (224, 224)) for frame in frame_list]
            flow_frames = [cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in frame_resized]
            #optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            #flow_frames = np.array([optical_flow.calc(flow_frames[i],flow_frames[i+1], None) for i in range(len(flow_frames)) if i+2<=len(flow_frames) ])
            print(f'extracting flow frames ....')
            flow_frames = np.array([cv2.calcOpticalFlowFarneback(flow_frames[i],flow_frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(flow_frames)) if i+2<=len(flow_frames) ])
            rate = math.ceil(n_frames/8500) #8500 is the limit 
            frame_resized = frame_resized[::rate]
            flow_frames = flow_frames[::rate]
            print(f'flow frames were extracted in total {len(flow_frames)} flow frames')
            print(f'extracting flow features ...')
            features_rgb, features_flow = self._extract_video_feature(frame_resized, flow_frames)
            print(f'flow features were extracted: len rgb {len(features_rgb)} and {len(features_flow)}')
            video_feat_for_train = np.array(video_feat_for_train)

            gt_video = self._get_ground_truth(self.dataset, gt_path, gt_filename, video_basename,
                                                 video_feat_for_train, n_frames, fps, gt_info)

            user_score = np.array(gt_video["user_score"].T, dtype=np.float32)
            n_frames = user_score.shape[1]

            change_points, n_frame_per_seg = self._get_change_points(video_feat_for_train, n_frames, fps)

            gtscore = np.mean(user_score[:, ::15], axis=0)

            self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = video_basename
            self.h5_file['video_{}'.format(video_idx+1)]['n_steps'] = np.array(np.array(list(picks)).shape[0])
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx+1)]['user_summary'] = user_score
            self.h5_file['video_{}'.format(video_idx+1)]['gtscore'] = gtscore
            self.h5_file['video_{}'.format(video_idx+1)]['features_rgb'] = features_rgb
            self.h5_file['video_{}'.format(video_idx+1)]['features_flow'] = features_flow

        if self.dataset=='tvsum':
            self.gt_path.close()

if __name__ == "__main__":
    #gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/videos/Air_Force_One.mp4', 
    #                        '/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/GT/Air_Force_One.mat',
    #                        'Air_Force_One.h5')

    #gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/TVsum/ydata-tvsum50-v1_1/video/', 
     #                      '/data/shuaman/video_summarization/datasets/raw_datasets/TVsum/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat',
      #                      'eccv16_dataset_tvsum_google_pool5.h5', dataset='tvsum')

    #gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/VSUMM/new_database/', 
     #                       '/data/shuaman/video_summarization/datasets/raw_datasets/VSUMM/newUserSummary/',
      #                          'eccv16_dataset_youtube_google_pool5.h5', dataset='youtube')
    '''
    gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/VSUMM/database/', 
                            '/data/shuaman/video_summarization/datasets/raw_datasets/VSUMM/UserSummary/',
                                'eccv16_dataset_ovp_google_pool5.h5', dataset='ovp')
    '''
    gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/CoSum/videos/', 
                            '/data/shuaman/video_summarization/datasets/raw_datasets/CoSum/',
                                'eccv16_dataset_cosum_google_pool5_i3d.h5', dataset='cosum')
    

    gen.generate_dataset()
    gen.h5_file.close()
    