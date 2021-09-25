"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import os, sys
#sys.path.append('../')
#os.chdir('../')
from models.CNN import ResNet, GoogleNet
#from utils.KTS.cpd_auto import cpd_auto
from KTS.cpd_auto import cpd_auto


from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
import scipy.io

class Generate_Dataset:
    def __init__(self, video_path, path_ground_truth, save_path):
        #self.model = ResNet()
        self.model = GoogleNet()
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = './frames'
        self.h5_file = h5py.File(save_path, 'w')
        self.gt_list = []
        self.gt_path = ''

        self._set_video_list(video_path)
        self._set_gt_lista(path_ground_truth)

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = [videoname for videoname in os.listdir(video_path) if videoname.endswith(".mp4")]
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))

    def _set_gt_lista(self, path_ground_truth):
        if os.path.isdir(path_ground_truth):
            self.gt_path = path_ground_truth
            self.gt_list = [gtvideo for gtvideo in os.listdir(path_ground_truth) if gtvideo.endswith(".mat")]
            self.gt_list.sort()
        else:
            self.gt_path = ''
            self.gt_list.append(path_ground_truth)


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
        change_points, _ = cpd_auto(kernel, m, 1, verbose=True)
        #change_points, _ = cpd_auto(kernel, seq_len-1, 1, verbose=True)
        
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

    def generate_dataset(self):
        for video_idx, video_gt in enumerate(tqdm(zip(self.video_list, self.gt_list), total=len(self.video_list))):
            video_filename, _ = video_gt
            video_path = video_filename
            gt_filename = video_filename
            gt_path = gt_filename

            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_basename = ".".join(os.path.basename(video_path).split('.')[:-1])

            if os.path.isdir(self.gt_path):
                gt_path = os.path.join(self.gt_path, video_basename + ".mat")

            

            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
                os.mkdir(os.path.join(self.frame_root_path, video_basename))
            video_capture = cv2.VideoCapture(video_path)
            gt_video = scipy.io.loadmat(gt_path) 

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
                #success, frame = video_capture.read()
                
                    frame_feat = self._extract_feature(frame)
                    picks.append(n_frames)
                    video_feat_for_train.append(frame_feat)

                    #img_filename = "{}.jpg".format(str(frame_idx).zfill(5))
                    #cv2.imwrite(os.path.join(self.frame_root_path, video_basename, img_filename), frame)

                n_frames += 1

            video_capture.release()
            video_feat_for_train = np.array(video_feat_for_train)

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

if __name__ == "__main__":
    #gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/videos/Air_Force_One.mp4', 
    #                        '/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/GT/Air_Force_One.mat',
    #                        'Air_Force_One.h5')

    gen = Generate_Dataset('/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/videos/', 
                           '/data/shuaman/video_summarization/datasets/raw_datasets/SumMe/GT/',
                            'eccv16_dataset_summe_google_pool5.h5')

    gen.generate_dataset()
    gen.h5_file.close()
    