import torch 
import h5py
import pickle
from sklearn import preprocessing

class VSMDataset(torch.utils.data.Dataset):
    """Video Summarizer Dataset
        Datasets: TVSum, Summe, VSUMM, CoSum, Visiocity
    """

    def __init__(self, hdfs_path, split=None, key_split=None,
                 googlenet=False, 
                 resnext=False, 
                 inceptionv3=False,
                 i3d_rgb=False,
                 i3d_flow=False,
                 resnet3d=False,
                 transformations_path=None
                ):
        """
        Args:
           hdfs_path (string): path of the hdfs processed data
           split (dict): idxs of the train/test split 
        """
        if not isinstance(hdfs_path, list):
            hdfs_paths = [hdfs_path]
        else:
            hdfs_paths = hdfs_path 

        if transformations_path is not None:
            transformations = pickle.load(open(transformations_path, 'rb'))
        
        self.labels = {}
        self.data = {}

        keys_to_avoid = ['gtscore', 'gtsummary', 'user_summary']
        
        if not googlenet:
            keys_to_avoid.append('features')
        if not resnext:
            keys_to_avoid.append('features_rn')
        if not inceptionv3:
            keys_to_avoid.append('features_iv3')
        if not i3d_rgb:
            keys_to_avoid.append('features_rgb')
        if not i3d_flow:
            keys_to_avoid.append('features_flow')
        if not resnet3d:
            keys_to_avoid.append('features_3D')        
        
        iterator_videos = 0
        for path in hdfs_paths:
            videos_info = h5py.File(path)  

            if 'tvsum' in path:
                name_dataset = 'tvsum'
            elif 'summe' in path:
                name_dataset = 'summe'
            elif 'ovp' in path:
                name_dataset = 'ovp'
            elif 'youtube' in path:
                name_dataset = 'youtube'
            elif 'cosum' in path:
                name_dataset = 'cosum'
            elif 'mvs1k' in path:
                name_dataset = 'mvs1k'
            elif 'visiocity' in path:
                name_dataset = 'visiocity'
            
            videos_to_iterate = list(videos_info)
            if split:
                if len(hdfs_paths)==1:
                    videos_to_iterate = [video for video in videos_to_iterate if video in split[key_split]]
                else:
                    key_videos = [video.split('/')[-1] for video in split[key_split] if name_dataset in video]
                    videos_to_iterate = [video for video in videos_to_iterate if video in key_videos]
                    

            for it, video in enumerate(videos_to_iterate):
                self.labels[iterator_videos] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key in ('gtscore', 'gtsummary', 'user_summary') )
                self.data[iterator_videos] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key not in keys_to_avoid )   
                self.data[iterator_videos]['name_dataset'] = name_dataset

                if "video_name" in self.data[iterator_videos].keys():
                    self.data[iterator_videos]["video_name"] = str(self.data[iterator_videos]["video_name"]) 
                
                if transformations_path is not None:
                    if "features_rn" in self.data[iterator_videos].keys():
                        features_rn = self.data[iterator_videos]["features_rn"]
                        self.data[iterator_videos]["features_rn"] = transformations["pca_rn"].transform(transformations["normalizer_rn"].transform(features_rn))
                    if "features_iv3" in self.data[iterator_videos].keys():
                        features_iv3 = self.data[iterator_videos]["features_iv3"]
                        self.data[iterator_videos]["features_iv3"] = transformations["pca_iv3"].transform(transformations["normalizer_iv3"].transform(features_iv3))
                    if "features_3D" in self.data[iterator_videos].keys():
                        features_3D = self.data[iterator_videos]["features_3D"]
                        self.data[iterator_videos]["features_3D"] = transformations["pca_3D"].transform(transformations["normalizer_3D"].transform(features_3D))

                #apply l2norm equal to 1 for each vector
                features_to_normalize = [key for key in self.data[iterator_videos].keys() if 'features' in key]
                for feat in features_to_normalize:
                    self.data[iterator_videos][feat] = preprocessing.normalize(self.data[iterator_videos][feat], norm='l2')
                    
                iterator_videos = iterator_videos + 1
                

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]        
        return X, y
    
    def get_feature(self, index, feature):
        X = self.data[index][feature]
        return X

def show_sample(idx, hdfs_path):

    vsm_dataset = VSMDataset(hdfs_path=hdfs_path)
    video_info, label = vsm_dataset[idx]

    return video_info, label