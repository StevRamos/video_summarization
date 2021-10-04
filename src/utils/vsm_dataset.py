import torch 
import h5py

class VSMDataset(torch.utils.data.Dataset):
    """Video Summarizer Dataset
        Datasets: TVSum, Summe, VSUMM, CoSum, Visiocity
    """

    def __init__(self, hdfs_path, split=None, 
                 googlenet=False, 
                 resnext=False, 
                 inceptionv3=False,
                 i3d_rgb=False,
                 i3d_flow=False,
                 resnet3d=False
                ):
        """
        Args:
           hdfs_path (string): path of the hdfs processed data
           split (dict): idxs of the train/test split 
        """
        videos_info = h5py.File(hdfs_path)
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
        
        for it, video in enumerate(list(videos_info)):
            self.labels[it] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key in ('gtscore', 'gtsummary', 'user_summary') )
            self.data[it] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key not in keys_to_avoid )   
            
            if "video_name" in self.data[it].keys():
                self.data[it]["video_name"] = str(self.data[it]["video_name"]) 
            
        if split:
            #TO-DO
            pass
        

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

