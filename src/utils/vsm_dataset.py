import torch 
import h5py

class VSMDataset(torch.utils.data.Dataset):
    """Video Summarizer Dataset
        Datasets: TVSum, Summe, VSUMM, CoSum, Visiocity
    """

    def __init__(self, hdfs_path, split=None, transform=None):
        """
        Args:
           hdfs_path (string): path of the hdfs processed data
           split (dict): idxs of the train/test split 
        """
        videos_info = h5py.File(hdfs_path)
        self.labels = {}
        self.data = {}
        self.transform = transform
        
        for it, video in enumerate(list(videos_info)):
            self.labels[it] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key in ('gtscore', 'gtsummary', 'user_summary') )
            self.data[it] = dict((key, videos_info[video][key][...])for key in list(videos_info[video]) if key not in ('gtscore', 'gtsummary', 'user_summary') )   
            
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

def show_sample(idx, hdfs_path):

    vsm_dataset = VSMDataset(hdfs_path=hdfs_path)
    video_info, label = vsm_dataset[idx]

    return video_info, label