# Video Summarization of Sports Videos

Based on https://github.com/TIBHannover/MSVA

To download the raw dataset (SumMe, TVSum, VSumm, CoSum, and Visiocity), I used the gshell library. It can be downloaded from pip. Since it has 16GB I recommend using it. Run the following command to get the preprocessed data set.

1. `pip3 install gshell==5.5.2`
2. `gshell init` to log into your account
3. `./scripts/downloadDataset.sh`

Manually, this is the [link](https://drive.google.com/uc?export=download&confirm=sSIJ&id=19XhWuwyA1ahGM8JYxMkvhjTAbel-YBeG)


More about gshell: https://pypi.org/project/gshell/

# dataset

################ Instructions ################
This folder contains four datasets for video summarization:
(1): eccv16_dataset_summe_google_pool5.h5
(2): eccv16_dataset_tvsum_google_pool5.h5
(3): eccv16_dataset_ovp_google_pool5.h5
(4): eccv16_dataset_youtube_google_pool5.h5



Each dataset follows the same data structure:
***********************************************************************************************************************************************
SumMe: video name is stored in video_i/video_name.
TVSum: video1-50 corresponds to the same order in ydata-tvsum50.mat, which is the original matlab file provided by TVSum.

  /key
  
    /features                 2D-array with shape (n_steps, feature-dimension)
                              contains feature vectors representing video frames. Each video frame can be represented by a feature vector (containing some semantic meanings), extracted by a pretrained convolutional neural network (e.g. GoogLeNet). 
                              Se usa en train, test e inferencia

    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
                              is the average of multiple importance scores (used by regression loss).
                              Se usa en train y test
    
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
                              contains multiple key-clips given by human annotators and we need to compare our machine summary with each one of the user summaries
                              Se usa en test

    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
                              corresponds to shot transitions, which are obtained by temporal segmentation approaches that segment a video into disjoint shots
                              num_segments is number of total segments a video is cut into. 
                              Se usa en test

    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
                              Se usa en test

    /n_frames                 number of frames in original video
                              Se usa en test

    /picks                    positions of subsampled frames in original video
                              is an array storing the position information of subsampled video frames. We do not process each video frame since adjacent frames are very similar. We can subsample a video with 2 frame per second or 1 frame per second, which will result in less frames but they are informative.
                              is useful when we want to interpolate the subsampled frames into the original video (say you have obtained importance scores for subsampled frames and you want to get the scores for the entire video
                              can indicate which frames are scored and the scores of surrounding frames can be filled with these frames).
                              Se usa en test

    /n_steps                  number of subsampled frames
                              No se usa en nada
    
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
                              is a binary vector indicating indices of keyframes, and is provided by original datasets as well (this label can be used for maximum likelihood loss).
                              No se usa en nada
    
    /video_name (optional)    original video name, only available for SumMe dataset
                              Se usa en train y test summe
***********************************************************************************************************************************************
Note: OVP and YouTube only contain the first three keys, i.e. ['features', 'gtscore', 'gtsummary']


* in case doesnt work  pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
