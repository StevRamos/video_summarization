# video_summarization

To download the raw dataset (SumMe, TVSum and VSumm):
  1. `pip3 install gshell`
  2. `gshell init` or `gdown`
  3. `gshell cd --with-id 1QDbSSW4CilBGI_eNkVrfyoWwHnZJQcGn`
  4. `gshell download --recursive raw_datasets`

# dataset

################ Instructions ################
This folder contains four datasets for video summarization:
(1): eccv16_dataset_summe_google_pool5.h5
(2): eccv16_dataset_tvsum_google_pool5.h5
(3): eccv16_dataset_ovp_google_pool5.h5
(4): eccv16_dataset_youtube_google_pool5.h5

Each dataset follows the same data structure:
***********************************************************************************************************************************************
  /key
  
    /features                 2D-array with shape (n_steps, feature-dimension)
                              contains feature vectors representing video frames. Each video frame can be represented by a feature vector (containing some semantic meanings), extracted by a pretrained convolutional neural network (e.g. GoogLeNet).

    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
                              is the average of multiple importance scores (used by regression loss).
    
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
                              contains multiple key-clips given by human annotators and we need to compare our machine summary with each one of the user summaries
    
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
                              corresponds to shot transitions, which are obtained by temporal segmentation approaches that segment a video into disjoint shots
                              num_segments is number of total segments a video is cut into. 

    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    
    /n_frames                 number of frames in original video
    
    /picks                    positions of subsampled frames in original video
                              is an array storing the position information of subsampled video frames. We do not process each video frame since adjacent frames are very similar. We can subsample a video with 2 frame per second or 1 frame per second, which will result in less frames but they are informative.
                              is useful when we want to interpolate the subsampled frames into the original video (say you have obtained importance scores for subsampled frames and you want to get the scores for the entire video
                              can indicate which frames are scored and the scores of surrounding frames can be filled with these frames).

    /n_steps                  number of subsampled frames
    
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
                              is a binary vector indicating indices of keyframes, and is provided by original datasets as well (this label can be used for maximum likelihood loss).
    
    /video_name (optional)    original video name, only available for SumMe dataset
***********************************************************************************************************************************************
Note: OVP and YouTube only contain the first three keys, i.e. ['features', 'gtscore', 'gtsummary']


# in case doesnt work  pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html