import h5py


path_data = "/data/shuaman/video_summarization/datasets/raw_datasets/TvSum/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat"
all_mat = h5py.File(path_data, 'r')

all_mat.close()