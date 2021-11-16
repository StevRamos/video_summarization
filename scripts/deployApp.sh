#execute this from the project home
echo 'Creating the environment ...'
conda create -y -n vsm python=3.8.11

echo 'Installing packages ...'
conda activate vsm
pip install -r requirements.txt
pip install gshell

echo 'Downloading weights of the model ...'
mkdir weights_model
cd weights_model
gshell cd --with-id 1Yc-xzYw3yEJE3_64KgF2sfFdw_u8XZAc #download weights of the model
gshell download transformations.pk
gshell download tvsum_random_non_overlap_0.6271.tar.pth
gshell download flow_imagenet.pt
gshell download r3d101_KM_200ep.pth
cd ..

echo 'Downloading api keys ...'
gshell cd --with-id 1P2Db3pAqADpLK5Jn9aDQoHaBGGlx0hrU #download api keys gdrive
gshell download client_secret_vsm_api.json
gshell download token_drive_v3.pickle
gshell download dummy.pk

echo 'Deploying the app ...'
nohup uvicorn app:app --reload --host 0.0.0.0 &

#
#echo 'Building docker image ...'
#sudo docker build -f Dockerfile -t vsm .
#echo 'Run the docker container'
#sudo docker run -d -p 8000:8000 -ti vsm /bin/bash -c "cd src/ && source activate vsm && uvicorn app:app --reload --host 0.0.0.0"