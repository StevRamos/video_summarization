import os
import shutil
from typing import Optional
from starlette.background import BackgroundTasks

#Pydantic
from pydantic import BaseModel
from pydantic import Field

#FastAPI
from fastapi import FastAPI
from fastapi import Body, Query, UploadFile, File
from fastapi import status
#from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

#Other
import cv2
import tempfile
import pickle

#internal
from api.googleDriveAPI import GoogleAPI
from src.models import VideoSumarizer
from src.utils import configure_model

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


GAPI = GoogleAPI()

#PATHS - GOOGLE DRIVE
PARENT_FOLDERID_VIDEO_INPUT = "15dDUM5zTwDlwsMVaFAMXMqcmxIPQI66B"
PARENT_FOLDERID_SUMMARY_OUTPUT = "105BwKhDM7ex532K7yx9IfvEMOwNz2mGD"
PARENT_FOLDERID_SPOTLIGHT_OUTPUT = "1ogajEOHmCf19dzaeXb3lQb9NjVX7JhHA"


#START video sm
use_wandb = False
config_file = "configs/config_deployment.json"
config = configure_model(config_file, use_wandb)

weights_path = "weights_model//tvsum_random_non_overlap_0.6271.tar.pth"
path_weights_flow = "weights_model/flow_imagenet.pt"
paht_weights_r3d101_KM = "weights_model/r3d101_KM_200ep.pth"
transformations_path = "weights_model/transformations.pk"
vsm = VideoSumarizer(config, use_wandb)
vsm.load_weights_descriptor_models(weights_path=weights_path,
                                    path_weights_flow=path_weights_flow,
                                    paht_weights_r3d101_KM=paht_weights_r3d101_KM,
                                    transformations_path=transformations_path)
#END video sm

PATH_DATA = tempfile.gettempdir()



class VideoresponseBase(BaseModel):
    video_name: str = Field(
        ..., 
        min_length=1,
        example="v71.avi"
    )
    tam: int = Field(
        ..., 
        gt=0,
        example="16553384"
    )
    res_w: int = Field(
        ...,
        gt=0,
        example="320"
    )
    res_h: int = Field(
        ...,
        gt=0,
        example="240"
    )
    fps: float = Field(
        ...,
        gt=0,
        example="29.91"
    )
    dur_orig: float = Field(
        ...,
        gt=0,
        example="275.03"
    )
    video_id: str = Field(
        ...,
        min_length=1,
        example="15451dasd"
    )
    summary_id: str = Field(
        ...,
        min_length=1,
        example="sum1288"
    )


class SummaryresponseBase(BaseModel):
    dur_spotlight: float = Field(
        ...,
        gt=0,
        example="41.11"
    )
    n_segments: int = Field(
        ...,
        gt=0,
        example="11"
    )
    spotlight_id: str = Field(
        ...,
        min_length=1,
        example="spotiij123"  
    )


@app.post(
    path="/summarize-video",
    status_code=status.HTTP_201_CREATED
)
def summarize_video(
    video: UploadFile = File(...)
):
    with open(os.path.join(PATH_DATA, video.filename), "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    video_filename = video.filename
    path_video = os.path.join(PATH_DATA, video_filename)
    
    #SAVE IN GOOGLE DRIVE
    file_video_ids = GAPI.upload_files([video_filename], PATH_DATA, PARENT_FOLDERID_VIDEO_INPUT)
    video_id = file_video_ids[video_filename]

    #START video sm
    video_name, tam, res_w, res_h, fps, dur_orig, summary, change_points, n_frames, n_frame_per_seg, picks = vsm.summarize_video(path_video)
    #video_name, tam, res_w, res_h, fps, dur_orig = "v71.avi", 16553384, 320, 240, 29.916666666666668, 275.0306406685237
    #dummy_summary = pickle.load(open("dummy.pk", 'rb'))
    #summary, change_points, n_frames, n_frame_per_seg, picks = dummy_summary['summary'], dummy_summary['change_points'], dummy_summary['n_frames'], dummy_summary['n_frame_per_seg'], dummy_summary['picks']
    os.remove(path_video)

    dict_summary = {
        "summary": summary,
        "change_points": change_points,
        "n_frames": n_frames,
        "n_frame_per_seg": n_frame_per_seg,
        "picks": picks
    }
    filename_dict_summmary = f'{video_filename}_{video_id}.pk'
    path_dict_summmary = os.path.join(PATH_DATA, filename_dict_summmary)
    pickle.dump(dict_summary, open(path_dict_summmary, 'wb'))
    file_summary_ids = GAPI.upload_files([filename_dict_summmary], PATH_DATA, PARENT_FOLDERID_SUMMARY_OUTPUT)
    summary_id = file_summary_ids[filename_dict_summmary]
    os.remove(path_dict_summmary)

    video_response = VideoresponseBase(video_name=video_filename, tam=tam,
                                        res_w=res_w, res_h=res_h, fps=fps, dur_orig=dur_orig,
                                        video_id=video_id, summary_id=summary_id)
    return video_response


@app.get(
    path="/get-spotlight",
    status_code=status.HTTP_201_CREATED
)
def get_spotlight(
    video_id: str = Query(
        ...,
        title="Id video",
        description="This is the video id",
        example="15451dasd"
    ),
    summary_id: str = Query(
        ...,
        title="Id summary",
        description="This is the summary id",
        example="sum1288"
    ),
    proportion: float = Query(
        ...,
        title="proportion of spotlight",
        description="This is the proportion of the desired spotlight",
        example="0.15"
    )
):
    file_video_ids = {f'{video_id}.mp4': video_id}
    file_summary_ids = {f'{summary_id}.pk': summary_id}
    correct_video = GAPI.download_file(file_video_ids, PATH_DATA)
    correct_summary = GAPI.download_file(file_summary_ids, PATH_DATA)

    dict_summary = pickle.load(open(os.path.join(PATH_DATA, f'{summary_id}.pk'), 'rb'))
    summary = dict_summary['summary']
    change_points = dict_summary['change_points']
    n_frames = dict_summary['n_frames']
    n_frame_per_seg = dict_summary['n_frame_per_seg']
    picks = dict_summary['picks']
    os.remove(os.path.join(PATH_DATA, f'{summary_id}.pk'))

    dur_spotlight, n_segments = vsm.generate_summary_proportion(os.path.join(PATH_DATA, f'{video_id}.mp4'),
                                        summary, change_points, n_frames, n_frame_per_seg, picks, proportion,
                                        os.path.join(PATH_DATA, f'{video_id}_spl.mp4'))
    os.remove(os.path.join(PATH_DATA, f'{video_id}.mp4'))

    file_spotlight_ids = GAPI.upload_files([f'{video_id}_spl.mp4'], PATH_DATA, PARENT_FOLDERID_SPOTLIGHT_OUTPUT)
    spotlight_id = file_spotlight_ids[f'{video_id}_spl.mp4']
    os.remove(os.path.join(PATH_DATA, f'{video_id}_spl.mp4'))

    summary_response = SummaryresponseBase(dur_spotlight=dur_spotlight, n_segments=n_segments, spotlight_id=spotlight_id)
    return summary_response




@app.get(
    path="/download-spotlight",
    status_code=status.HTTP_200_OK
)
async def download_spotlight(
    background_tasks: BackgroundTasks,
    spotlight_id: str = Query(
        ...,
        title="Id spotlight",
        description="This is the spotlight id",
        example="spotiij123"
    ),
):  
    def remove_file(path: str):
        os.unlink(path)
 
    file_spotlight_ids = {f'{spotlight_id}.mp4': spotlight_id}

    stream = GAPI.download_file(file_spotlight_ids, PATH_DATA)
    spotlight_path = os.path.join(PATH_DATA, f'{spotlight_id}.mp4')
    
    file_response = FileResponse(spotlight_path)
    background_tasks.add_task(remove_file, spotlight_path)
    return file_response
