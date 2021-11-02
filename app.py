import os
import shutil

from fastapi import FastAPI
from fastapi import Body, Query, UploadFile, File
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import tempfile

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    path="/summarize-video",
    status_code=status.HTTP_201_CREATED
)
def summarize_video(
    video: UploadFile = File(...)
):
    with open(os.path.join(tempfile.gettempdir(), video.filename), "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    xd = cv2.VideoCapture(os.path.join(tempfile.gettempdir(), video.filename))
    succes, img_xd = xd.read()
    cv2.imwrite("dummy.jpg", img_xd)
    xd.release()
    #temp_filename = os.path.join(tempfile.gettempdir(), video.filename)
    #local_temp_file = open(temp_filename, "w+")
    #video.file.seek(0)
    #local_temp_file.write(video.file.read())
    #local_temp_file.close()

    #contents = video.file.read()
    #cap = cv2.VideoCapture(contents.name)
    #cap = cv2.VideoCapture(local_temp_file.name)
    os.remove(os.path.join(tempfile.gettempdir(), video.filename))

    return {
        "filename": video.filename,
        "format": video.content_type,
        "path": os.path.join(tempfile.gettempdir(), video.filename)
    }


@app.get(
    path="/get-spotlight",
    status_code=status.HTTP_200_OK
)
async def get_spotlight(
    video_name: str = Query(
        ...,
        title="Name of the video",
        description="This is the video name",
        example="output.mp4"
    )
):
    path = "./"
    video_path = os.path.join(path, video_name)
    file_response = FileResponse(video_path)
    os.remove("dummy.jpg")
    return file_response