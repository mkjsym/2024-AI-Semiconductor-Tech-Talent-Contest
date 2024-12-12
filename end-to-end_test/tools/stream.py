import os
import subprocess
import sys
import time
import json
import cv2
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

HOME_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(HOME_DIR)
from utils.parse_params import get_output_paths
from utils.result_img_process import ImageMerger

app = FastAPI()
CFG = sys.argv[1]
MERGER = ImageMerger()

def read_tracking_data():
    # tracking_results 디렉토리 내에 있는 tracking_data.json 파일을 읽음
    tracking_file_path = os.path.join(HOME_DIR, "tracking_results", "tracking_data.json")
    with open(tracking_file_path, "r") as file:
        tracking_data = json.load(file)
    return tracking_data

@app.get("/")
async def stream():
    def getByteFrame():
        output_paths = get_output_paths(CFG)  # 여기서 yaml 읽어서 파일 경로 알아낸 후

        for frame in MERGER(output_paths, full_grid_shape=(720, 1280)):
            time.sleep(0.01)
            ret, out_img = cv2.imencode(".jpg", frame)
            out_frame = out_img.tobytes()

            # 이미지와 함께 tracking_data를 함께 전송
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n"
                )

    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/tracking-data")
async def get_tracking_data():
    # tracking_data.json 파일을 읽어 JSON 형식으로 반환
    tracking_data = read_tracking_data()
    # 로그로 출력
    return Response(
        content=json.dumps(tracking_data, indent=2), 
        media_type="application/json"
    )

if __name__ == "__main__":
    uvicorn.run(app="stream:app", host="0.0.0.0", port=20001, reload=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)