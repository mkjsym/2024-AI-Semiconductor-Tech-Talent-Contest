import cv2
import os, subprocess, asyncio
import time
import threading

from utils.preprocess import YOLOPreProcessor
from utils.postprocess import ObjDetPostprocess
from furiosa.runtime.sync import create_runner
from furiosa.runtime import create_queue

#Using Runners#########################################################################################################################################################
model_path = r"/home/ubuntu/ym/2024-AI-Semiconductor-Tech-Talent-Contest/object_detection/yolov8n_opt_i8.onnx"
# data_dir = "../val2017"
# data_name = os.listdir(data_dir)[0]

if os.path.exists("result"):
    subprocess.run(["rm", "-rf", "result"])
os.makedirs("result")

#Using Runners
# def furiosa_runtime_sync(model_path, input_img):
#     with create_runner(model_path, device = "warboy(2)*1") as runner:
#         preds = runner.run([input_img]) # FuriosaAI Runtime
        
#         return preds

#use mp4 video as source
# cap = cv2.VideoCapture('rtsp://58.142.226.112:8554/mystream')
cap = cv2.VideoCapture(r'/home/ubuntu/ym/2024-AI-Semiconductor-Tech-Talent-Contest/object_detection/fire.mp4')
preprocessor = YOLOPreProcessor()
postprocessor = ObjDetPostprocess()
runner = create_runner(model_path, device = "warboy(2)*2")

while cap.isOpened():
    success, frame = cap.read()
    if (not success):
        print('failed to read video')
        break

    input_, contexts = preprocessor(frame, new_shape=(640, 640), tensor_type="uint8")

    #Using Runners
    # result = furiosa_runtime_sync(model_path, input_)
    result = runner.run([input_])
    # output_img = postprocessor(result, contexts, input_)

    print(result)

cap.release()

#Using Queues#########################################################################################################################################################
# model_path = r"/home/ubuntu/ym/2024-AI-Semiconductor-Tech-Talent-Contest/object_detection/yolov8n_opt_i8.onnx"
# data_dir = "../val2017"
# data_name = os.listdir(data_dir)[0]

# if os.path.exists("result"):
#     subprocess.run(["rm", "-rf", "result"])
# os.makedirs("result")

# #Using Queues
# async def submit_with(submitter, input_, contexts):
#     for _ in range(1000):
#         await submitter.submit(input_, context=(contexts))

# async def recv_with(receiver, input_img, data_name):
#     postprocessor = ObjDetPostprocess()
#     for _ in range(1000):
#         contexts, outputs = await receiver.recv()
#         output_img = postprocessor(outputs, contexts, input_img)
#         cv2.imwrite(os.path.join("result", data_name), output_img)

# async def furiosa_runtime_queue(model_path, input_img, input_, contexts, data_name):

#     async with create_queue(
#         model=model_path, worker_num=8, device="warboy(2)*1"
#     ) as (submitter, receiver):
#         submit_task = asyncio.create_task(submit_with(submitter, input_, contexts))
#         recv_task = asyncio.create_task(recv_with(receiver, input_img, data_name))
#         await submit_task
#         await recv_task

# #use mp4 video as source
# cap = cv2.VideoCapture('rtsp://58.142.226.112:8554/mystream')
# preprocessor = YOLOPreProcessor()

# while cap.isOpened():
#     success, frame = cap.read()
#     if (not success):
#         print('failed to read video')
#         continue

#     input_, contexts = preprocessor(frame, new_shape=(640, 640), tensor_type="uint8")

#     #Using Queues
#     result = asyncio.run(furiosa_runtime_queue(model_path, frame, input_, contexts, data_name))

#     print(result)

# cap.release()
