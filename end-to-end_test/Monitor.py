import cv2
import numpy as np
import mmap
import ffmpeg
import threading
import signal
import time
import logging

logging.basicConfig(level=logging.DEBUG)
image_width = 1024
image_height = 576
bytes_per_pixel = 4  # RGBA 형식이므로 4바이트
shared_memory_size = image_width * image_height * bytes_per_pixel  # 4개의 카메라 데이터 포함

rtmp_url = "rtmp://14.39.59.81/live/"
frame_rate = 30
fourcc = cv2.VideoWriter_fourcc(*'X264')

running = True
def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
lock = threading.Lock()
# 공유 메모리에서 데이터를 읽어오는 함수 (offset 기반)
def read_from_shared_memory(memory_name, size):
    mm = mmap.mmap(0, shared_memory_size, tagname=memory_name)
    image_data = mm.read(size)  # 지정된 크기만큼 데이터 읽기
    mm.close()
    
    # 읽은 데이터를 NumPy 배열로 변환
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image_np = image_np.reshape((image_height, image_width, 4))  # RGBA 포맷
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)  # BGRA -> BGR 변환
    image_np = cv2.flip(image_np, 0)  # 상하 반전
    
    return image_np

def increase_brightness(image, value=100):
    image = np.clip(image + value, 0, 255)
    return image

def camera_stream_worker(camera_id, rtmp_url):
    attempt = 0  # 스트림 시도 횟수
    max_attempts = 5  # 최대 재시도 횟수
    while attempt < max_attempts:
        try:
            # 스트리밍을 위한 ffmpeg 프로세스 시작
            process = (
                ffmpeg
                .input("pipe:0", framerate=30, format="rawvideo", pix_fmt="bgr24", s="1024x576")
                .output(rtmp_url, format="flv", vcodec="libx264", preset="fast", crf=28, video_bitrate=1000)
                .run_async(pipe_stdin=True)
            )#여기서 s에 들어가는 문자열에 영상 사이즈로 바꿔야함
            
            mmname = f"CameraSharedMemory_{camera_id}"
            print(mmname)
            logging.debug(f"Starting stream for {mmname} to {rtmp_url}")

            #cap = cv2.VideoCapture(rf"C:\Users\ykh45\cam{camera_id+1}.avi")=====================================
            while running:
                # 공유 메모리에서 이미지 데이터 읽기
                with lock:
                    image = read_from_shared_memory(mmname, image_width * image_height * bytes_per_pixel)
                    # bright_image = increase_brightness(image)  # 밝기 증가 (구현 필요)

                # 비디오 파일에서 한 프레임 읽기====================
                #_, bright_image = cap.read()================================
                # if not _:==================================
                #     logging.warning(f"Failed to read frame from camera {camera_id}, retrying...")=============================
                #     break  # 프레임 읽기 실패 시 스트림을 중단하고 재시도==============================

                process.stdin.write(image.tobytes())  # 이미지 데이터를 스트리밍
                time.sleep(0.03)
            
            process.stdin.close()
            process.wait()

            # 성공적으로 스트리밍을 마쳤다면, 반복문을 종료
            logging.debug(f"Successfully streamed from camera {camera_id}")
            break  # 스트리밍 성공 후 루프 탈출

        except Exception as e:
            # 예외 발생 시 로그 기록 및 재시도
            logging.error(f"Error with camera {camera_id} streaming: {e}")
            attempt += 1
            logging.info(f"Attempt {attempt} of {max_attempts} failed. Retrying in 5 seconds...")
            time.sleep(0.1)  # 재시도 전 대기
            if attempt >= max_attempts:
                logging.error(f"Maximum attempts reached for camera {camera_id}. Giving up.")

if __name__ == "__main__":
    camera_threads = []
    rtmp_urls = [f"{rtmp_url}{i:04d}" for i in range(4)] 
    """=====================================================
    중요           여기 4가 스트림 할 화면 수임 최대 4개^
    ======================================================
    위에 동작부 중에 ===========붙은 부분은 저장된 영상으로 테스트 하는 용도임 시뮬레이터로 할거면 제거
    시뮬레이터와 공유 메모리로 주고 받는건 윈도우에서만 됨
    """
    for i in range(4):
        t = threading.Thread(target=camera_stream_worker, args=(i, rtmp_urls[i]))
        t.start()
        time.sleep(0.1)
        camera_threads.append(t)

    for t in camera_threads:
        t.join()
