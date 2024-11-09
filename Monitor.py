import cv2
import numpy as np
import mmap

# 공유 메모리 크기 (RGBA 이미지의 크기: 1280 * 720 * 4바이트 * 4개 카메라)
image_width = 1280
image_height = 720
bytes_per_pixel = 4  # RGBA 형식이므로 4바이트
shared_memory_size = image_width * image_height * bytes_per_pixel * 4  # 4개의 카메라 데이터 포함

# 공유 메모리에서 데이터를 읽어오는 함수 (offset 기반)
def read_from_shared_memory(memory_name, offset, size):
    # Windows에서 태그 기반으로 메모리 매핑
    mm = mmap.mmap(0, shared_memory_size, tagname=memory_name)
    mm.seek(offset)  # 지정된 offset 위치로 이동
    image_data = mm.read(size)  # 지정된 크기만큼 데이터 읽기
    mm.close()
    
    # 읽은 데이터를 NumPy 배열로 변환
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image_np = image_np.reshape((image_height, image_width, 4))  # RGBA 포맷
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)  # BGRA -> BGR 변환
    image_np = cv2.flip(image_np, 0)  # 상하 반전
    
    return image_np

# 네 개의 카메라 영상을 합쳐서 한 화면에 출력하는 함수
def display_combined_camera_feeds():
    while True:
        # 각 카메라 이미지 읽기 (오프셋과 크기를 사용하여 각각의 카메라 데이터 읽기)
        cam1_image = read_from_shared_memory("CameraSharedMemory", 0, image_width * image_height * bytes_per_pixel)
        cam2_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * bytes_per_pixel, image_width * image_height * bytes_per_pixel)
        cam3_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * bytes_per_pixel * 2, image_width * image_height * bytes_per_pixel)
        cam4_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * bytes_per_pixel * 3, image_width * image_height * bytes_per_pixel)
        
        # 각 카메라 이미지를 640x360으로 리사이즈
        cam1_resized = cv2.resize(cam1_image, (640, 360))
        cam2_resized = cv2.resize(cam2_image, (640, 360))
        cam3_resized = cv2.resize(cam3_image, (640, 360))
        cam4_resized = cv2.resize(cam4_image, (640, 360))

        # 상단, 하단 각각 두 개의 영상을 합치기
        top_row = np.hstack((cam1_resized, cam2_resized))
        bottom_row = np.hstack((cam3_resized, cam4_resized))

        # 상단과 하단을 세로로 합쳐서 전체 화면 구성
        combined_image = np.vstack((top_row, bottom_row))

        # 결합된 화면 출력
        cv2.imshow('Combined Camera Feeds', combined_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# 프로그램 실행
if __name__ == "__main__":
    display_combined_camera_feeds()
