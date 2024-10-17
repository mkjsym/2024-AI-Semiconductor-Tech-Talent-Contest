import cv2
from submodules import *

def sfm():
    img_path = 'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-AI-Semiconductor-Tech-Talent-Contest/sfm/data/nutellar2/'

    img1_name = 'nutella13.jpg'
    img2_name = 'nutella14.jpg'

    img1, img2 = load_image(img_path, img1_name, img2_name)

    matches_good, img1_kp, img2_kp = SIFT(img1, img2)
    E, p1_inlier, p2_inlier = Estimation_E(matches_good, img1_kp, img2_kp)
    CameraMatrix = EM_Decomposition(E, p1_inlier, p2_inlier)
    Rt0, Rt1 = initialize_CM(CameraMatrix)
    p1, p2 = rescale_point(p1_inlier, p2_inlier, len(p1_inlier))
    point3d = make_3dpoint(p1, p2, Rt0, Rt1)

    visualize_3d(point3d)

#웹캠 두개를 이용해서 sfm을 수행하는 코드
def sfm_webcam():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    while cap1.isOpened() and cap2.isOpened():
        success1, img1 = cap1.read()
        success2, img2 = cap2.read()

        if ((not success1) or (not success2)):
            print("Webcam Mode")
            continue

        img_concat = np.hstack((img1, img2))

        cv2.imshow('Webcam', img_concat)

        #ESC를 눌러 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

#시뮬레이터를 사용하여 네 개의 카메라 이미지에 대해서 sfm을 수행하는 함수
def sfm_sim():

    while True:
        # 각 카메라 이미지 읽기 (오프셋과 크기를 사용하여 각각의 카메라 데이터 읽기)
        cam1_image = read_from_shared_memory("CameraSharedMemory", 0, image_width * image_height * 3)
        cam2_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * 3, image_width * image_height * 3)
        cam3_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * 3 * 2, image_width * image_height * 3)
        cam4_image = read_from_shared_memory("CameraSharedMemory", image_width * image_height * 3 * 3, image_width * image_height * 3)

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
    sfm()
    # sfm_webcam()
    # sfm_sim()
