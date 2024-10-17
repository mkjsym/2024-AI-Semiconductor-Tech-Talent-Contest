from submodules import *

img_path = 'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-AI-Semiconductor-Tech-Talent-Contest/sfm/data/nutellar2/'

img1_name = 'nutella13.jpg'
img2_name = 'nutella14.jpg'

img1, img2 = load_image(img_path, img1_name, img2_name)

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while cap1.isOpened() and cap2.isOpened():
    success1, image1 = cap1.read()
    success2, image2 = cap2.read()

    if ((not success1) or (not success2)):
        print("Webcam Mode")
        continue

    matches_good, img1_kp, img2_kp = SIFT(img1, img2)
    E, p1_inlier, p2_inlier = Estimation_E(matches_good, img1_kp, img2_kp)
    CameraMatrix = EM_Decomposition(E, p1_inlier, p2_inlier)
    Rt0, Rt1 = initialize_CM(CameraMatrix)
    p1, p2 = rescale_point(p1_inlier, p2_inlier, len(p1_inlier))
    point3d = make_3dpoint(p1, p2, Rt0, Rt1)

    visualize_3d(point3d)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
