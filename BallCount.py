import cv2
import numpy as np


def Count_Balls(img):
    img = cv2.resize(img, (700, 700))
    img = cv2.subtract(255, img)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10;
    params.maxThreshold = 250;

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 5000

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.4

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    print('Number of Balls in frame: {}' .format(len(keypoints)))

    imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", imgKeyPoints)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


cap = cv2.VideoCapture('balls.mp4')

while cap.isOpened():

    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        Count_Balls(gray)
        cap.release()
        cv2.destroyAllWindows()

    elif key == ord("q"):
        break

else:
    cap.release()

cv2.destroyAllWindows()
