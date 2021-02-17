import cv2
import numpy as np


def Isolate_Red(img):
    import cv2
    import numpy as np

    # img = cv2.imread("3.jpg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 10, 50])
    upper_red = np.array([5, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    # cv2.imshow("Output", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    return output_img


def Count_Balls(img):
    img = cv2.resize(img, (700, 700))
    img = cv2.subtract(255, img)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10;
    params.maxThreshold = 250;

    params.filterByArea = True
    params.minArea = 400
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

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        RedBallImg = Isolate_Red(frame)
        gray = cv2.cvtColor(RedBallImg, cv2.COLOR_BGR2GRAY)
        Count_Balls(gray)
        cap.release()
        cv2.destroyAllWindows()

    elif key == ord("q"):
        break

else:
    cap.release()

cv2.destroyAllWindows()