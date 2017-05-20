"""
    Abdul Rahman Dabbour
    Sabanci University
    Faculty of Engineering and Natural Sciences
    Cognitive Robotics Laboratory

    General Hand Gesture Recoginition using OpenCV
    Details: https://github.com/thedabbour/hand-gesture-recognition/README.md
"""

import cv2
import numpy as np

FLAG = False

CAP = cv2.VideoCapture(0)
TRACKER = cv2.Tracker_create('KCF')

# Try this for now
LOWER = np.array([0, 48, 80], dtype="uint8")
UPPER = np.array([20, 255, 255], dtype="uint8")
################################################

while True:
    # Capture camera input frame-by-frame
    RET, FRAME = CAP.read()

    if RET:
        # Set frame to mirror mode
        FRAME = cv2.flip(FRAME, 1)

        # Define a region of interest (ROI)
        if not FLAG:
            HEIGHT, WIDTH = FRAME.shape[:2]
            HEIGHT = float(HEIGHT)
            WIDTH = float(WIDTH)

            X = int(((2*WIDTH)/3.0) - ((WIDTH/3.0)*0.5))
            Y = int((HEIGHT/3.0) - ((HEIGHT/3.0)*0.5))
            W = int((WIDTH/3.0))
            H = int(1.5*W)

        if FLAG:
            OK, ROI_TRACKER = TRACKER.update(FRAME)
            if OK:
                X = int(ROI_TRACKER[0])
                Y = int(ROI_TRACKER[1])
                W = int(ROI_TRACKER[2])
                H = int(ROI_TRACKER[3])

        # Color-based filtering
        CONVERTED = cv2.cvtColor(FRAME, cv2.COLOR_BGR2HSV)
        SKINMASK = cv2.inRange(CONVERTED, LOWER, UPPER)

        KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        SKINMASK = cv2.erode(SKINMASK, KERNEL, iterations=2)
        SKINMASK = cv2.dilate(SKINMASK, KERNEL, iterations=2)

        SKINMASK = cv2.GaussianBlur(SKINMASK, (3, 3), 0)
        SKIN = cv2.bitwise_and(FRAME, FRAME, mask=SKINMASK)

        W1 = X
        W2 = X + W
        H1 = Y
        H2 = Y + H

        ROI = FRAME[H1:H2, W1:W2]

        # Create a mask that covers everything but the ROI
        MASK = np.zeros(FRAME.shape, np.uint8)
        MASK[H1:H2, W1:W2] = FRAME[H1:H2, W1:W2]

        # Apply Canny edge detector to the mask
        CANNY = cv2.Canny(MASK, 100, 200)

        cv2.rectangle(FRAME, (W1, H1), (W2, H2), (255, 0, 0), 2)

        IM2, CONTOURS, HIERARCHY = cv2.findContours(CANNY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        KERNEL_MOD = np.ones((5, 5), np.uint8)
        MORPH = cv2.erode(CANNY, KERNEL, iterations=2)
        MORPH = cv2.dilate(CANNY, KERNEL, iterations=2)

        MORPH_MOD = cv2.erode(CANNY, KERNEL_MOD, iterations=2)
        MORPH_MOD = cv2.dilate(CANNY, KERNEL_MOD, iterations=2)

        cv2.drawContours(FRAME, CONTOURS, -1, (0, 255, 0), 3)

        # Display the resulting frames
        cv2.imshow('Camera Output', FRAME)
        # cv2.imshow('Canny Edge Detection', CANNY)
        cv2.imshow('Dilated and Eroded Canny', MORPH)
        cv2.imshow('Dilated and Eroded Canny with simple', MORPH_MOD)
        # cv2.imshow("Skin Filter", SKIN)
        # cv2.imshow('Contours', IM2)
        # cv2.imshow("images", np.hstack([FRAME, MORPH, MORPH_MOD]))

    INTERRUPT = cv2.waitKey(1)

    if INTERRUPT & 0xFF == ord('s'):
        TRACKER.init(FRAME, (X, Y, W, H))
        FLAG = True

    if INTERRUPT & 0xFF == ord('f'):
        FLAG = False

    if INTERRUPT & 0xFF == ord('q'):
        break

# When everything done, release the capture
CAP.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# img = cv2.imread('input.png', 0)

# kernel = np.ones((5,5), np.uint8)

# img_erosion = cv2.erode(img, kernel, iterations=1)
# img_dilation = cv2.dilate(img, kernel, iterations=1)

# cv2.imshow('Input', img)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)

# cv2.waitKey(0)
