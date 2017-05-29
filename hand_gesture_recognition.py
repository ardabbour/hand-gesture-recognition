"""
    Abdul Rahman Dabbour
    Sabanci University
    Faculty of Engineering and Natural Sciences
    Cognitive Robotics Laboratory

    General Hand Gesture Recoginition using OpenCV
    Details: https://github.com/thedabbour/hand-gesture-recognition/README.md/
"""

import cv2
import numpy as np

# Create a flag to indicate when the application should start working
FLAG = False

# Create camera source as webcam
CAP = cv2.VideoCapture(0)

# Read first frame.
OK, FRAME = CAP.read()

# Define an initial region of interest (ROI) based on frame size
HEIGHT, WIDTH = FRAME.shape[:2]
HEIGHT = float(HEIGHT)
WIDTH = float(WIDTH)

X = int(((2*WIDTH)/3.0) - ((WIDTH/3.0)*0.5))
Y = int((HEIGHT/3.0) - ((HEIGHT/3.0)*0.5))
W = int((WIDTH/3.0))
H = int(1.5*W)

ROI = (X, Y, W, H)


# Create and initialize Kernalized Correlation Filters tracker around ROI
# More information on KCF object tracker: http://www.robots.ox.ac.uk/~joao/circulant/
TRACKER = cv2.Tracker_create('KCF')
OK = TRACKER.init(FRAME, ROI)

# Define approximate hue, saturation and value range for skin color
# HSV values for skin taken from http://pyimagesearch.com/
LOWER = np.array([0, 48, 80], dtype="uint8")
UPPER = np.array([20, 255, 255], dtype="uint8")

while True:
    # Capture camera input frame-by-frame
    RET, FRAME = CAP.read()

    if RET:
        # Set frame to mirror mode
        FRAME = cv2.flip(FRAME, 1)

        # Begin hand gesture recognition
        if FLAG:
            # Convert camera colorspace to HSV
            CONVERTED = cv2.cvtColor(FRAME, cv2.COLOR_BGR2HSV)

            # Filter image to show HSV values of skin only
            SKINMASK = cv2.inRange(CONVERTED, LOWER, UPPER)

            # Create structuring element for morphological operations
            KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

            # Erode and dilate to remove small misidentified blobs
            SKINMASK = cv2.erode(SKINMASK, KERNEL, iterations=2)
            SKINMASK = cv2.dilate(SKINMASK, KERNEL, iterations=2)

            # Filter with Gaussian to smooth and remove noise
            SKINMASK = cv2.GaussianBlur(SKINMASK, (3, 3), 0)

            # Show only areas of original image with skin
            SKIN = cv2.bitwise_and(FRAME, FRAME, mask=SKINMASK)

            OK, ROI_TRACKER = TRACKER.update(FRAME)
            if OK:
                X = int(ROI_TRACKER[0])
                Y = int(ROI_TRACKER[1])
                W = int(ROI_TRACKER[2])
                H = int(ROI_TRACKER[3])

                W1 = X
                W2 = X + W
                H1 = Y
                H2 = Y + H

            # Create mask to remove everything but ROI from image
            AREAMASK = np.zeros(SKIN.shape, np.uint8)
            AREAMASK[Y:Y + H, X:X + W] = SKIN[Y:Y + H, X:X + W]

            # Apply Canny edge detector to the mask
            CANNY = cv2.Canny(AREAMASK, 100, 200)

            IM2, CONTRS, HRRCHY = cv2.findContours(CANNY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            KERNEL = np.ones((5, 5), np.uint8)

            MORPH = cv2.erode(CANNY, KERNEL, iterations=2)
            MORPH = cv2.dilate(CANNY, KERNEL, iterations=2)

            cv2.drawContours(FRAME, CONTRS, -1, (0, 255, 0), 3)

            # Draw a rectangle around the ROI
            cv2.rectangle(FRAME, (W1, H1), (W2, H2), (255, 0, 0), 2)

            # For diagnostics, show different outputs
            # cv2.imshow('Canny Output', CANNY)
            # cv2.imshow('Area-Masked Output', AREAMASK)
            # cv2.imshow('Color-Masked Output', SKIN)

        # Prepare for start/Reset
        else:
            # Define a region of interest (ROI)
            HEIGHT, WIDTH = FRAME.shape[:2]
            HEIGHT = float(HEIGHT)
            WIDTH = float(WIDTH)

            X = int(((2*WIDTH)/3.0) - ((WIDTH/3.0)*0.5))
            Y = int((HEIGHT/3.0) - ((HEIGHT/3.0)*0.5))
            W = int((WIDTH/3.0))
            H = int(1.5*W)

            W1 = X
            W2 = X + W
            H1 = Y
            H2 = Y + H

            # Draw a rectangle around the ROI
            cv2.rectangle(FRAME, (W1, H1), (W2, H2), (255, 0, 0), 2)

        # Display the original image with tracking
        cv2.imshow('Camera Input', FRAME)

    INTERRUPT = cv2.waitKey(1)

    # Press 's' to start hand gesture recognition
    if INTERRUPT & 0xFF == ord('s'):
        FLAG = True

    # Press 'e' to end hand gesture recognition
    if INTERRUPT & 0xFF == ord('e'):
        FLAG = False

    # Press 'q' to quit the program
    if INTERRUPT & 0xFF == ord('q'):
        break

# When everything is done, release the camera and close all windows
CAP.release()
cv2.destroyAllWindows()
