"""
    Abdul Rahman Dabbour
    Sabanci University
    Faculty of Engineering and Natural Sciences
    Cognitive Robotics Laboratory
"""

"""
    General Hand Gesture Recoginition using OpenCV
    Details: https://github.com/thedabbour/hand-gesture-recognition/README.md
"""

import cv2
import numpy as np

CAP = cv2.VideoCapture(0)

while(True):
    # capture frame-by-frame
    RET, FRAME = CAP.read()


    # Set frame to mirror mode
    if RET == True:
        FRAME = cv2.flip(FRAME, 1)
        HEIGHT, WIDTH = FRAME.shape[:2]

        HEIGHT = float(HEIGHT)
        HEIGHT = float(WIDTH)

        H1 = int((HEIGHT/3.0) - ((HEIGHT/3.0)*0.5))
        W1 = int(((2*WIDTH)/3.0) - ((WIDTH/3.0)*0.5))
        W2 = int(((2*WIDTH)/3.0) + ((WIDTH/3.0)*0.5))
        H2 = int(H1 + 1.5*(W2-W1))

        print(H1, H2, W1, W2)

        ROI = FRAME[H1:H2, W1:W2]
        CANNY = cv2.Canny(ROI, 100, 200)

        FRAME = cv2.rectangle(FRAME, (W1, H1), (W2, H2), (255, 0, 0), 2)

        # Display the resulting FRAMEs
        cv2.imshow('Camera Output', FRAME)
        cv2.imshow('Canny Edge Detection', CANNY)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
CAP.release()
cv2.destroyAllWindows()
