import numpy as np
from numpy.linalg import inv
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import os
import imutils
import glob


def createChessboard():
    # get camera with src=0
    cap = cv2.VideoCapture(0)
    save_path = r'.\chessboard'
    index = 1
    while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(os.path.join(save_path,'chessboard_%d.jpg'%(index)),frame)
            print('%s saved'%(os.path.join(save_path,'chessboard_%d.jpg'%(index))))
            index += 1
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
createChessboard()
