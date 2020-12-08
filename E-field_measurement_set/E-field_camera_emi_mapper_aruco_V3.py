# import imutils #pip3 install imutils
import time
import cv2 #sudo apt install opencv-data opencv-doc python-opencv && pip3 install opencv-contrib-python
from cv2 import aruco
import PIL
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import argparse
import datetime
from imutils.video import WebcamVideoStream
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
import serial
import glob
import os


#Inferno colormap data
cm_data = [[0.002810891, 0.00239715, 0.013984976],[0.009578015, 0.00418497, 0.039273895],[0.021372785, 0.010229468, 0.073098944],
[0.039013179, 0.017535454, 0.107679598],[0.060479342, 0.025103492, 0.143493094],[0.083077128, 0.031730573, 0.180633309],
[0.107150529, 0.036069744, 0.218930917],[0.133061474, 0.036665832, 0.257672709],[0.160538307, 0.03289627, 0.295456813],
[0.188894276, 0.025561629, 0.330141008],[0.217088714, 0.017179794, 0.359591651],[0.244283768, 0.010726816, 0.382809196],
[0.270205887, 0.008037756, 0.400177892],[0.295020185, 0.009521928, 0.412778955],[0.319024021, 0.014831967, 0.421732459],
[0.342509393, 0.02335207, 0.427919965],[0.365696203, 0.034505352, 0.431968781],[0.388740909, 0.047346263, 0.434295378],
[0.411745277, 0.059771453, 0.435174425],[0.434782328, 0.07160348, 0.43476524],[0.457891095, 0.082910695, 0.433160264],
[0.481089526, 0.093787482, 0.430415748],[0.504382524, 0.104330222, 0.426542446],[0.527760369, 0.114651181, 0.421550408],
[0.551200328, 0.124857904, 0.415434358],[0.574670764, 0.135061199, 0.408189444],[0.598129715, 0.145376544, 0.399818633],
[0.621527609, 0.15592399, 0.390325042],[0.644805559, 0.166824566, 0.3797311],[0.667893652, 0.178203136, 0.368061022],
[0.690715703, 0.19018666, 0.35536275],[0.7131868, 0.202897315, 0.341683912],[0.735215541, 0.21645397, 0.327095723],
[0.756703173, 0.23096321, 0.3116759],[0.77754886, 0.246520816, 0.295501632],[0.797646257, 0.263201353, 0.278662025],
[0.816894047, 0.281058794, 0.261238427],[0.835193896, 0.300119974, 0.243305408],[0.852451408, 0.320388885, 0.224916259],
[0.868586465, 0.341843925, 0.206100782],[0.88352864, 0.364441708, 0.186857484],[0.89721913, 0.38812106, 0.167146173],
[0.909613336, 0.41280737, 0.146884076],[0.920673649, 0.438427088, 0.125946831],[0.930377784, 0.464893517, 0.104148782],
[0.938706536, 0.492124283, 0.081251836],[0.945642636, 0.520044107, 0.056978798],[0.951174041, 0.548584852, 0.031562004],
[0.95528605, 0.577681957, 0.012693898],[0.957963333, 0.607275215, 0.004180212],[0.959191151, 0.637313882, 0.008003995],
[0.958954209, 0.667745484, 0.026832268],[0.957243294, 0.698517135, 0.06073042],[0.954056878, 0.72957236, 0.097705539],
[0.949412109, 0.760847757, 0.136596414],[0.943392595, 0.792245511, 0.177940744],[0.936153616, 0.823638485, 0.222443168],
[0.928075941, 0.854808259, 0.270954189],[0.919955705, 0.885382504, 0.324472473],[0.913435751, 0.914708557, 0.383922739],
[0.911593617, 0.941717451, 0.449013013],[0.918232238, 0.965311849, 0.516542561],[0.934816618, 0.985396212, 0.581365496],[0.959399507, 1, 0.640626478]]

# Convert a colormap to BRG for openCV
for s in cm_data:
	s[0], s[-1] = s[-1], s[0]
inferno_map = LinearSegmentedColormap.from_list('inferno', cm_data)

#Connecting to the arduino board
ser = serial.Serial('COM4',57600)
time.sleep(2)
print('E-field probe is connected')
ser.flushInput()

# parse args
parser = argparse.ArgumentParser(description='E-field mapping with E-field probe and webcam.')
parser.add_argument('-c', '--camera', type=int, help='camera id (default=0)',default=0)
args = parser.parse_args()

# Blur filter is currently disabled
def gaussian_with_nan(U, sigma=0):
    """Computes the gaussian blur of a numpy array with NaNs.
    """
    np.seterr(divide='ignore', invalid='ignore')
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma)

    return VV/WW

#This function using images with the chessboard to calibrate camera coordinates
def cameraCalibration():
    img_path = r'.\chessboard'

    objp = np.zeros((9*6,3),dtype=np.float32)
    # 22mm for chessboard
    objp[:,:2]=22*np.mgrid[:9,:6].T.reshape(-1,2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

    objPoints = []
    imgPoints=[]

    imgs = glob.glob(os.path.join(img_path,'*.jpg'))
    for img in imgs:
        img = cv2.imread(img)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray,(9,6))

        if ret == True:
            objPoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgPoints.append(corners2)

            img = cv2.drawChessboardCorners(img,(9,6),corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    ret,mtx,dist,rvect,tvect = cv2.calibrateCamera(objPoints,imgPoints,gray.shape[::-1],None,None)
    return ret,mtx,dist,rvect,tvect

#Read value from the E-field probe
def get_Value(signal):
    ser_bytes = ser.readline()
    signal = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    return signal

def main():
    print("Starting capture")
    # read from specified webcam
    vs = WebcamVideoStream(src=args.camera).start()

    # initialize variables
    powermap = None
    firstFrame = None
    
    # interpolate colormap from 64 to 256 elements
    cm_data_256 = np.array(cm_data)
    x_64 = np.linspace(0, 1, num=cm_data_256.shape[0], endpoint=True)
    x_256 = np.linspace(0, 1, num=256, endpoint=True)
    interp_func = interp1d(x_64,cm_data,axis=0)
    cm_data_256 = np.round(255*interp_func(x_256)).astype(np.uint8)

    '''
    # comment by Kaixuan
    # # Init OpenCV object tracker objects
    # tracker = cv2.TrackerCSRT_create()
    # init_tracking_BB = None
    '''
    # init aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    BB_geometry = np.array([4,4])
    ret,mtx,dist,rvect,tvect = cameraCalibration()

    frame = vs.read()
    signal = 0.0
    
    # if the first frame is None, initialize it
    firstFrame = frame
    powermap = np.empty((len(frame),len(frame[0])))
    powermap.fill(np.nan)

    powermap_3ch = np.empty((len(frame),len(frame[0]),3),dtype=np.uint8)
    powermap_3ch.fill(np.nan)
    
    #Live colormap level settings
    min_window = 0.0001
    max_window = 0.25
    
    #An array with the data
    records=[]

    while True:

        frame = vs.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame,aruco_dict,parameters=parameters)
        
        if corners !=[]:
            rvec,tvec,_ = aruco.estimatePoseSingleMarkers(corners,0.028,mtx,dist)
            corners = np.squeeze(corners)
            x_center = np.mean(corners[:,1])
            y_center = np.mean(corners[:,0])

            # fill map
            power = get_Value(signal)
            powermap[int(x_center-BB_geometry[0]/2):int(x_center+BB_geometry[0]/2),int(y_center-BB_geometry[1]/2):int(y_center+BB_geometry[1]/2)] = power          

            # sturation limits for the live colormap (here also some patch for the wrong values from ADC)
            if power<min_window:
                power = min_window
            
            if power>4:
                continue
            if power>max_window:
                power = max_window
                
            
            #Write data to the file
            temp_record = np.append(np.squeeze(tvec)[0:2],power)
            
            #Live colormap
            power_adj = int(np.round(255*(power - min_window)/(max_window - min_window)))
            power_patch = (np.multiply(cm_data_256[power_adj,:],np.ones((BB_geometry[0],BB_geometry[1],3)))).astype(np.uint8)
            powermap_3ch[int(x_center-BB_geometry[0]/2):int(x_center+BB_geometry[0]/2),int(y_center-BB_geometry[1]/2):int(y_center+BB_geometry[1]/2),:] = power_patch   
            
            frame2 = np.where(powermap_3ch==0,frame,powermap_3ch)
            frame = cv2.addWeighted(frame,0.3,frame2,0.7,0)
            cv2.imshow("Frame", frame)
            records.append(temp_record)

        # debug only
        # cv2.imshow("Thresh", thresh)
        # cv2.imshow("Frame Delta", frameDelta)
 
        # handle keypresses and save output file
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            np.savetxt(os.path.join(os.getcwd(),'700.txt'),np.array(records))
            break

    # gracefully free the resources
    cv2.destroyAllWindows()
    
    '''
    # generate picture
    if init_tracking_BB is not None and powermap is not None and firstFrame is not None:
    '''
    if powermap is not None and firstFrame is not None:
        blurred = gaussian_with_nan(powermap, sigma=0)
        plt.imshow(cv2.cvtColor(firstFrame, cv2.COLOR_BGR2RGB))
        plt.imshow(blurred, cmap='inferno', interpolation='nearest',alpha=0.6)
        plt.axis('on')
        plt.title("E-Field map (min. "+"%.2f" % np.nanmin(powermap)+" V, max. "+"%.2f" % np.nanmax(powermap)+" V)")
        cbar=plt.colorbar()
        #plt.clim(min_window, max_window) #Colormap limits like in the live measurements
        cbar.set_label('Probe voltage (V)', rotation=270)
        plt.show()
    else:
    	print("Warning: nothing captured, nothing to do")
        
if __name__== "__main__":
  main()
