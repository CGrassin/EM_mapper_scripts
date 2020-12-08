import time
from PIL import Image
import cv2 #sudo apt install opencv-data opencv-doc python-opencv && pip3 install opencv-contrib-python
from cv2 import aruco
from rtlsdr import RtlSdr # pip3 install pyrtlsdr
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import argparse
from imutils.video import WebcamVideoStream
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
import glob
import os

#Parula colormap data
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

# Convert a colormap to BRG for openCV
for s in cm_data:
	s[0], s[-1] = s[-1], s[0]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

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

def print_sdr_config(sdr):
    """Prints the RTL-SDR configuration in the console.
    """
    print("RTL-SDR config:")
    print("    * Using device",sdr.get_device_serial_addresses())
    print("    * Device opened:", sdr.device_opened)
    print("    * Center frequency:",sdr.get_center_freq(),"Hz")
    print("    * Sample frequency:",sdr.get_sample_rate(),"Hz")
    print("    * Gain:",sdr.get_gain(),"dB")
    print("    * Available gains:",sdr.get_gains())
	
def get_RMS_power(sdr):
    """Measures the RMS power with a RTL-SDR.
    """
    samples = sdr.read_samples(1024*4)
    freq,psd = scipy.signal.welch(samples,sdr.sample_rate/1e6,nperseg=512,return_onesided=0)
    return 10*np.log10(np.sqrt(np.mean(psd**2)))

def main():
    print("Usage:")
    print("    * Press q to display the EMI map and exit.")
    print("Call with -h for help on the args.")
    		
    # parse args
    parser = argparse.ArgumentParser(description='EMI mapping with camera and RTL-SDR.')
    parser.add_argument('-c', '--camera', type=int, help='camera id (default=0)',default=0)
    parser.add_argument('-f', '--frequency', type=float, help='sets the center frequency on the SDR, in MHz (default: 114.9).',default=114.9)
    parser.add_argument('-g', '--gain', type=int, help='sets the SDR gain (default: 496).',default=496)
    args = parser.parse_args()
    
    # configure SDR device
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.center_freq = args.frequency * 1e6
    sdr.gain = args.gain
    sdr.set_agc_mode(0)
    #print_sdr_config(sdr)

    # read from specified webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FPS,60)
    vs = WebcamVideoStream(src=args.camera).start()

    if cap is None or not cap.isOpened():
            print('Error: unable to open video source: ', args.camera)
    else:
        # wait some time for the camera to be ready
        time.sleep(2.0)


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
    BB_geometry = np.array([10,10]) #Size of the "pixel" we are using to draw
    ret,mtx,dist,rvect,tvect = cameraCalibration()

    frame = vs.read()

    # if firstFrame is None:
    firstFrame = frame
    powermap = np.empty((len(frame),len(frame[0])))
    powermap.fill(np.nan)

    powermap_3ch = np.empty((len(frame),len(frame[0]),3),dtype=np.uint8)
    powermap_3ch.fill(np.nan)
    
    #Live colormap level settings
    min_window = -32
    max_window = -25

    #An array with the data
    records=[]

    while True:

        frame = vs.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame,aruco_dict,parameters=parameters)
        #If QR code marker was detected in frame
        if corners !=[]:
            rvec,tvec,_ = aruco.estimatePoseSingleMarkers(corners,0.028,mtx,dist) #28mm size of the qr code marker
            corners = np.squeeze(corners)
            x_center = np.mean(corners[:,1])
            y_center = np.mean(corners[:,0])
            
            # fill map
            power = get_RMS_power(sdr)
            powermap[int(x_center-BB_geometry[0]/2):int(x_center+BB_geometry[0]/2),int(y_center-BB_geometry[1]/2):int(y_center+BB_geometry[1]/2)] = power          

            # sturation limits for the live colormap
            if power<min_window:
                power = min_window
            if power>max_window:
                power = max_window
            
            #Write data to the file
            temp_record = np.append(np.squeeze(tvec)[0:3],power)

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
            np.savetxt(os.path.join(os.getcwd(),'Field_map.txt'),np.array(records))
            break

    # gracefully free the resources
    sdr.close()
    cap.release()
    cv2.destroyAllWindows()

    '''
    # generate picture
    if init_tracking_BB is not None and powermap is not None and firstFrame is not None:
    '''
    if powermap is not None and firstFrame is not None:
        # Convert the colormap back to RGB
        for s in cm_data:
            s[0], s[-1] = s[-1], s[0]

        parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

        blurred = gaussian_with_nan(powermap, sigma=0) # Not used
        plt.imshow(cv2.cvtColor(firstFrame, cv2.COLOR_BGR2RGB))
        plt.imshow(blurred, cmap=parula_map, interpolation='nearest',alpha=0.6)
        plt.axis('on')
        plt.title("Noise level map (min. "+"%.2f" % np.nanmin(powermap)+" dBm, max. "+"%.2f" % np.nanmax(powermap)+" dBm)")
        cbar=plt.colorbar()
        # plt.clim(min_window, max_window) #Colormap limits like in the live measurements
        cbar.set_label('Noise level (dBm)', rotation=270)
        plt.show()

    else:
    	print("Warning: nothing captured, nothing to do")
        
if __name__== "__main__":
  main()
