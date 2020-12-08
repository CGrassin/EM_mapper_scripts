# Usage

1. First, it's needed to calibrate your camera. Print an image with the chessboard and attach it to some rigid surface like a book.
2. Create a folder called "chessboard"
3. Run the script called "Camera_calibration.py" place the chessboard in front of your camera and press s to save a picture into the "chessboard" folder. The more different positions/angles will be saved the better the result. q for exit
4. Print a QR code from the set. The size of the Qr code can be set up in the script "camera_emi_mapper_aruco_V3.py" (28mm default) and attach it to the field probe.
5. Check the script "camera_emi_mapper_aruco_V3.py" to set frequency, QR code size and chessboard size (22mm default).
6. Run the script "camera_emi_mapper_aruco_V3.py" and perform the measurements. Besides the picture, it will generate an output file "Field_map.txt", which contains the coordinates and field value.
7. Run the script "4D_results_plot.py" to plot 3d field distribution.
