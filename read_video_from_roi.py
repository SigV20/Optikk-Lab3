"""
Read the input video file, display first frame and ask the user to select a
region of interest.  Will then calculate the mean of each frame within the ROI,
and return the means of each frame, for each color channel, which is written to
file.

Similar to read_video_and_extract_roi.m, but for Python.

Requirements:

* Probably ffmpeg. Install it through your package manager.

* OpenCV python package.
    For some reason, the default packages in Debian/Raspian (python-opencv and
    python3-opencv) seem to miss some important features (selectROI not present
    in python3 package, python2.7 package not compiled with FFMPEG support), so
    install them (locally for your user) using pip:
    - pip install opencv-python
    - (or pip3 install opencv-python)
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


output_filnavn_mp4 = "TR5.mp4"
output_filnavn_h264 = "TR5.h264"
#CLI options

filename = 'TR5.mp4'
output_filename = 'TR5'

#read video file
cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Could not open input file. Wrong filename, or your OpenCV package might not be built with FFMPEG support. See docstring of this Python script.")
    exit()

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

mean_signal = np.zeros((num_frames, 3))

#loop through the video
count = 0
while cap.isOpened():
    ret, frame = cap.read() #'frame' is a normal numpy array of dimensions [height, width, 3], in order BGR
    if not ret:
        break

    #display window for selection of ROI
    if count == 0:
        window_text = 'Select ROI by dragging the mouse, and press SPACE or ENTER once satisfied.'
        ROI = cv2.selectROI(window_text, frame) #ROI contains: [x, y, w, h] for selected rectangle
        cv2.destroyWindow(window_text)
        print("Looping through video.")

    #calculate mean
    cropped_frame = frame[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2], :]
    mean_signal[count, :] = np.mean(cropped_frame, axis=(0,1))
    count = count + 1

cap.release()

#save to file in order R, G, B.
np.savetxt(output_filename, np.flip(mean_signal, 1))
print("Data saved to '" + output_filename + "', fps = " + str(fps) + " frames/second")

Ti_data = np.genfromtxt("./Haringenbetydning.txt", delimiter=",")

names = [] 
marks = [] 
  







#R_values = data[:, 0] # This selects all rows of the first column



  

    
'''


def refl_simulation():
    wavelengths = np.arange(blue_wavelength, red_wavelength + 1)
    R_values = []

    for i in wavelengths:
        mua_simulated = mua_blood_oxy(i) * bvf + mua_other
        musr_simulated = 100 * (17.6 * (i / 500) ** -4 + 18.78 * (i / 500) ** -0.22)
        R_simulated = np.sqrt(3 * ((musr_simulated) / mua_simulated) + 1)
        R_values.append(R_simulated)
        
    plt.plot(wavelengths, R_values, label='Reflection')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection')
    plt.title('Reflection vs. Wavelength')
    plt.legend()
    plt.show()

    return 0

def transmisjon (R, d,C,musr,mua):
    #penetrasjonsd
    dybde = np.sqrt(1/(3*(musr+mua)*mua))
        
    #For evt rapport, plot 
        
    transmitted = np.exp(-C*d)
    #red_reference = R[0]

    print(f"Transmission red:", + transmitted[0], "// Expected probing depth:", R[0] )
    print(f"Transmission green:", + transmitted[1], "// Expected probing depth:", + R[1])
    print(f"Transmission blue:", + transmitted[2], "// Expected probing depth:", + R[2])
    return 0
    #transmisjon(R, d, C,musr,mua)
'''


