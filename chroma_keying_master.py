# Chroma Keying using OpenCV
# Author : Suprojit Nandy
# Affiliation : Hardware and Embedded Systems Lab, NTU Singapore.

import cv2
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

from DataPath import DATA_PATH


# Definition of the Chroma Keying Function Here :
def chroma_key(img, background, gaussian_coeff):
    # make a copy of the image Here :
    img_cp = np.copy(img)
    # We use HSV color space for this from theory:
    img_hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)
    # Define range of green color in HSV here :
    l_green = np.array([36, 0, 0])
    u_green = np.array([86, 255, 255])

    # Define range for
    # Threshold the green HUE components in this range

    greenscreen_mask = cv2.inRange(img_hsv, l_green, u_green)

    # mask_out = cv2.bitwise_not(greenscreen_mask)
    kernel_size = (3, 3)
    kernel_ellp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # mask_out_f = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel_ellp, iterations=7)
    mask_out_f = cv2.morphologyEx(greenscreen_mask, cv2.MORPH_OPEN, kernel_ellp,
                                  iterations=7)  # Use Morphological Closing to denoise the image mask :

    # Mask out the image :
    # Apply Gaussian to the subject here :
    img = cv2.GaussianBlur(img,(gaussian_coeff, gaussian_coeff),sigmaX=0)

    mask_im_v = np.zeros(img.shape, np.uint8)
    gray_cvt_v = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_g_v, thresh_g_v = cv2.threshold(gray_cvt_v, 50, 255, cv2.THRESH_BINARY)
    # Extract the external contours to smoothen the borders
    contours_v, hierarchy_v = cv2.findContours(thresh_g_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Opencv Documentations
    cv2.drawContours(mask_im_v, contours_v, -1, (0,0,255), 5)
    blurred_border_out_v = np.where(mask_im_v == np.array([0,0,255]), img, img_cp)

    blurred_border_out_v[mask_out_f != 0] = [0, 0, 0]
    #img[mask_out_f != 0] = [0, 0, 0]

    # crop out the background image here :

    background[mask_out_f == 0] = [0, 0, 0]

    # Add the background with the image of the car ::::
    # Chroma Key function image here :
    #chroma_out = img + background
    chroma_out = blurred_border_out_v + background
    #chroma_out_blur = cv2.GaussianBlur(chroma_out, (gaussian_coeff, gaussian_coeff), sigmaX=0)
    # Dislpay Mask here :
    # cv2.imshow('Mask of the frame', mask_out_f)
    # cv2.imshow('Background with the mask incorporated', chroma_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return chroma_out

    # sys.exit()


list_vertices = []
vertex_point1 = []
vertex_point2 = []


# mouse_event_counter = 0


def background_selector(action, x, y, flags, userdata):
    # imgk_cp = np.copy(imgk)
    global vertex_point1, vertex_point2, list_vertices

    if action == cv2.EVENT_LBUTTONDOWN:
        vertex_point1 = (x, y)
        list_vertices.append(vertex_point1)
        # mouse_event_counter += 1
    elif action == cv2.EVENT_LBUTTONUP:
        vertex_point2 = (x, y)
        list_vertices.append(vertex_point2)
        # mouse_event_counter += 1
    # cv2.rectangle(image, vertex_point1, vertex_point2, (200,55,200), 2)
    # cv2.imshow(image)
    # cv2.waitKey(0)
    # cv2.destroyWindow()
    # sys.exit()
    # # Draw the ROI in this case :
    # cv2.rectangle()


# Smoothening the mask here :
maxScaleUp = 25 # Toggle this if necessary
scaleFactor = 1

windowName = "Smooth Image"
trackbarValue = "Smooth Constant"
#trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"
#im = cv2.imread("truth.png")
capture = cv2.VideoCapture(DATA_PATH + "greenscreen-asteroid.mp4")
ret_i,frame_i= capture.read()
if ret_i == True:
    initial_frame = frame_i

#sys.exit()
initial_frame = cv2.resize(initial_frame, None, fx= 0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)# Resize to fit
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
#im_t = cv2.imread('ocean3.jpg')
# Callback functions
def smooth_scaler(*args):
    global scaleFactor
    #global scaleType
    scaleFactor = args[0]
    if scaleFactor == 0:
        scaleFactor = 1
    blur_img = cv2.GaussianBlur(initial_frame, (scaleFactor, scaleFactor), sigmaX=0)
    print(scaleFactor)
    # Create a mask of the image here :
    mask_im = np.zeros(initial_frame.shape, np.uint8)
    gray_cvt = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    ret_g, thresh_g = cv2.threshold(gray_cvt, 50, 255, cv2.THRESH_BINARY)
    # Extract the external contours to smoothen the borders
    contours, hierarchy = cv2.findContours(thresh_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Opencv Documentations
    cv2.drawContours(mask_im, contours, -1, (0,0,255), 5)
    blurred_border_out = np.where(mask_im == np.array([0,0,255]), blur_img, initial_frame)
    cv2.imshow(windowName, blurred_border_out)


cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, smooth_scaler)
#cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleImage)


#flag = False
#cap_initial = cv2.VideoCapture('greenscreen-asteroid.mp4')
#ret_i, frame_i = cap_initial.read()
#frame_window = "Frame Specifics"
#cv2.namedWindow(frame_window, cv2.WINDOW_AUTOSIZE)
#trackbar_value = "Gaussian Blur Spread"
#cv2.createTrackbar(trackbar_value, frame_window, 1, max_kernel_dimension, smooth_scaler)

smooth_scaler(7)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break
cv2.destroyAllWindows()

print(scaleFactor)
#sys.exit()

cap = cv2.VideoCapture(DATA_PATH + 'greenscreen-asteroid.mp4')
img_background = cv2.imread(DATA_PATH + 'ocean3.jpg')
img_background_t = cv2.transpose(img_background)
img_cp = img_background_t.copy()
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Reset the Background for each frame for new mask generation
    img_cp = img_background_t.copy()
    if ret == True:

        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        cv2.imshow('Frame', chroma_key(frame, img_background_t, scaleFactor))
        img_background_t = img_cp
        # Press esc on keyboard to  exit
        if cv2.waitKey(100) & 0xFF == 27:
            break

    # Break the loop
    else:
        break

sys.exit()

















# Read the Image here :::
#
# # video_in = cv2.VideoCapture()
# video_in = cv2.VideoCapture('greenscreen-asteroid.mp4')
# img_background = cv2.imread('ocean3.jpg')
# img_background_t = cv2.transpose(img_background)
#
# if video_in.isOpened() == False:
#     print('Problem streaming input video')
# else:
#     print('Video File Opening Properly')
#
# while video_in.isOpened():
#     # Read frame by frame :
#     ret, read_frame = video_in.read()
#
#     # If there is an image to be read here :
#     if ret == True:
#
#         frame_chroma = chroma_key(read_frame, img_background_t)
#
#         cv2.imshow('chroma_frame', frame_chroma)
#         #cv2.imshow('Frame', chroma_key(read_frame, img_background_t))
#         #cv2.waitKey(100000)
#         # Introduce some delay :
#         #k = cv2.waitKey(0)
#         # Press esc on keyboard to  exit
#         if cv2.waitKey(100) & 0xFF == 27:
#             break
#         # if k == ord('q'):
#         #     break
#         else:
#             break
#
# # sys.exit()
# #
# # image = cv2.imread('greenscreen-car.jpg')
# # img_cloner = image.copy()
# #
# # chroma_key(image, img_background_t)
# #
# # # Convert background file to RGB form :
# # """
# # img_b = cv2.cvtColor(img_background, cv2.COLOR_BGR2RGB)
# # img_b_t = cv2.transpose(img_b)
# # cv2.imshow('Background Image', img_b_t[:,:,::-1])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # # Shape of the image here :
# # print(image.shape)
# # print(img_b_t.shape)
# #
# # """
# #
# # sys.exit()
# # # cv2.imshow('Frame', image)
# # # chroma_key(image)
# # # sys.exit()
# #
# # cv2.namedWindow('Window')
# # cv2.setMouseCallback('Window', background_selector)
# #
# # k = 0
# # while True:
# #     cv2.imshow('Window', image)
# #     k = cv2.waitKey(1) & 0xFF
# #
# #     if k == ord('r'):
# #         image = img_cloner.copy()
# #     elif k == ord('q'):
# #         break
# # # print(list_vertices[0])
# # # print(list_vertices[1])
# # # print(list_vertices[0][0])
# # # print(list_vertices[0][1])
# # # print(list_vertices[1][0])
# # # print(list_vertices[1][1])
# # print(len(list_vertices))
# #
# # # Check number of mouse events :
# # if len(list_vertices) == 2:
# #     # cropped = img_cloner[list_vertices[0][0]:list_vertices[0][1], list_vertices[1][0]:list_vertices[1][1]]
# #     cropped = img_cloner[list_vertices[0][1]:list_vertices[1][1], list_vertices[0][0]:list_vertices[1][0]]
# #     cv2.imshow('Cropped', cropped)
# #     cv2.waitKey(0)
# #
# # cv2.destroyAllWindows()
# #
# # sys.exit()
# #
# #
# # #
# #
# #
# # def main():
# #     # Import the Demo car image here :
# #     # global cropper
# #     image = cv2.imread('greenscreen-car.jpg', 1)
# #     image_clone = image.copy()
# #     cv2.namedWindow("Frame_Considered")
# #     cv2.setMouseCallback("Frame_Considered", background_selector)
# #
# #     k_coeff = 0
# #     # Loop till escape is pressed :
# #     while k_coeff != 27:
# #         cv2.imshow("Window", image)
# #         k_coeff = cv2.waitKey(20) & 0xFF
# #
# #         # Incase we need to reset the cropping zone press r:
# #         if k_coeff == ord('r'):
# #             image = image_clone.copy()
# #     # we have global reference
# #     # check for 2 different points here :
# #     sys.exit()
# #
# #     if len(list_vertices) == 2:
# #         cropper = image_clone[list_vertices[0][1]:list_vertices[1][1], list_vertices[0][0]:list_vertices[1][0]]
# #     # Make a copy of the image here :
# #     # img_cp = np.copy(image)
# #     # print(list_vertices[0][1], list_vertices[0][1])
# #     cv2.imshow("The cropped Image Segment", cropper)
# #     # sys.exit()
# #
# #     # print('The Image Type is :', type(image))
# #     # print('The Image Dimension is :', image.shape)
# #     # # sys.exit()
# #     # # Dislpay the image here :
# #     #
# #     # # Read the Video file here :::
# #     #
# #     # chroma_key(image)
# #     #
# #     # cv2.imshow('Frame', image)
# #     # # cv2.imshow()
# #     # cv2.waitKey(0)
# #     # cv2.destroyWindow()
# #
# #
# # main()
