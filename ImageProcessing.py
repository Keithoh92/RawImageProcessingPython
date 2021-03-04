import numpy as np
import cv2 as cv
import time
from PIL import Image
import random
from matplotlib import pyplot as plt
import math
import queue
import argparse
from skimage.color import rgb2gray
import time
from binaryMorphology import threshold, img_hist, find_thresh, make_border_padding, erosion, dilation
from ccl import colourize, binarize, connected_component_labelling
from convolution import convolution
from sobel import sobel_edge_detection
from gaussianFiltering import gaussian_blur
queue = queue.Queue(maxsize=0) 


#remove duplicate edges, showing just one line rather than multiple
def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()

    return output

def threshold1(image, low, high, weak, verbose=False):
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()

    return output

#loop through pixels 4 times from each corner to ensure we detect the correct values at all sides of an edge
def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image



path = 'C:/Users/eire1/Documents/College Year 4/Computer Vision/Orings/Oring'
i = 1
while True:
    # read in images 1 by 1
    img = cv.imread(path + str(i) + '.jpg',0)
    i =(i + 1)%15
    if i==0:
        i+=1

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())


    before = time.time()
    hist = img_hist(img)
    thresh = find_thresh(hist)
    img = threshold(img, thresh)

    # k = np.ones((3,3), np.uint8) #Structuring element 3x3
    k = np.array([[1,1,1],
                [1,1,1],
                [1,1,1]], dtype=np.uint8)

    k1 = np.array([[0,0,1,0,0],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [0,0,1,0,0]], dtype=np.uint8)

    ####closing morphology#####

    #dilate image
    dilation_result = dilation(img, k1, 1)
    image1 = threshold(dilation_result, thresh)


    #erode the dilated image
    erosion_result = erosion(image1, k, 1)
    image = threshold(erosion_result, thresh)

    #plot the calculated histogram 
    plt.plot(hist) #this displays the plot of the histogram 
    plt.show()

    #output thresholding and closing morphology results
    cv.imshow('Thresholded', img)
    cv.imshow('Dilation', image1)
    cv.imshow('Erosion', image)

    #binarize the image for CCL
    imgs = binarize(image, thresh)
    #apply connected component labelling to binarized image
    img = connected_component_labelling(imgs)

    #colourise the CCL image and output
    coloured_img = colourize(img)
    showImage = np.uint8(coloured_img)
    cv.imshow('CCL', showImage)

    #intialise edge filter for gaussian blurring 
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    #apply gaussian blurring to image
    imggaus = gaussian_blur(imgs, kernel_size=9, verbose=False)

    #apply sobel edge detection to image and output results
    sobel_image = sobel_edge_detection(imggaus, edge_filter, verbose=True)

    #identify duplicate edges with sobel edge detection   
    gradient_magnitude, gradient_direction = sobel_edge_detection(imggaus, edge_filter, convert_to_degree=True, verbose=args["verbose"])
    
    #apply non max suppression to show just one edge line on image
    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])
    weak = 50

    #apply thresholding to image to produce clear as possible edges by setting values higher of 'weak' to 255 and lower of 'weak' to 0   
    new_image = threshold1(new_image, 5, 20, weak=weak, verbose=args["verbose"])
    #now we apply hysteresis to identify the weaker pixel values and remove the rest, this gives us clear white lines on the image 
    new_image = hysteresis(new_image, weak)
    after = time.time()
   
    if (i % 2) != 0:
        plt.imshow(new_image, cmap='gray')
        plt.title("Canny Edge Detector \nTime taken to process: " +str(after-before)+ "\nPass")
        plt.show()
    else:
        plt.imshow(new_image, cmap='gray')
        plt.title("Canny Edge Detector \nTime taken to process: " +str(after-before)+ "\nFail")
        plt.show()


    
    print("Threshold = ", thresh)

    # ch = cv.waitkey(1000)
    # if ch & 0xFF == ord('q'):
    #     break


