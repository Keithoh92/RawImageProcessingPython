import numpy as np
import cv2 as cv
import math


#gets the pixels of image and creates histogram
def img_hist(img):
    hist = np.ones(256)
    for i in range(0, img.shape[0]):
        # print('Rows=', i)
        for j in range(0,img.shape[1]):
            # print('Columns', j)
            hist[img[i, j]] +=1
            # print('HIST', hist)
        return hist


#defines the fore/back-ground colour of pixels depending on the threshold
def threshold(img, thresh):
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img

def find_thresh(hist):
    
    s_max = (0,0)
    
    for threshold in range(256):

        # update
        w_0 = sum(hist[:threshold])
        w_1 = sum(hist[threshold:])

        mu_0 = sum([i * hist[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0       
        mu_1 = sum([i * hist[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate - inter class variance
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2

        if s > s_max[1]:
            s_max = (threshold, s)
                        
    return s_max[0]

def make_border_padding(image, padding, value):
    return cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=value)


####################my code for erosion##############################
def erosion(img, kernel, padding=0):
        new_image = img.copy()

        my_padding = 1 #add padding around image border
        padded = make_border_padding(img, padding, my_padding)

        y = padded.shape[0] - kernel.shape[0] #templates final position over image on the y axis
        x = padded.shape[1] - kernel.shape[1] #templates final position over image on the x axis 

        #starting position
        y_pos = 0

        while y_pos <= y: # move down image
            x_pos = 0

            while x_pos <= x: # move across image
                erode_flag = False # erosion is set to False until we come across a 0 in the image pixels underneath

                #index of SE
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        if kernel[i][j] == 1:
                            #if we find a 255 on image pixel underneath SE set erosion to True and break out of second for loop
                            if padded[y_pos+i][x_pos+j] == 255:
                                erode_flag = True
                                break
                    #if no match is found break out of 1st for loop
                    if erode_flag:#if corresponding pixels on image are 255 set new image pixel to 255
                        new_image[y_pos, x_pos] = 255
                        break

                x_pos += 1
            y_pos += 1
        return new_image

def dilation(img, kernel, padding=0):
    new_image = img.copy()

    my_padding = 0
    padded = make_border_padding(img, padding, my_padding)

    y = padded.shape[0] - kernel.shape[0] #templates final position over image on the y axis
    x = padded.shape[1] - kernel.shape[1] #templates final position over image on the x axis 

    #starting position
    y_pos = 0

    while y_pos <= y: # move down image
        x_pos = 0

        while x_pos <= x: # move across image
            dilate_flag = False # dilation is set to False until we come across a 1 in the image pixels underneath

                #index of SE
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    if kernel[i][j] == 1:
                        #if we find a 1 on image pixel underneath SE set dilation to True and break out of second for loop
                        if padded[y_pos+i][x_pos+j] == 0:
                            dilate_flag = True
                            break
                #if no match is found break out of 1st for loop
                if dilate_flag:
                    new_image[y_pos, x_pos] = 0
                    break

            x_pos += 1
        y_pos += 1
    
    return new_image