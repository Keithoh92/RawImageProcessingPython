import numpy as np
import cv2 as cv
import time
from PIL import Image
import random
import queue
import argparse
from skimage.color import rgb2gray



##function to visualise CCL
def colourize(img):
    height, width = img.shape
    colors = []
    colors.append([])
    colors.append([])
    color = 1
    #Displating distinctive components with distinct colors
    coloured_img = Image.new("RGB", (width, height))
    coloured_data = coloured_img.load()

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 0:
                if img[i][j] not in colors[0]:
                    colors[0].append(img[i][j])
                    colors[1].append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

                ind = colors[0].index(img[i][j])
                coloured_data[j, i] = colors[1][ind]

    return coloured_img


##function to binarize image
def binarize(img_array, threshold):
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i][j] > threshold:
                img_array[i][j] = 0
            else:
                img_array[i][j] = 1
    return img_array

def connected_component_labelling(img):
    ##first pass###
    curlab = 1
    img = np.array(img)
    labels = np.array(img)

    label_conv = []
    label_conv.append([])
    label_conv.append([])

    count = 0
    #loop through pixels in image
    for i in range(1, len(img)):
        for j in range(1, len(img[0])):

            if img[i][j] > 0:##if pixel is foreground pixel

                ##always setting neighbouring img pixels to left and above to label_x and label_y 
                label_x = labels[i][j-1] 
                label_y = labels[i-1][j]

                #if neighbouring pixels are foreground pixels
                if label_x > 0:
                    if label_y > 0: 
                        ##and pixels are not equal
                        if not label_x == label_y:
                            ##keeping track of the foreground pixels label sets and subsets so we can connect them together

                            labels[i][j] = min(label_x, label_y)#set new array pixel to be the minimum of the labels x & y
                            if max(label_x, label_y) not in label_conv[0]: 
                                #if the max pixel of labels x(above) & y(left) is not already in the label_conversion array[0] add it 
                                label_conv[0].append(max(label_x, label_y))
                                #then add the smaller pixel of above and left pixel to the label_conversion array[1]
                                label_conv[1].append(min(label_x,  label_y))
                            
                            #else if the max of labels x & y has already been visited and stored in label_conversion array[0] get its index
                            elif max(label_x, label_y) in label_conv[0]:
                                ind = label_conv[0].index(max(label_x, label_y))
                                
                                #if label_conv[1][index] at label_conv[0][index] is greater than the smaller of the 2 pixels get that index of label_conv[1]
                                if label_conv[1][ind] > min(label_x, label_y):
                                    l = label_conv[1][ind]
                                    #change the pixel in label_conv[1][index] to be the smaller pixel
                                    label_conv[1][ind] = min(label_x, label_y)

                                    #now we connect the components   
                                    while l in label_conv[0] and count < 100:
                                        count += 1
                                        ind = label_conv[0].index(1)
                                        l = label_conv[1][ind]
                                        label_conv[1][ind] = min(label_x, label_y)
                                    
                                    label_conv[0].append(1)
                                    label_conv[1].append(min(label_x, label_y))
                        else:
                            labels[i][j] = label_y
                            #only x has a label
                    else:
                        labels[i][j] = label_x
                #only y has a label
                elif label_y > 0:
                    labels[i][j] = label_x

                else:
                    labels[i][j] = curlab
                    curlab += 1

    ###starting second pass#####
    count = 1
    for index, val in enumerate(label_conv[0]):

        if label_conv[1][index] in label_conv[0] and count < 100:
            count += 1
            ind = label_conv[0].index(label_conv[1][index])
            label_conv[1][index] = label_conv[1][ind]

    for i in range(1, len(labels)):
        for j in range(1, len(labels[0])):

            if labels[i][j] in label_conv[0]:
                ind = label_conv[0].index(labels[i][j])
                labels[i][j] = label_conv[1][ind]
    
    return labels