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
queue = queue.Queue(maxsize=0) 

#defines the fore/back-ground colour of pixels depending on the threshold
def threshold(img, thresh):
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img

# def binaryMorph(img, thresh):
#     img[img > thresh] = 1
#     img[img <= thresh] = 0
#     return img

#gets the pixels of image and creates histogram
def img_hist_erode(img):
    hist = np.ones(256)
    for i in range(0, img.shape[0]):
        # print('Rows=', i)
        for j in range(0,img.shape[1]):
            # print('Columns', j)
            hist[img[i, j]] +=1
            # print('HIST', hist)
        return hist

def img_hist_dilute(img):
    hist = np.zeros(256)
    for i in range(0, img.shape[0]):
        # print('Rows=', i)
        for j in range(0,img.shape[1]):
            # print('Columns', j)
            hist[img[i, j]] +=1
            # print('HIST', hist)
        return hist

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


def morph_image_dilute(img):
    kernel = np.ones((3,3), np.uint8)
    dilation=cv.dilate(img, kernel, iterations=3)

    return dilation

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



#############1st attempt Component labelling function #########################
# def connected_component_labelling(image: np.uint8) -> np.int:
#     nrow = image.shape[0]
#     ncol = image.shape[1]
#     component_map = np.full((nrow, ncol), -1, dtype=np.int)
#     curlab = 1


#     #loop through each pixel and visit its neighbours
#     #if the pixel has been visited or is 0 go to next pixel
#     #if the pixel is foreground pixel and not already labelled, label and add to queue -> 
#         #loop through neighbours of newly added pixel in the queue and label them as appropriate
#     for k in range(nrow):
#         for q in range(ncol):
#             labeled_pixel = False
#             queue.put((k, q))
#             while queue.not_empty:
#                 i, j = queue.get()
#                 if component_map[i, j] != -1: #if the pixel has already been visited continue
#                     continue

#                 if image[i, j] == 0: #if the pixel is 0 stay stay as zero and continue
#                     component_map[i,j] = 0
#                     continue

#                 labeled_pixel = True
#                 component_map[i,j] = curlab #if the pixel is not 0 and not already visited label it

#                 #add the neighbours
#                 if j > 0:
#                     queue.put((i, j-1))#add left pixel to queue 
#                 if j < ncol -1:
#                     queue.put((i, j+1))#add right pixel to queue
#                 if i > 0:
#                     queue.put((i-1, j))#add north pixel to queue 
#                 if i < nrow - 1:
#                     queue.put((i+1, j))#add south pixel to queue
                
#             if labeled_pixel:#when finished labelling neighbours go to next pixel and increment curlab by 1
#                 curlab += 1

#     return component_map, curlab -1

            
####################################################################
# def print_image(image):
#     for y, row in enumerate(image):
#         print(row)

############### 2nd attempt at CCL###############################

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


##################convolution and gaussian filtering##################
def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
 
    print("Kernel Shape : {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()
 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    print("Output Image size : {}".format(output.shape))
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
 
    return output

    
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 

def gaussian_kernel(size, sigma=1, verbose=False):
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
 
    return kernel_2D

def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)
 
####################################################################








####################################################################



# path = 'C:/Users/eire1/Documents/College Year 4/Computer Vision/Orings/Oring'
# i = 1
# while True:
    #read in images 1 by 1
img = cv.imread('C:/Users/eire1/Documents/College Year 4/Computer Vision/Orings/Oring3.JPG',0)
# i =(i + 1)%15
# if i==0:
#     i+=1

hist = img_hist_erode(img)
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

cv.imshow('Thresholded', img)
cv.imshow('Dilation', image1)
cv.imshow('Erosion', image)

imgs = binarize(image, thresh)
img = connected_component_labelling(imgs)

coloured_img = colourize(img)
coloured_img.show()

imggaus = gaussian_blur(img, 5, verbose=True)
cv.imshow("Gaussian: ", imggaus)

print("Threshold = ", thresh)



#plot the calculated histogram 
plt.plot(hist) #this displays the plot of the histogram 
plt.show()


# ch = cv.waitkey(1000)
# if ch & 0xFF == ord('q'):
#     break


