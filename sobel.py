import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from convolution import convolution
from gaussianFiltering import gaussian_blur


def sobel_edge_detection(image, filter, convert_to_degree=False, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge Detection")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
    
    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge Detection")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # if verbose:
    #     plt.imshow(gradient_magnitude, cmap='gray')
    #     plt.title("Gradient Magnitude, Time taken to process: ")
    #     plt.show()

    gradient_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180

    return gradient_magnitude, gradient_direction