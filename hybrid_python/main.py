import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
from scipy import signal
import numpy as np
from skimage import io

##question 2.3
def build_gaussian_filter(filter_size, sigma):
    gaussian = cv2.getGaussianKernel(filter_size, sigma)
    filter = np.outer(gaussian, gaussian.T)
    return filter

def gaussian_stack(image, stack_height, filter_size=5, sigma=10):
    filter = build_gaussian_filter(filter_size, sigma)
    result = [image]
    if len(image.shape) == 3:
        for _ in range(0, stack_height):
            r = signal.convolve2d(result[-1][:,:,0],filter,mode='same')  
            g = signal.convolve2d(result[-1][:,:,1],filter,mode='same')  
            b = signal.convolve2d(result[-1][:,:,2],filter,mode='same')  
            img = np.dstack([r, g, b])
            result.append(img)
    else:
        for _ in range(0, stack_height):
            img = signal.convolve2d(result[-1],filter,mode='same')  
            result.append(img)
    return result[1:]

def laplacian_stack(image, gaussian_stack):
    lapalcidan_stack = []
    current = image
    stack_height = len(gaussian_stack)
    for i in range(0, stack_height-1):
        img = current - gaussian_stack[i]
        current = gaussian_stack[i]
        lapalcidan_stack.append(img)
    lapalcidan_stack.append(gaussian_stack[-1])
    return lapalcidan_stack

def main():
    lincoln = plt.imread('./lincoln.jpg')/255
    gaussian_lincoln = gaussian_stack(lincoln, 5)
    gaussian_lincoln_image = np.concatenate(tuple(gaussian_lincoln), axis = 1)
    io.imsave("lincoln_gaussion.jpg", gaussian_lincoln_image)
    lapalcidan_lincoln = laplacian_stack(lincoln, gaussian_lincoln)
    lapalcidan_lincoln_image = np.concatenate(tuple(lapalcidan_lincoln), axis = 1)
    io.imsave("lincoln_lapalcidan.jpg", lapalcidan_lincoln_image)
    apple_hybride = plt.imread('./applebad_applehybride.jpg')/255
    gaussian_apple = gaussian_stack(apple_hybride, 5)
    gaussian_apple_image = np.concatenate(tuple(gaussian_apple), axis = 1)
    io.imsave("apple_gaussion.jpg", gaussian_apple_image)
    lapalcidan_apple = laplacian_stack(apple_hybride, gaussian_apple)
    lapalcidan_apple_image = np.concatenate(tuple(lapalcidan_apple), axis = 1)
    io.imsave("apple_lapalcidan.jpg", lapalcidan_apple_image)
    return 
    
if __name__ == '__main__':
    main()