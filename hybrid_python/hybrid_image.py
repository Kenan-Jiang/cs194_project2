import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
from scipy import signal
import numpy as np
import argparse
from skimage import io

parser = argparse.ArgumentParser(description='cs194 hybrid_image.py args')
parser.add_argument('--img_high', required=True, type=str,
                    help='high frequency image path')
parser.add_argument('--img_low', required=True, type=str,
                    help='low frequency image path')
parser.add_argument('--cutoff1', default=12, type=int,
                    help='cutoff frequency for img_high')
parser.add_argument('--cutoff2', default=10, type=int,
                    help='cutoff frequency for img_low') 
parser.add_argument('--filter_size', default=5, type=int,
                    help='filter size of gaussian kernel')      
parser.add_argument('--pyramid_height', default=5, type=int,
                    help='height of lapacian and gaussian pyramid')     
args = parser.parse_args() 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def low_pass(image, filter_size, cutoff):
    gaussian = cv2.getGaussianKernel(filter_size, cutoff)
    gaussian = gaussian * gaussian.T
    filtered = signal.convolve2d(image, gaussian, mode='same')
    return filtered

def high_pass(image, filter_size, cutoff):
    gaussian = cv2.getGaussianKernel(filter_size, cutoff)
    gaussian = gaussian*gaussian.T
    filtered = signal.unit_impulse((image.shape[0], image.shape[1])) - signal.convolve2d(image, gaussian, mode='same')
    return filtered

def hybrid_image(im1, im2, cutoff1, cutoff2, filter_size):
    low_im = low_pass(im1, filter_size, cutoff1)
    high_im = high_pass(im2, filter_size, cutoff2)
    return (low_im + high_im)/2

def fourier(image):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))

def main():
    # First load images
    # high sf
    im1 = plt.imread(args.img_high)/255.
    # low sf
    im2 = plt.imread(args.img_low)/255
    # Next align images (this code is provided, but may be improved)
    if len(im1.shape) != len(im2.shape) or im1.shape[2] != im2.shape[2]:
        print("two image cannot be aligned with ease, pick another pair")
        return 
    im1_aligned, im2_aligned = align_images(im1, im2)
    # im1_aligned = rgb2gray(im1_aligned)
    # im2_aligned = rgb2gray(im2_aligned)
    #change to single color channel
    hybrid = hybrid_image(im1_aligned, im2_aligned, args.cutoff1, args.cutoff2, args.filter_size)
    plt.imshow(hybrid)
    io.imsave(args.img_high[:-4]+args.img_low[:-4]+"hybride.jpg", hybrid)
    if args.img_high == "apple.jpg":
        gray_apple = rgb2gray(im1)
        gray_bad = rgb2gray(im2)
        filtered_high = high_pass(im1_aligned, args.filter_size, args.cutoff1)
        filtered_low = high_pass(im2_aligned, args.filter_size, args.cutoff2)
        io.imsave("fourier_high.jpg", fourier(gray_apple))
        io.imsave("fourier_low.jpg", fourier(gray_bad))
        io.imsave("fourier_filtered_high.jpg", fourier(filtered_high))
        io.imsave("fourier_filtered_low.jpg", fourier(filtered_low))
        io.imsave("fourier_hybride.jpg", fourier(hybrid))
    ## Compute and display Gaussian and Laplacian Pyramids
    ## You also need to supply this function
    return 

if __name__ == '__main__':
    main()

