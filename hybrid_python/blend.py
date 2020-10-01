import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
from scipy import signal
import numpy as np
import argparse
from skimage import io

from main import build_gaussian_filter, gaussian_stack, laplacian_stack

parser = argparse.ArgumentParser(description='cs194 blend.py args')
parser.add_argument('--img1', required=True, type=str,
                    help='left image')
parser.add_argument('--img2', required=True, type=str,
                    help='right image')
parser.add_argument('--cutoff1', default=12, type=int,
                    help='cutoff frequency for img_high')
parser.add_argument('--cutoff2', default=10, type=int,
                    help='cutoff frequency for img_low') 
parser.add_argument('--filter_size', default=5, type=int,
                    help='filter size of gaussian kernel')      
parser.add_argument('--pyramid_height', default=5, type=int,
                    help='height of lapacian and gaussian pyramid')     
args = parser.parse_args() 

#https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop(image, cropx, cropy):
    y, x, z = image.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return image[starty:starty+cropy, startx:startx+cropx, :]

def make_mask(mode, image):
    if mode == "normal":
        mask = np.zeros(image.shape)
        for i in range(image.shape[1]//2):
            mask[:, i, :] = 1
    else:
        np.random.seed(42)
        mask = np.random.rand(image.shape[0], image.shape[1], 3)
        mask[mask>0.5] = 1
        mask[mask<0.5] = 0
    return mask

def blend(mask_stack, img1_stack, img2_stack, img1_gau, img2_gau):
    result = []
    assert len(mask_stack) == len(img1_stack)
    assert len(img1_stack) == len(img2_stack)
    for i in range(len(mask_stack)):
        ls = mask_stack[i]*img1_stack[i]+(np.ones(mask_stack[i].shape)-mask_stack[i])*img2_stack[i]
        result.append(ls)
    return sum(result)

def main():
    img1 = plt.imread(args.img1)/255.
    img2 = plt.imread(args.img2)/255
    if img1.shape != img2.shape:
        cropy = min(img1.shape[0], img2.shape[0])
        cropx = min(img1.shape[1], img2.shape[1])
        img1 = crop(img1, cropx, cropy)
        img2 = crop(img2, cropx, cropy)
    if args.img1 == "./spline/sea.jpg" or args.img2 == "./spline/sea.jpg":
        mask = make_mask("irregular", img1)
    else:
        mask = make_mask("normal", img1)
    gaussian_mask = gaussian_stack(mask, 6, filter_size=25, sigma=4)
    if args.img1 == "./spline/apple.jpeg" or args.img2 == "./spline/apple.jpeg":
        io.imsave("orangeapple_mask.jpg", gaussian_mask[-1])
    elif args.img1 == "./spline/sea.jpg" or args.img2 == "./spline/sea.jpg":
        io.imsave("skysea_mask.jpg", gaussian_mask[-1])
    else:
        io.imsave("cityblend_mask.jpg", gaussian_mask[-1])

    gaussian_img1 = gaussian_stack(img1, 6)
    laplacian_im1 = laplacian_stack(img1, gaussian_img1)

    gaussian_img2 = gaussian_stack(img2, 6)
    laplacian_im2 = laplacian_stack(img2, gaussian_img2)
    result = blend(gaussian_mask, laplacian_im1, laplacian_im2, gaussian_img1, gaussian_img2)
    if args.img1 == "./spline/apple.jpeg" or args.img2 == "./spline/apple.jpeg":
        io.imsave("orangeapple.jpg", result)
    elif args.img1 == "./spline/sea.jpg" or args.img2 == "./spline/sea.jpg":
        io.imsave("skysea.jpg", result)
    else:
        io.imsave("cityblend.jpg", result)
    return 
if __name__ == '__main__':
    main()