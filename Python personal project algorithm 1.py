import numpy as np
import time
import matplotlib.pyplot as plt

def decolor(image) :
    d_Image = np.zeros_like(image) + (np.sum(image, axis=2) / 3).reshape(np.shape(image))

    return d_Image

def diff_Amplification1(image) :
    boundary_constant = 1
    image_mean = np.mean(image)
    boundary_halfsize = boundary_constant * np.std(image)
    mask_high = image > image_mean + boundary_halfsize
    mask_low = image < image_mean - boundary_halfsize
    low_bound = 0
    high_bound = 255

    amp_image = ((image.astype(float)-image_mean+boundary_halfsize)*(high_bound-low_bound)/2/boundary_halfsize+low_bound).astype(np.uint8)
    ## 분포 영역 변환

    amp_image[mask_high] = 255
    amp_image[mask_low] = 0

    return amp_image

def diff_Amplification2(image) :
    color_width = 10
    amp_image = (image/color_width).astype(np.uint8)*color_width

    return amp_image

def diff_Amplification3(image) :
    return

input_Image = plt.imread(input("Image with filetype : "))

startTime = time.time()

decolored_Image = decolor(input_Image)

amplified_Image1 = diff_Amplification1(decolored_Image)
amplified_Image2 = diff_Amplification2(input_Image)

endTime = time.time()

print("Processing time %.3f ms"%((endTime-startTime)*1000))

fig = plt.figure()
im1 = fig.add_subplot(2,2,1)
im2 = fig.add_subplot(2,2,2)
im3 = fig.add_subplot(2,2,3)
im4 = fig.add_subplot(2,2,4)

im1.imshow(input_Image)
im2.imshow(decolored_Image)
im3.imshow(amplified_Image1)
im4.imshow(amplified_Image2)

plt.show()
