import numpy as np
import time
import matplotlib.pyplot as plt

def decolor(image) :
    ##d_Image = np.zeros_like(image) + np.sum(image, axis=2) / 3   ## broadcasting rule
    d_Image = np.zeros_like(image)
    image_sum = np.sum(image, axis=2) // 3
    for i in range(3) :
        d_Image[:,:,i] = image_sum

    return d_Image

def diff_Amplification1(image) :
    boundary_constant = 1
    image_mean = np.mean(image)
    boundary_halfsize = boundary_constant * np.std(image)
    high_noise = image_mean + boundary_halfsize
    low_noise = image_mean - boundary_halfsize
    mask_high = image > high_noise
    mask_low = image < low_noise
    high_bound = 255
    low_bound = 0

    amp_image = ((image-low_noise)*(high_bound-low_bound)/2/boundary_halfsize+low_bound).astype(np.uint8)
    ## 분포 영역 변환

    amp_image[mask_high] = 255
    amp_image[mask_low] = 0

    return amp_image

def diff_Amplification2(image) :
    color_width = 10
    amp_image = image/color_width*color_width
    return amp_image.astype(eval("np."+image.dtype.name))

def diff_Amplification3(image) :
    amp_image = image.copy()
    boundary_constant = 1
    image_mean = np.mean(image)
    boundary_halfsize = boundary_constant * np.std(image)
    high_noise = image_mean + boundary_halfsize
    low_noise = image_mean - boundary_halfsize
    mask_high = image > high_noise
    mask_low = image < low_noise
    high_bound = 255
    low_bound = 0

    amp_image[mask_high] = 255
    amp_image[mask_low] = 0

    pix_per_RGB = np.size(image[(image >= low_noise)&(image <= high_noise)])//(high_bound - low_bound)
    count = 0
    for i in range(int(low_noise), int(high_noise+1)) :
        amp_image[image == i] = count//pix_per_RGB
        count += np.size(image[image == i])

    return amp_image


def hLinTrans(n, image):  ## 가로 방향 선형 변환
    row = np.shape(image)[0]
    column = np.shape(image)[1]
    hor = column
    mImage = np.zeros_like(image)

    hLinTrans_diag = np.eye(hor) * (1 + n) / 2
    mask = np.arange(hor).reshape((1, hor)) - np.arange(hor).reshape((hor, 1))
    hLinTrans_next = (np.abs(mask) == 1).astype(np.uint8) * (1 - n) / 4
    hLinTrans_next[1, 0] = (1 - n) / 2
    hLinTrans_next[hor - 2, hor - 1] = (1 - n) / 2
    hLinTrans_matrix = hLinTrans_diag + hLinTrans_next  ## 변환 행렬 생성

    for i in range(3):
        mImage[:, :, i] = np.dot(np.reshape(image[:, :, i], (row, column)), hLinTrans_matrix)  ## 선형 변환

    return mImage


def vLinTrans(n, image):  ## 세로 방향 선형 변환 행렬 생성
    row = np.shape(image)[0]
    column = np.shape(image)[1]
    ver = row
    mImage = np.zeros_like(image)

    vLinTrans_diag = np.eye(ver) * (1 + n) / 2
    mask = np.arange(ver).reshape((1, ver)) - np.arange(ver).reshape((ver, 1))
    vLinTrans_next = (np.abs(mask) == 1).astype(np.uint8) * (1 - n) / 4
    vLinTrans_next[ver - 1, ver - 2] = (1 - n) / 2
    vLinTrans_next[0, 1] = (1 - n) / 2
    vLinTrans_matrix = vLinTrans_diag + vLinTrans_next  ## 변환 행렬 생성

    for i in range(3):
        mImage[:, :, i] = np.dot(vLinTrans_matrix, np.reshape(image[:, :, i], (row, column)))  ## 선형 변환

    return mImage

def diff_Amplification4(image) :
    amp_constant = 1
    amp_image = hLinTrans(amp_constant, image) / 2 + vLinTrans(amp_constant, image) / 2

    return amp_image.astype(eval("np."+image.dtype.name))

input_Image = plt.imread(input("Image with filetype : "))
output_mode = int(input("Select output mode 1or2 : "))

time1 = time.time()
decolored_Image = decolor(input_Image)
time2 = time.time()
amplified_Image1 = diff_Amplification1(input_Image)
time3 = time.time()
amplified_Image2 = diff_Amplification2(input_Image)
time4 = time.time()
amplified_Image3 = diff_Amplification3(input_Image)
time5 = time.time()
amplified_Image4 = diff_Amplification4(input_Image)
time6 = time.time()

if output_mode==1:
    fig = plt.figure()
    for i in range(1, 7):
        code = compile("im" + str(i) + " = fig.add_subplot(2,3," + str((i - 1) // 2 + (i - 1) % 2 * 3 + 1) + ")",
                       '<string>', 'single')
        exec(code)

    im1.imshow(input_Image)
    im2.imshow(decolored_Image)
    im3.imshow(amplified_Image1)
    im4.imshow(amplified_Image2)
    im5.imshow(amplified_Image3)
    im6.imshow(amplified_Image4)

    im1.set_title("Input Image")
    im2.set_title("Decolored Image %dms" % ((time2 - time1) * 1000))
    im3.set_title("Process1 %dms" % ((time3 - time2) * 1000))
    im4.set_title("Process2 %dms" % ((time4 - time3) * 1000))
    im5.set_title("Process3 %dms" % ((time5 - time4) * 1000))
    im6.set_title("Trial1 %dms" % ((time6 - time5) * 1000))

    fig.tight_layout()
    plt.show()

else:
    plt.figure(1)
    plt.imshow(input_Image)
    plt.title("Input Image")
    plt.figure(2)
    plt.imshow(decolored_Image)
    plt.title("Decolored Image %dms" % ((time2 - time1) * 1000))
    plt.figure(3)
    plt.imshow(amplified_Image1)
    plt.title("Process1 %dms" % ((time3 - time1) * 1000))
    plt.figure(4)
    plt.imshow(amplified_Image2)
    plt.title("Process2 %dms" % ((time4 - time3) * 1000))
    plt.figure(5)
    plt.imshow(amplified_Image3)
    plt.title("Process3 %dms" % ((time5 - time4) * 1000))
    plt.figure(6)
    plt.imshow(amplified_Image4)
    plt.title("Trial1 %dms" % ((time6 - time5) * 1000))

    plt.show()

