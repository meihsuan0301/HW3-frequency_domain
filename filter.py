import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image
import time
from FFT import *

# Low-pass filter
# https://github.com/ShashwatNigam99/Digital_Image_Processing_Assignments/blob/master/assignment3/src/q2.m

# 0：Gaussian, 1：uniform
def add_noise(distribution,img):
    noisy = []
    mean,var = img.mean(), img.var()
    if distribution==0:
        noise = np.random.normal(0, 50, img.shape)
    else:
        noise = np.random.uniform(0, 100, img.shape)
    n = img + noise
    # avoid going over bounds
    n[n > 255] = 255
    n[n < 0] = 0
    n = np.array(n).astype(np.uint8)

    plt.imshow(n, cmap='gray')
    plt.title('Add noise', fontsize=10)
    plt.show()
    #n = n.reshape(512,512)
    return n

def idea_Low_pass_filter(img, D=50):
    # 做FFT
    FFT2_img = FFT2(img)
    #fimg = np.log(np.abs(FFT2_img))
    fft_shift = fftshift(FFT2_img)
    
    #Low_pass_filter
    but_fil = np.zeros((img.shape))
    M, N =img.shape
    for i in range(M):
        for j in range(N):
            duv = ((i-(M/2))**2 + (j-(N/2))**2)**0.5
            if duv>D:
                but_fil[i,j]=0
            else:
                but_fil[i,j]=1
            #but_fil[i,j] = 1/(1+(duv/d)**(2*n))
    
    but_filt_fft = fft_shift*(but_fil)
    but_filtered_img = abs(IFFT2(fftshift(but_filt_fft)))
    plt.imshow(but_filtered_img, cmap='gray')
    plt.show()
    return but_fil

def gaussian_Low_pass_filter(img):
    FFT2_img = FFT2(img)
    #fimg = np.log(np.abs(FFT2_img))
    fft_shift = fftshift(FFT2_img)
    
    but_fil = np.zeros((img.shape))
    M, N =img.shape
    sigam= np.sqrt(np.var(img))

    for i in range(M):
        for j in range(N):
            duv = ((i-(M/2))**2 + (j-(N/2))**2)**0.5
            but_fil[i,j] = np.exp(-duv**2/(2*sigam))
    but_filt_fft = fft_shift*(but_fil)
    but_filtered_img = abs(IFFT2(fftshift(but_filt_fft)))
    plt.imshow(but_filtered_img, cmap='gray')
    plt.show()
    return but_fil

#High pass filter

def idea_high_pass_filter(img, Threshold=50):
    # 做FFT
    FFT2_img = FFT2(img)
    #fimg = np.log(np.abs(FFT2_img))
    fft_shift = fftshift(FFT2_img)
    
    #Low_pass_filter
    but_fil = np.zeros((img.shape))
    M, N =img.shape
    for i in range(M):
        for j in range(N):
            duv = ((i-(M/2))**2 + (j-(N/2))**2)**0.5
            if duv<Threshold:
                but_fil[i,j]=0
            else:
                but_fil[i,j]=1
            #but_fil[i,j] = 1/(1+(duv/d)**(2*n))
    
    but_filt_fft = fft_shift*(but_fil)
    but_filtered_img = abs(IFFT2(fftshift(but_filt_fft)))
    plt.imshow(but_filtered_img, cmap='gray')
    plt.title('Idea high pass', fontsize=10)
    plt.show()
    return but_fil

def Butterworth_high_pass_filter(img, D=50, n=2):
    # 做FFT
    FFT2_img = FFT2(img)
    #fimg = np.log(np.abs(FFT2_img))
    fft_shift = fftshift(FFT2_img)
    
    #Low_pass_filter
    but_fil = np.zeros((img.shape))
    M, N =img.shape
    for i in range(M):
        for j in range(N):
            duv = ((i-(M/2))**2 + (j-(N/2))**2)**0.5
            but_fil[i,j] = 1/(1+(duv/D)**(2*n))
    
    but_filt_fft = fft_shift*(1-but_fil)
    but_filtered_img = abs(IFFT2(fftshift(but_filt_fft)))
    #print(but_filtered_img.min(),but_filtered_img.max())
    but_filtered_img = (but_filtered_img-np.amin(but_filtered_img))/(np.amax(but_filtered_img)-np.amin(but_filtered_img))
    plt.imshow(but_filtered_img, cmap='gray')
    plt.title('Butterworth high pass', fontsize=10)
    plt.show()
    return but_fil

# 邊緣檢測-2 Laplacian
class Laplacian_Filter():
    #初始化
    def __init__(self, kernal_size=3):
        self.kernal_size=kernal_size
    
    # 濾波函數
    def filter(self, img):
        img = np.array(img)
        height = img.shape[0]
        width = img.shape[1]
        newimg = np.zeros((height, width))
        halfk= int((self.kernal_size-1)/2)
        if self.kernal_size ==3:
            template = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]) #建立3*3濾波模板
        elif self.kernal_size ==5:
            #template = np.array([[0,0,1,0,0], [0,0,2,0,0], [1,2,-16,2,1],[0,0,2,0,0],[0,0,1,0,0]]) #建立5*5濾波模板
            template = np.array([[0,0,-1,0,0], [0,0,-2,0,0], [-1,-2,16,-2,-1],[0,0,-2,0,0],[0,0,-1,0,0]])
        
        
        for i in range(halfk, height-halfk):
            for j in range(halfk, width-halfk):
                img_f = img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
                kernal_f = np.multiply(img_f, template)
                newimg[i, j] = kernal_f.sum()
        newImage = Image.fromarray(newimg)
        return newimg
    
def Do_Laplacian_Filter(img, kernal_size):
    if kernal_size%2==0:
        print('Kernal size please enter an odd value！')
    else:
        Lap_Filter=Laplacian_Filter(kernal_size)#宣告高斯模糊類
        image = Lap_Filter.filter(img)
        #print(image)
        #print(image.min(),image.max())
        #image = (image-np.min(image))/(np.max(image)-np.min(image))
        image = np.uint8(image)
        plt.imshow(image, cmap='gray')#圖片顯示
        plt.show()

# Image denoising
# Inverse-Filtering
def Inverse_Filtering(img):
    
    F = FFT2(img)
    F2 = fftshift( F )

    # normalize to [0,1]
    H = img/255.

    # calculate the damaged image
    G = H * F2

    # Inverse Filter 
    F_hat = G / H

    # cheat? replace division by zero (NaN) with zeroes
    a = np.nan_to_num(F_hat)
    f_hat = IFFT2(fftshift(a) )
    f_hat = abs(f_hat)
    plt.imshow(f_hat, cmap='gray')
    plt.show()
    return f_hat

# Wiener filter
# https://github.com/lvxiaoxin/Wiener-filter/blob/master/main.py
# https://github.com/tranleanh/wiener-filter-image-restoration

# 仿真运动模糊
def motion_process(imgsize, len=100 ):
    sx, sy = imgsize
    PSF = np.zeros((sy, sx))
    PSF[int(sy / 2):int(sy /2 + 1), int(sx / 2 - len / 2):int(sx / 2 + len / 2)] = 1
    return PSF / PSF.sum() # 归一化亮度

def make_blurred(input, PSF, eps):
    input_fft = FFT2(input)
    PSF_fft = FFT2(PSF) + eps
    blurred = IFFT2(input_fft * PSF_fft)
    blurred = np.abs(fftshift(blurred))
    return blurred

def wiener(input, PSF, eps):
    input_fft = FFT2(input)
    W = FFT2(PSF) + eps #噪声功率，这是已知的，考虑epsilon
    result = IFFT2(input_fft / W) #计算F(u,v)的傅里叶反变换
    result = np.abs(fftshift(result))
    return result


def wiener_filter(img, noise_r=50):
    PSF = motion_process(img.shape, noise_r)
    blurred = make_blurred(img, PSF, 1e-3)
    result = wiener(blurred, PSF, 1e-3)
    
    plt.subplot(1,2,1)
    plt.imshow(blurred, cmap='gray')
    plt.title('with Motion blurred', fontsize=10)

    plt.subplot(1,2,2)
    plt.imshow(result)
    plt.title('After Wiener Filtering', fontsize=10)
    plt.show()

    return result



