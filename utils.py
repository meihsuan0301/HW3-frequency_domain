import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import matplotlib
#matplotlib.use('TkAgg')

def to_image(img):
    image = Image.fromarray(img) 
    return image
    
def to_array(array):
    image = array.astype('uint8')
    image = Image.fromarray(image)
    return image

# 關於 中間 10*10 pixel
def pixel10(img):
    Y = img[251:261, 251:261]
    return Y

# raw圖片處理
def data_raw_img(path):
    img = np.fromfile(path, dtype='uint8')
    img = img.reshape(512,512)
    return img


# 對數變換、伽馬變換和圖像負片
def Log_tranfrom(c, img):     # 對數變換
    output = c*np.log(1+img)
    output = np.uint8(output+0.5)
    return output

def gamma_trans(img, gamma):  # gamma函式處理
    img = img.astype('uint8')
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立對映表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 顏色值為整數
    gamma_table = cv2.LUT(img, gamma_table)
    gamma_table = np.asarray(gamma_table).astype('float')
    gamma_table = np.uint8(gamma_table)
    return gamma_table  # 圖片顏色查表。另外可以根據光強（顏色）均勻化原則設計自適應演算法。

# def gamma_trans(img, gamma):  # gamma函式處理
#     img = img/255
#     img = img**gamma
#     img = 255*img
#     return img  # 圖片顏色查表。另外可以根據光強（顏色）均勻化原則設計自適應演算法。

def negative_tranfrom(img):  # 圖像負片
    output = 255-img
    return output

def Bilinear_interpolation(new_size, img):
    dst_h, dst_w = new_size  #目標圖高寬
    img_h, img_w = img.shape[:2]  #原圖高寬
    scale_x = float(img_w)/dst_w  #x縮放比率
    scale_y = float(img_h)/dst_h
    
    dst = np.zeros((dst_h, dst_w), dtype = np.uint8)
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 目標在原圖上的座標
            img_x = (dst_x +  0.5) * scale_x -0.5
            img_y = (dst_y +  0.5) * scale_y -0.5
            # 計算在原圖上四個鄰近點的位置
            img_x_0 = int(np.floor(img_x))
            img_y_0 = int(np.floor(img_y))
            img_x_1 = min(img_x_0 + 1, img_w -1)
            img_y_1 = min(img_y_0 + 1, img_h -1)
            
            # 雙線性差值
            value0 = (img_x_1 - img_x) * img[img_y_0, img_x_0] + (img_x - img_x_0) * img[img_y_0, img_x_1]
            value1 = (img_x_1 - img_x) * img[img_y_1, img_x_0] + (img_x - img_x_0) * img[img_y_1, img_x_1]
            dst[dst_y, dst_x] = int((img_y_1 - img_y) * value0 + (img_y - img_y_0) * value1)
    dst = np.uint8(dst)
    return dst

def Nearest_neighbor_interpolation(new_size, img):
    ratio_h = (img.shape[0]-1)/ (new_size[0]-1)
    ratio_w = (img.shape[1]-1)/ (new_size[1]-1)
    target_img = np.zeros((new_size))
    
    for tar_h in range(new_size[0]):
        for tar_w in range(new_size[1]):
            img_h = round(ratio_h*tar_h)
            img_w = round(ratio_w*tar_w)
            target_img[tar_h, tar_w] = img[img_h, img_w]
    target_img = np.uint8(target_img)
    return target_img

# 直方圖均衡
def get_gray_histogram(img):
    """獲取圖像的灰度直方圖"""
    height = img.shape[0]
    width = img.shape[1]
    gray = np.zeros(256)  # 保存各個灰度級（0-255）的出現次數

    for h in range(height):
        for w in range(width):
            gray[img[h][w]] += 1
    # 將直方圖歸一化, 即使用頻率表示直方圖
    gray /= (height * width)  # 保存灰度的出現頻率，即直方圖

    return gray

def plot_histogram(img):
    x = np.arange(0, 256)
    plt.bar(x, img, width=1)
    plt.show()

def show_pic(img, afterimg):

    x = np.arange(0, 256)
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.bar(x, img, width=1)
    plt.title('before', fontsize = 20)
    
    plt.subplot(122)
    plt.bar(x, afterimg, width=1)
    plt.title('after', fontsize = 20)
    plt.show()

def get_gray_cumulative_prop(gray):
    """獲取圖像的累積分佈直方圖，即就P{X<=x}的概率
    - 大X表示隨機變量
    - 小x表示取值邊界
    """
    cum_gray = []
    sum_prop = 0.
    for i in gray:
        sum_prop += i
        cum_gray.append(sum_prop)  # 累計概率求和
    return cum_gray

def pix_fill(img, cum_gray):
    """像素填充"""
    height, width = img.shape
    des_img = np.zeros((height, width), dtype=np.int)  # 定義目標圖像矩陣

    for h in range(height):
        for w in range(width):
            # 把每一個像素點根據累積概率求得均衡化後的像素值
            des_img[h][w] = cum_gray[img[h][w]] * 255
    return des_img

def global_equalization(img):
    """圖像均衡化執行函數"""
    gray = get_gray_histogram(img)  # 獲取圖像的直方圖
    cum_gray = get_gray_cumulative_prop(gray)  # 獲取圖像的累積直方圖
    des_img = pix_fill(img, cum_gray)  # 根據均衡化函數（累積直方圖）將像素映射到新圖片
    
    plt.figure(figsize=(16,8))
    
    plt.subplot(121)
    plt.title('Before global equalization', fontsize=18)
    plt.imshow(img, cmap ='gray')

    plt.subplot(122)
    plt.title('After global equalization', fontsize=18)
    plt.imshow(des_img, cmap ='gray')
    plt.show()

def global_histogram_equalization(img):
    """圖像均衡化執行函數"""
    gray = get_gray_histogram(img)  # 獲取圖像的直方圖
    cum_gray = get_gray_cumulative_prop(gray)  # 獲取圖像的累積直方圖
    des_img = pix_fill(img, cum_gray)  # 根據均衡化函數（累積直方圖）將像素映射到新圖片
    new_gray = get_gray_histogram(des_img)  # 獲取圖像均衡化後的直方圖
    show_pic(gray, new_gray)
    plt.show()
    
def Local_histogram_equalization(img, window_size=3):
    img_size = img.shape
    var = np.var(img)
    mean = np.mean(img)
    k_0 = 0.4
    k_1 = 0.02
    k_2 = 0.4
    E = 4
    LHE_img = img
    for i in range(0,img_size[0]-window_size):
        for j in range(0,img_size[1]-window_size):
            kernel = img[i:i+window_size,j:j+window_size]
            if np.mean(kernel)<=k_0*mean and k_1*var<np.var(kernel)<k_2*var:
                rank = 4 * kernel
                LHE_img[i:i+window_size,j:j+window_size] =  rank      
    #LHE_img = np.array(LHE_img, dtype = np.uint8)
    plt.imshow(LHE_img, cmap ='gray')
    plt.show()
    return LHE_img

def Histogram_matching(img, Target, hist=False):
    Original_hist = get_gray_histogram(img)
    Target_hist = get_gray_histogram(Target)
    
    Original_cdf_gray = get_gray_cumulative_prop(Original_hist)
    Target_cdf_gray = get_gray_cumulative_prop(Target_hist)

    matching_new_gray = pix_fill(img, Target_cdf_gray)
    new_gray_hist = get_gray_histogram(matching_new_gray)
    
    if hist==False:
        plt.figure(figsize=(16,8))
        plt.subplot(121)
        plt.title('Before', fontsize=18)
        plt.imshow(img, cmap ='gray')
        plt.subplot(122)
        plt.title('After', fontsize=18)
        plt.imshow(matching_new_gray, cmap ='gray')
    elif hist:
        show_pic(Original_hist, new_gray_hist)
    return matching_new_gray

# 題二
class Filter():
    #初始化
    def __init__(self, kernal_size=3, sigema=None):
        self.kernal_size=kernal_size
        self.sigema=1.5
    
    # 濾波函數
    def Gaus(self, x, y):
        part1 = 1/(2*np.pi*self.sigema**2)
        part2 = np.exp(-(x**2+y**2)/(2*self.sigema**2))
        return part1*part2
    # 建立濾波模板
    def Gaus_template(self):
        sideLength = self.kernal_size
        mask = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                mask[i, j] = self.Gaus(i-(self.kernal_size-1)/2, j-(self.kernal_size-1)/2) 
        all = np.sum(mask)
        return mask/all
    
    def Averaging_template(self):
        template = np.repeat(1/self.kernal_size**2, self.kernal_size**2).reshape(self.kernal_size,self.kernal_size) #建立濾波模板
        return template
    
    def Sobel_template(self):
        if self.kernal_size ==3:
            template = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])#建立3*3濾波模板
        return template
    
    def Laplacian_template(self):
        if self.kernal_size ==3:
            template = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]) #建立3*3濾波模板
        elif self.kernal_size ==5:
            #template = np.array([[0,0,1,0,0], [0,0,2,0,0], [1,2,-16,2,1],[0,0,2,0,0],[0,0,1,0,0]]) #建立5*5濾波模板
            template = np.array([[0,0,-1,0,0], [0,0,-2,0,0], [-1,-2,16,-2,-1],[0,0,-2,0,0],[0,0,-1,0,0]])
        return template
    
    
    
    def filter(self, img, template):
        img = np.array(img)
        height = img.shape[0]
        width = img.shape[1]
        newimg = np.zeros((height, width))
        halfk= int((self.kernal_size-1)/2)
        for i in range(halfk, height-halfk):
            for j in range(halfk, width-halfk):
                img_f = img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
                kernal_f = np.multiply(img_f, template)
                newimg[i, j] = kernal_f.sum()
        newImage = Image.fromarray(newimg)
        return newImage
    
    
    def Do_Filter(self,img, kernal_size, method, sigema=None):
        if kernal_size%2==0:
            print('Kernal size please enter an odd value！')
        else:
            
            if method=='GaussianBlur':
                temp=self.Gaus_template()
                image=self.filter(img, temp)
            elif method=='Averaging':
                temp=self.Averaging_template()
                image=self.filter(img, temp)
            elif method=='Laplacian':
                temp=self.Laplacian_template()
                image=self.filter(img, temp)
            elif method=='Sobel':
                temp=self.Sobel_template()
                image=self.filter(img, temp)
            elif method=='Laplacian':
                temp=self.Laplacian_template()
                image=self.filter(img, temp)
            elif method=='Unsharp_Maskimg':
                temp=self.Gaus_template()
                unsharp_maskimg=self.filter(img, temp)
                image = np.uint8(img-unsharp_maskimg)
            elif method=='Kernel_1':
                temp= np.array([[-1,0,-1], [0,6,0], [-1,0,-1]])
                image=self.filter(img, temp)
            elif method=='Kernel_2':
                temp=np.array([[1,2,1], [0,5,0], [4,2,4]])*1/25
                image=self.filter(img, temp)
                
            image = np.uint8(image)
            #image=Do_Filter.filter(img, temp)#高斯模糊濾波，得到新的圖片
            plt.imshow(image, cmap='gray')#圖片顯示
            #plt.show()    
        return image 

def statistic_order_filter(img, kernal_size=3, methon='median'):
    img = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    newimg = np.zeros((height, width))
    halfk= int((kernal_size-1)/2)
    for i in range(halfk, height-halfk):
        for j in range(halfk, width-halfk):
            img_f = img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
            if methon=='median':
                newdata = np.median(img_f)
            if methon=='max':
                newdata = np.max(img_f)
            if methon=='min':
                newdata = np.min(img_f)
            #kernal_f = np.multiply(img_f, template)
            newimg[i, j] = newdata
    newImage = Image.fromarray(newimg)
    plt.imshow(newImage, cmap='gray')#圖片顯示
    plt.show()
    return newImage


def Bilateral_filter( img, sigma_c=5, sigma_s=5, kernal_size=3, without_smooth=False ):

    # check the input
#     if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
#         raise ValueError('Expected a 2D numpy.ndarray with float32 elements')
    img = img.astype(np.float32)/255.0

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp(-r2**2/(2*sigma**2)))

    img = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    newimg = np.zeros((height, width))
    halfk= int((kernal_size-1)/2)

    sideLength = kernal_size
    W_c = np.zeros((sideLength, sideLength))
    for i in range(sideLength):
        for j in range(sideLength):
            
                W_c[i, j] = gaussian((i-(kernal_size-1)/2)**2+(j-(kernal_size-1)/2)**2, sigma_c )
            
    newdata = np.zeros((height, width))            
    for i in range(halfk, height-halfk):
        for j in range(halfk, width-halfk):
            r2_s = img[i,j]-img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
            W_s = gaussian(r2_s, sigma_s)
            
            if not without_smooth:
                result = W_c*W_s*img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
            else:
                result = img[i-halfk:i+halfk+1, j-halfk:j+halfk+1]
            wgt_sum = W_c*W_s
            newdata[i, j] = (result.sum()/wgt_sum.sum())
    #newImage = Image.fromarray(newimg)
    plt.imshow(newdata, cmap='gray')#圖片顯示
    plt.show()
    # normalize the result and return
    return newdata


def nonLocalMeans(img, bigWindowSize=20, smallWindowSize=5, h=14):
    padwidth = bigWindowSize//2
    image = img.copy()

    # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
    paddedImage = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
    paddedImage = paddedImage.astype(np.uint8)
    paddedImage[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = image
    paddedImage[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(image[:,0:padwidth])
    paddedImage[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(image[:,image.shape[1]-padwidth:image.shape[1]])
    paddedImage[0:padwidth,:] = np.flipud(paddedImage[padwidth:2*padwidth,:])
    paddedImage[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImage[paddedImage.shape[0] - 2*padwidth:paddedImage.shape[0] - padwidth,:])
  

    iterator = 0
    totalIterations = image.shape[1]*image.shape[0]*(bigWindowSize - smallWindowSize)**2

    outputImage = paddedImage.copy()

    smallhalfwidth = (smallWindowSize-1)//2


    # For each pixel in the actual image, find a area around the pixel that needs to be compared
    for imageX in range(padwidth, padwidth + image.shape[1]):
        for imageY in range(padwidth, padwidth + image.shape[0]):
            bWinX = imageX - padwidth
            bWinY = imageY - padwidth

            #comparison neighbourhood
            compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth ,imageX-smallhalfwidth:imageX+smallhalfwidth]
      
      
            pixelColor = 0
            totalWeight = 0

            # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
            for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
                for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):
                    #find the small box       
                    smallNbhd = paddedImage[sWinY:sWinY+(smallWindowSize-1) ,sWinX:sWinX+(smallWindowSize-1) ]
                    #print(smallNbhd.shape, compNbhd.shape)
                    euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
                    #weight is computed as a weighted softmax over the euclidean distances
                    weight = np.exp(-euclideanDistance/h)
                    totalWeight += weight
                    pixelColor += weight*paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
                    iterator += 1

                    if verbose:
                        percentComplete = iterator*100/totalIterations
                        if percentComplete % 5 == 0:
                            print('% COMPLETE = ', percentComplete)

        pixelColor /= totalWeight
        outputImage[imageY, imageX] = pixelColor
    
    out_img = outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]
    plt.imshow(out_img, cmap='gray')#圖片顯示
    plt.show()
    return out_img

# SINGLE-IMAGE DERAINING USING AN ADAPTIVE NONLOCAL MEANS FILTER
def get_block( win, img):
    h, w, height, width = win
    return img[h:h+height, w:w+width]



def get_bina( win, img, alpha):
    h, w, height, width = win
    imgO = get_block(win, img)
    winh1 = [h + 1, w, height, width]
    winh2 = [h - 1, w, height, width]
    winw1 = [h, w + 1, height, width]
    winw2 = [h, w - 1, height, width]
    imgH = (get_block(winh1, img) + get_block(winh2, img)) / 2
    imgW = (get_block(winw1, img) + get_block(winw2, img)) / 2
    imgC = (imgH + imgW) / 4
    outputs = (np.abs(imgO - imgC) > alpha) + 0.
    return outputs

def nonLocalMeans_improve(img, bigWindowSize=20, smallWindowSize=5, sigma=0.5, alpha=0.6):
    padwidth = bigWindowSize//2
    img = img / 255
    var = sigma ** 2
    h, w = img.shape
    image = img.copy()

    # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
    paddedImage = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
    paddedImage = paddedImage.astype(np.uint8)
    paddedImage[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = image
    paddedImage[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(image[:,0:padwidth])
    paddedImage[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(image[:,image.shape[1]-padwidth:image.shape[1]])
    paddedImage[0:padwidth,:] = np.flipud(paddedImage[padwidth:2*padwidth,:])
    paddedImage[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImage[paddedImage.shape[0] - 2*padwidth:paddedImage.shape[0] - padwidth,:])
  
    win = [padwidth, padwidth, h, w]
    binmap = get_bina(win, paddedImage, alpha)
    paddedImagemap = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
    paddedImagemap = paddedImage.astype(np.uint8)
    paddedImagemap[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = binmap
    paddedImagemap[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(binmap[:,0:padwidth])
    paddedImagemap[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(binmap[:,image.shape[1]-padwidth:image.shape[1]])
    paddedImagemap[0:padwidth,:] = np.flipud(paddedImagemap[padwidth:2*padwidth,:])
    paddedImagemap[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImagemap[paddedImagemap.shape[0] - 2*padwidth:paddedImagemap.shape[0] - padwidth,:])
  

    iterator = 0
    totalIterations = image.shape[1]*image.shape[0]*(bigWindowSize - smallWindowSize)**2
    outputImage = paddedImage.copy()
    smallhalfwidth = (smallWindowSize-1)//2

    for imgh in range(h):
        for imgw in range(w):
            imgPh, imgPw = imgh + padwidth, imgw + padwidth
            Swindow = paddedImage[imgPh-smallhalfwidth:imgPh+smallhalfwidth+1, imgPw-smallhalfwidth:imgPw+smallhalfwidth+1]
            Smap = (-1) * (1 - paddedImagemap[imgPh-smallhalfwidth:imgPh+smallhalfwidth+1, imgPw-smallhalfwidth:imgPw+smallhalfwidth+1])

            hp, wp = imgPh - padwidth, imgPw - padwidth

            weightSum = 1e-6
            valueSum = 0.
            for hi in range(hp, hp + bigWindowSize - smallWindowSize):
                for wi in range(wp, wp + bigWindowSize - smallWindowSize):
                    Bwindow = paddedImage[hi:hi + smallWindowSize, wi:wi + smallWindowSize]
                    Bmap = (-1) * (1 - paddedImagemap[hi:hi + smallWindowSize, wi:wi + smallWindowSize])
                    
                    maps = Smap * Bmap
                    N = (maps == 1.).sum() + 1e-6

                    similarty = np.sqrt((((Bwindow - Swindow) * maps) ** 2).sum()) / (N * 2 * var) * (-1)
                    weight = np.exp(similarty)
                    value = weight * paddedImage[hi + smallhalfwidth, wi + smallhalfwidth]
                    valueSum += value
                    weightSum += weight

            c = valueSum / weightSum
            outputImage[imgh, imgw] = c
    outputImage = outputImage*255.
    #out_img = outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]
    plt.imshow(outputImage, cmap='gray')#圖片顯示
    plt.show()
    return outputImage

    