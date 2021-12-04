import numpy as np
import matplotlib.pyplot as plt
import math

def FFT(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

def IFFT(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    #print(X.shape)

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        factor = np.exp(1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    X = X.ravel()
    X = X / X.shape[0]
    #print( X.shape[0])
    return X


# Fast fourier transform 2D

def FFT2(img):
    FFT_along_row = np.apply_along_axis(FFT, axis=1, arr=img)
    FFT_along_row_col = np.apply_along_axis(FFT, axis=0, arr=FFT_along_row)
    return FFT_along_row_col

def IFFT2(img):
    IFFT_along_row = np.apply_along_axis(IFFT, axis=1, arr=img)
    IFFT_along_row_col = np.apply_along_axis(IFFT, axis=0, arr=IFFT_along_row)
    return IFFT_along_row_col

def FFT2_plot(img):
    FFT_along_row = np.apply_along_axis(FFT, axis=1, arr=img)
    FFT_along_row_col = np.apply_along_axis(FFT, axis=0, arr=FFT_along_row)

    fimg = np.log(np.abs(FFT_along_row_col))
    F = fftshift(fimg)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(fimg,cmap='gray')
    plt.title('FFT', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.imshow(F,cmap='gray')
    plt.title('FFT shift', fontsize=10)
    plt.show()
    return FFT_along_row_col

def IFFT2_plot(img):
    FFT_along_row = np.apply_along_axis(FFT, axis=1, arr=img)
    FFT_along_row_col = np.apply_along_axis(FFT, axis=0, arr=FFT_along_row)

    IFFT_along_row = np.apply_along_axis(IFFT, axis=1, arr=FFT_along_row_col)
    IFFT_along_row_col = np.apply_along_axis(IFFT, axis=0, arr=IFFT_along_row)
    plt.imshow(IFFT_along_row_col,cmap='gray')
    plt.title('IFFT', fontsize=10)
    plt.show()
    return IFFT_along_row_col



def fftshift(F):
    M, N = F.shape
    R1, R2 = F[0: int(M/2), 0: int(N/2)], F[int(M/2): M, 0: int(N/2)]
    R3, R4 = F[0: int(M/2), int(N/2): N], F[int(M/2): M, int(N/2): N]
    sF = np.zeros(F.shape,dtype = F.dtype)
    sF[int(M/2): M, int(N/2): N], sF[0: int(M/2), 0: int(N/2)] = R1, R4
    sF[int(M/2): M, 0: int(N/2)], sF[0: int(M/2), int(N/2): N]= R3, R2
    return sF

def plot_fftdomain(img):
    FFT2_img = FFT2(img)
    fimg = np.log(np.abs(FFT2_img))
    F = fftshift(fimg)
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(fimg,cmap='gray')
    plt.title('FFT', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.imshow(F,cmap='gray')
    plt.title('FFT shift', fontsize=10)
    plt.show()
    return FFT2_img

def plot_ffthist(img):
    FFT2_img = FFT2(img)
    fimg = np.log(np.abs(FFT2_img))
    F = fftshift(fimg)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.hist(fimg)
    plt.title('Histogram of spectrum', fontsize=10)
    plt.show()

    return FFT2_img