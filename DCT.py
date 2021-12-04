import numpy as np
import matplotlib.pyplot as plt
import math

def dct(audio_signal):
    
    # Get the number of samples
    window_length = len(audio_signal)

    # Pre-process the signal to make the DCT-I matrix orthogonal
    # (copy the signal to avoid modifying it outside of the function)
    audio_signal = audio_signal.copy()
    audio_signal[[0, -1]] = audio_signal[[0, -1]] * np.sqrt(2)

    # Compute the DCT-I using the FFT
    audio_dct = np.concatenate((audio_signal, audio_signal[-2:0:-1]))
    audio_dct = np.fft.fft(audio_dct)
    audio_dct = np.real(audio_dct[0:window_length]) / 2

    # Post-process the results to make the DCT-I matrix orthogonal
    audio_dct[[0, -1]] = audio_dct[[0, -1]] / np.sqrt(2)
    audio_dct = audio_dct * np.sqrt(2 / (window_length - 1))

    return audio_dct

def Idct(audio_signal):
    
    # Get the number of samples
    window_length = len(audio_signal)

    # Pre-process the signal to make the DCT-I matrix orthogonal
    # (copy the signal to avoid modifying it outside of the function)
    audio_signal = audio_signal.copy()
    audio_signal[[0, -1]] = audio_signal[[0, -1]] * np.sqrt(2)

    # Compute the DCT-I using the FFT
    audio_dct = np.concatenate((audio_signal, audio_signal[-2:0:-1]))
    audio_dct = np.fft.fft(audio_dct)
    audio_dct = np.real(audio_dct[0:window_length]) / 2

    # Post-process the results to make the DCT-I matrix orthogonal
    audio_dct[[0, -1]] = audio_dct[[0, -1]] / np.sqrt(2)
    audio_dct = audio_dct * np.sqrt(2 / (window_length - 1))
    
    audio_dct = audio_dct/audio_dct.shape[0]

    return audio_dct

# Fast fourier transform 2D
def DCT2(img):
    DCT_along_row = np.apply_along_axis(dct, axis=1, arr=img)
    DCT_along_row_col = np.apply_along_axis(dct, axis=0, arr=DCT_along_row)
    fimg = np.log(np.abs(DCT_along_row_col))
    return DCT_along_row_col

def IDCT2(img):
    IDCT_along_row = np.apply_along_axis(Idct, axis=1, arr=img)
    IDCT_along_row_col = np.apply_along_axis(Idct, axis=0, arr=IDCT_along_row)
    return IDCT_along_row_col

def DCT2_plot(img):
    DCT_along_row = np.apply_along_axis(dct, axis=1, arr=img)
    DCT_along_row_col = np.apply_along_axis(dct, axis=0, arr=DCT_along_row)
    fimg = np.log(np.abs(DCT_along_row_col))
    plt.imshow(fimg,cmap='gray')
    plt.title('DCT', fontsize=10)
    plt.show()
    return DCT_along_row_col

def IDCT2_plot(img):
    IDCT_along_row = np.apply_along_axis(Idct, axis=1, arr=img)
    IDCT_along_row_col = np.apply_along_axis(Idct, axis=0, arr=IDCT_along_row)
    plt.imshow(IDCT_along_row_col,cmap='gray')
    plt.show()
    return IDCT_along_row_col

def DCT_ideal_low_pass(img):
    a= DCT2(img)
    mask = np.zeros((img.shape))
    h, w = img.shape
    mask[0:int(3*h/4),0:int(3*w/4)] = 1.
    I = mask*a
    b= IDCT2(I )
    plt.imshow(b,cmap='gray')
    plt.title('After DCT ideal low pass', fontsize=10)
    plt.show()
    return b