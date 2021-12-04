UI檔案存在google雲端中，下載連結如下：
https://drive.google.com/file/d/1e2J_3KMxQYu47Sn1x-qQrWz53ZqvB2dr/view?usp=sharing

使用UI_Windows說明

注意：在Frequency domain下拉視窗中的操作，因為涉及頻率域的轉換，特徵空間會被更改，所以圖片以彈跳視窗方式展現，且執行完後須重新開啟照片，以進行下一個操作

先點選匯入圖片，再輸入打算執行的轉換的參數，在欄中輸入完參數後，即可點選想要進行的轉換鈕，
其中參數填寫方式如下：

log：c (ex 42)

gamma：r (ex 1.2)

negative：空

Bilinear：h, w (ex 128,128 )

Nearest：h, w (ex 128,128 )

GaussianBlur：kernel size, sigema (ex 3 , 1.5)

Averaging Filter：kernel size (ex 5)

Unsharp Maskimg：kernel size, method (ex 3, 'Gaussian_Blur')

add_noise：distribution (ex 0)
           make the noise by distribution，0：Gaussian, 1：uniform

idea_high_pass_filter：Threshold (ex 50)

Butterworth_high_pass_filter：D, n (ex 50,2)

wiener_filter：noise_r (ex 100)

Guided_Filter：r, eps (ex 1,0.001)

