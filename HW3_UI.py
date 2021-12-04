import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk
from PIL import ImageTk,Image
import numpy as np
#import os
#from os import listdir
import numpy as np
import cv2
from utils import *
# import matplotlib
from PIL import Image, ImageTk
from PIL.Image import fromarray
import matplotlib.pyplot as plt

from FFT import *
from filter import *
from DCT import *
from BM3D import *
from Guided_Filter import *
#plt.switch_backend('agg')

class UI_Window:
    def __init__(self, name='UI'):
        self.name = name
        self.window = tk.Tk()
        
        self.window.title('image transfers')
        self.window.geometry('700x700')
        
        self.canvas = tk.Canvas(self.window, width=700, height=630, bg="white")
        self.canvas.place(x=0, y=70)
        
        self.label = tk.Label(self.window, text='請輸入參數:')
        self.label.place(x=400, y=0)

        self.parameter = tk.Text(self.window, width=20, height=1)
        self.parameter.place(x=400, y=20)
        
        self.menubar = Menu(self.window)
        self.file_menu = Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Open Image", command=self.open_img)
        # file_menu.add_command(label="Save Image", command=save_image)
        #self.file_menu.add_separator()
        # file_menu.add_command(label="Exit", command=gui.destroy)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.opr_menu = Menu(self.menubar, tearoff=0)
        self.opr_menu.add_command(label="10pixel", command=self.transfer_funs('10pixel'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="log", command=self.transfer_funs('log'))
        self.opr_menu.add_command(label="gamma", command=self.transfer_funs('gamma'))
        self.opr_menu.add_command(label="negative", command=self.transfer_funs('negative'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="bilinear", command=self.transfer_funs('bilinear'))
        self.opr_menu.add_command(label="nearest", command=self.transfer_funs('nearest'))
        self.menubar.add_cascade(label="Operation", menu=self.opr_menu)
        self.window.config(menu=self.menubar)

        self.opr_menu = Menu(self.menubar, tearoff=0)
        self.opr_menu.add_command(label="global_equalization", command=self.filter_funs('global_equalization'))
        self.opr_menu.add_command(label="global_histogram_equalization", command=self.filter_funs('global_histogram_equalization'))
        self.opr_menu.add_command(label="Local_histogram_equalization", command=self.filter_funs('Local_histogram_equalization'))
        self.opr_menu.add_command(label="Histogram_matching", command=self.filter_funs('Histogram_matching'))        
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="GaussianBlur", command=self.filter_funs('GaussianBlur'))
        self.opr_menu.add_command(label="Averaging", command=self.filter_funs('Averaging'))
        self.opr_menu.add_command(label="Laplacian", command=self.filter_funs('Laplacian'))
        self.opr_menu.add_command(label="Sobel", command=self.filter_funs('Sobel'))
        self.opr_menu.add_command(label="Unsharp_Maskimg", command=self.filter_funs('Unsharp_Maskimg'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="Kernel_1", command=self.filter_funs('Kernel_1'))
        self.opr_menu.add_command(label="Kernel_2", command=self.filter_funs('Kernel_2'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="statistic_order", command=self.filter_funs('statistic_order'))
        self.opr_menu.add_command(label="Bilateral", command=self.filter_funs('Bilateral_filter'))
        self.opr_menu.add_command(label="nonLocalMeans", command=self.filter_funs('nonLocalMeans'))
        self.opr_menu.add_command(label="nonLocalMeans_improve", command=self.filter_funs('nonLocalMeans_improve'))
        self.menubar.add_cascade(label="Filter", menu=self.opr_menu)
        self.window.config(menu=self.menubar)
        
        self.opr_menu = Menu(self.menubar, tearoff=0)
        self.opr_menu.add_command(label="FFT", command=self.frequency_funs('FFT2_plot'))
        self.opr_menu.add_command(label="IFFT", command=self.frequency_funs('IFFT2_plot'))
        self.opr_menu.add_command(label="plot_ffthist", command=self.frequency_funs('plot_ffthist'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="add_noise", command=self.frequency_funs('add_noise'))
        self.opr_menu.add_command(label="Idea_Low_pass_filter", command=self.frequency_funs('idea_Low_pass_filter'))
        self.opr_menu.add_command(label="Gaussian_Low_pass_filter", command=self.frequency_funs('gaussian_Low_pass_filter'))
        self.opr_menu.add_command(label="Idea_high_pass_filter", command=self.frequency_funs('idea_high_pass_filter'))
        self.opr_menu.add_command(label="Butterworth_high_pass_filter", command=self.frequency_funs('Butterworth_high_pass_filter'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="Inverse_Filtering", command=self.frequency_funs('Inverse_Filtering'))
        self.opr_menu.add_command(label="Wiener_filter", command=self.frequency_funs('wiener_filter'))
        self.opr_menu.add_command(label="BM3D", command=self.frequency_funs('BM3D'))
        self.opr_menu.add_command(label="Guided_Filter", command=self.frequency_funs('Guided_Filter'))
        self.opr_menu.add_separator()
        self.opr_menu.add_command(label="DCT", command=self.frequency_funs('DCT2_plot'))
        self.opr_menu.add_command(label="IDCT", command=self.frequency_funs('IDCT2_plot'))
        self.opr_menu.add_command(label="DCT_ideal_low_pass", command=self.frequency_funs('DCT_ideal_low_pass'))
        self.menubar.add_cascade(label="Frequency domain", menu=self.opr_menu)
        self.window.config(menu=self.menubar)
#         self.btn = tk.Button(self.window, text='open image', command=self.open_img)
#         self.btn.place(x=0, y=0)
    
    
    def open_img(self):
        #filename = filedialog.askopenfilename(title='open')
        filename = filedialog.askopenfile(mode='r')
        fileType = filename.name.split('.')[-1]
        if fileType == 'raw':
            img = np.fromfile(filename.name, dtype=np.uint8).reshape(512, 512)
            print(filename.name)
            self.img = Image.fromarray(img)      
        else:
            self.img = Image.open(filename.name)
    
#     def save_img(self):
#         imgPath = filedialog.asksaveasfile()
#         path = imgPath.name
#         self.img.save(path)
            
        #img = img.resize((250, 250), Image.ANTIALIAS)
#         self.img = ImageTk.PhotoImage(self.img)
#         panel = Label(self.window, image = self.img)
#         panel.image = self.img
        imgCanvas = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
        self.canvas.image = imgCanvas
        
        
            
    def transfer_funs(self, t):
            def transfer_fun():
                if t == '10pixel':
                    trans = pixel10
                    para = {}
                    
                if t == 'log':
                    trans = Log_tranfrom
                    para = {'c': float(self.parameter.get(1.0, tk.END))}

                if t == 'gamma':
                    trans = gamma_trans
                    para = {'gamma': float(self.parameter.get(1.0, tk.END))}

                if t == 'negative':
                    trans = negative_tranfrom
                    para = {}

                if t == 'bilinear':
                    trans = Bilinear_interpolation
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p = list(map(int, p))
                    para = {'new_size': p}

                if t == 'nearest':
                    trans = Nearest_neighbor_interpolation
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p = list(map(int, p))
                    para = {'new_size': p}
                    
                img = np.asarray(self.img).astype('float')
                img = trans(img = img, **para)
                img = np.uint8(img)
                self.img = Image.fromarray(img)
                imgCanvas = ImageTk.PhotoImage(self.img)
                image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
                self.canvas.image = imgCanvas
            return transfer_fun

    
    def filter_funs(self, t):
            def filter_fun():
                do_Filter=Filter(kernal_size=3, sigema=None)

                if t == 'global_equalization':
                    trans = global_equalization
                    para = {}
                
                if t == 'global_histogram_equalization':
                    trans = global_histogram_equalization
                    para = {}
                    
                if t == 'Local_histogram_equalization':
                    trans = Local_histogram_equalization
                    para = {'window_size': float(self.parameter.get(1.0, tk.END))}

                if t == 'Histogram_matching':
                    trans = Histogram_matching
                    flower = np.fromfile(filedialog.askopenfile(mode='r'), dtype=np.uint8).reshape(512,512)
                    para = {'Target': flower}
                    

                if t == 'GaussianBlur':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'GaussianBlur',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}

                if t == 'Averaging':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Averaging',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}

                if t == 'Laplacian':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Laplacian',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}
                
                if t == 'Sobel':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Sobel',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}
                
                if t == 'Unsharp_Maskimg':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Unsharp_Maskimg',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}

                if t == 'Kernel_1':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Kernel_1',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}

                if t == 'Kernel_2':
                    trans = do_Filter.Do_Filter
                    para = {'method': 'Kernel_2',
                            'kernal_size': float(self.parameter.get(1.0, tk.END))}
                    

                if t == 'statistic_order':
                    trans = statistic_order_filter
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p[0] = int(p[0])
                    p[1] = str(p[1][:-1])
                    para = {'kernal_size': p[0],
                            'methon':p[1]}

                if t == 'Bilateral_filter':
                    trans = Bilateral_filter
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p[0] = int(p[0])
                    p[1] = int(p[1])
                    p[2] = int(p[2])
                    p[3] = str(p[3][:-1])
                    para = {'sigma_c':p[0] ,
                            'sigma_s':p[1],
                            'kernal_size':p[2],
                            'without_smooth':p[3]}
                    
                if t == 'nonLocalMeans':
                    trans = nonLocalMeans
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p[0] = int(p[0])
                    p[1] = int(p[1])
                    p[2] = int(p[2])
                    
                    para = {'bigWindowSize':p[0] ,
                            'smallWindowSize':p[1],
                            'h':p[2]}

                if t == 'nonLocalMeans_improve':
                    trans = nonLocalMeans
                    p = self.parameter.get(1.0, tk.END).split(',')
                    p[0] = int(p[0])
                    p[1] = int(p[1])
                    p[2] = int(p[2])
                    p[3] = int(p[3])

                    para = {'bigWindowSize':p[0] ,
                            'smallWindowSize':p[1],
                            'sigma':p[2],
                            'alpha':p[3]}


                img = np.asarray(self.img).astype('uint8')
                img = trans(img = img, **para)
                #img = np.uint8(img)
                self.img = Image.fromarray(img)
                imgCanvas = ImageTk.PhotoImage(self.img)
                image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
                self.canvas.image = imgCanvas
            return filter_fun

    def frequency_funs(self, t):
        def frequency_fun():
            BM3D_model = BM3D()
            if t == 'FFT2_plot':
                trans = FFT2_plot
                para = {}
                
            if t == 'IFFT2_plot':
                trans = IFFT2_plot
                para = {}

            if t == 'plot_ffthist':
                trans = plot_ffthist
                para = {}

            if t == 'add_noise':
                trans = add_noise
                para = {'distribution': float(self.parameter.get(1.0, tk.END))}

            if t == 'idea_Low_pass_filter':
                trans = idea_Low_pass_filter
                para = {'D': float(self.parameter.get(1.0, tk.END))}

            if t == 'gaussian_Low_pass_filter':
                trans = gaussian_Low_pass_filter
                para = {}

            if t == 'idea_high_pass_filter':
                trans = idea_high_pass_filter
                para = {'Threshold': float(self.parameter.get(1.0, tk.END))}

            if t == 'Butterworth_high_pass_filter':
                trans = Butterworth_high_pass_filter
                p = self.parameter.get(1.0, tk.END).split(',')
                p[0] = int(p[0])
                p[1] = int(p[1])
                para = {'D':p[0] ,
                        'n':p[1]}
            
            if t == 'Inverse_Filtering':
                trans = Inverse_Filtering
                para = {}

            if t == 'wiener_filter':
                trans = wiener_filter
                para = {'noise_r': float(self.parameter.get(1.0, tk.END))}

            if t == 'BM3D':
                trans = BM3D_model.BM3D
                para = {}
            
            if t == 'Guided_Filter':
                trans = Guided_Filter
                p = self.parameter.get(1.0, tk.END).split(',')
                p[0] = int(p[0])
                p[1] = float(p[1][:-1])
                para = {'r':p[0] ,
                        'eps':p[1]}

            if t == 'DCT2_plot':
                trans = DCT2_plot
                para = {}
            
            if t == 'IDCT2_plot':
                trans = IDCT2_plot
                para = {}

            if t == 'DCT_ideal_low_pass':
                trans = DCT_ideal_low_pass
                para = {}
                
            img = np.asarray(self.img).astype('float')
            img = trans(img = img, **para)
            # img = np.uint8(img)
            self.img = Image.fromarray(img)
            imgCanvas = ImageTk.PhotoImage(self.img)
            image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
            self.canvas.image = imgCanvas
        return frequency_fun

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    window = UI_Window()
    window.run()