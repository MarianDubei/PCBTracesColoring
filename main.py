from PIL import ImageTk, Image, ImageDraw
from tkinter import filedialog as fd
from tkinter import font as tkFont
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
import numpy as np
import colorsys
import random

import cv2 as cv

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PCB Traces Coloring')
        self.geometry("1000x800")
        self.resizable(width = True, height = True)
        
        self.filename_var = tk.StringVar()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)

        self.create_widgets()

    def create_widgets(self):
        s = ttk.Style()
        s.configure('my.TButton', font=('Segoe UI', 16))
        self.padding = {'padx': 25, 'pady': 25}
        # label
        ttk.Label(self, text='Select image:', font=('Segoe UI', 18)).grid(column=0, row=0, sticky=tk.S, **self.padding)

        # Input
        open_button = ttk.Button(self, text='Open a File', style='my.TButton', command=self.select_file)
        open_button.grid(column=0, row=1, sticky=tk.N, **self.padding)
        
        # Input image label
        self.input_img_label = ttk.Label(self)
    
        # Coloring traces button
        self.color_traces_buton = ttk.Button(self, text='Color Traces', style='my.TButton', command=self.display_colored_image)
    
        # Output image label
        self.output_img_label = ttk.Label(self)
        
        # Save to file button
        self.save_button = ttk.Button(self, text="Save to File", style='my.TButton', command=self.save_to_file)
        
        # Show histogram button
        s2 = ttk.Style()
        s2.configure('my2.TButton', font=('Segoe UI', 12))
        self.hist_button = ttk.Button(self, text="Show Histogram", style='my2.TButton', command=self.plot_hist)
    
    def select_file(self):
        filetypes = (('All Picture Files', '*.png *.jpg *.jpeg *.jpe *.jfif *.bmp'), ('All Files', '*.*'))

        filename = fd.askopenfilename(title='Open a file', initialdir="/", filetypes=filetypes)
        if not filename:
            return
        
        self.filename_var.set(filename)
        
        self.columnconfigure(1, weight=1)
        
        # print image
        img = Image.open(self.filename_var.get())
        baseheight = 250
        hpercent = (baseheight/float(img.size[1]))
        resized_img = img.resize((int((float(img.size[0])*float(hpercent))), baseheight), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_img)
        
        self.input_img_label.configure(image=tk_image)
        self.input_img_label.image = tk_image        
        self.input_img_label.grid(column=1, row=0, rowspan=2, **self.padding)
        
        self.color_traces_buton.grid(column=0, row=2, sticky=tk.S, **self.padding)

    def save_to_file(self):
        filetypes = (('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif'))
        save_filename = fd.asksaveasfile(mode='w', initialdir="/", title='Save to file', filetypes=filetypes, defaultextension=filetypes)
        if not save_filename:
            return
        self.colored_img.save(save_filename.name)

    def display_colored_image(self):
                        
        self.color_traces()
        
        self.colored_img = Image.fromarray(self.np_rgb_img)
        
        baseheight = 250
        hpercent = (baseheight/float(self.colored_img.size[1]))
        resized_rgb_img = self.colored_img.resize((int((float(self.colored_img.size[0])*float(hpercent))), baseheight), Image.Resampling.LANCZOS)
        tk_rgb_image = ImageTk.PhotoImage(resized_rgb_img)
        
        self.output_img_label.configure(image=tk_rgb_image)
        self.output_img_label.image = tk_rgb_image
        self.output_img_label.grid(column=1, row=2, rowspan=3, **self.padding)
        
        self.save_button.grid(column=0, row=3, sticky=tk.S, **self.padding)
        

    def is_valid(self, x, y):
        if x < 0 or x > self.img_width - 1 or\
            y < 0 or y > self.img_height - 1:
            return False
        if self.np_wb_img[x][y] == self.wb_color_to_update or\
            (not self.start_color_range[0] <= self.np_wb_img[x][y] <= self.start_color_range[1]):
            return False
        else:
            return True
            
    def color_pixel(self, x, y, color_to_update):
        self.np_rgb_img[x][y][0] = color_to_update[0] # R
        self.np_rgb_img[x][y][1] = color_to_update[1] # G
        self.np_rgb_img[x][y][2] = color_to_update[2] # B
        self.np_wb_img[x][y] = self.wb_color_to_update

    def hsv2rgb(self, h,s,v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

    def generate_color(self):
        hue_step = 0.02
        rand_hue = np.random.choice(self.color_hue_range)
        rgb_tuple = self.hsv2rgb(rand_hue,0.68,0.8)
        self.color_hue_range = [ item for item in self.color_hue_range if not rand_hue - hue_step <= item <= rand_hue + hue_step ]
        if len(self.color_hue_range) == 0:
            self.color_hue_range = np.arange(0, 1, 0.0001)
        return rgb_tuple

    def flood_fill(self, x,y,color_to_update):
        queue = []
        queue.append([x,y])
        self.color_pixel(x, y, color_to_update)
        while queue:
            current_pixel = queue.pop()
            x = current_pixel[0]
            y = current_pixel[1]
            
            neighbors = [(x-1,y),(x+1,y),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x,y-1),(x,y+1)]
            for neighbor in neighbors:
                if self.is_valid(neighbor[0], neighbor[1]):
                    self.color_pixel(neighbor[0], neighbor[1], color_to_update)
                    queue.append(neighbor)

    def plot_hist(self):
        plot_wb_img = Image.open(self.filename_var.get()).convert('L')
        x = np.array(plot_wb_img).flatten()
        fig, ax = plt.subplots(figsize=(12, 6))
        n, bins, patches = ax.hist(x, 256, density=True, histtype='step',
                           label='Empirical') #, cumulative=True)
        ax.axvline(self.threshold, color='r')
        ax.grid(True)
        ax.set_title('Pixel intensity histogram')
        ax.set_xlabel('Pixel intensity')
        ax.set_ylabel('Pixels')
        plt.show()
    
    def calculate_threshold(self):
        img = cv.imread(self.filename_var.get())
        b,g,r = cv.split(img)
        rgb_img = cv.merge([r,g,b])
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        thresh_value, thresh_img = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        return thresh_value

    def color_traces(self):
        rgb_img = Image.open(self.filename_var.get()).convert("RGB")
        wb_img = Image.open(self.filename_var.get()).convert('L')
        self.np_rgb_img = np.array(rgb_img)
        self.np_wb_img = np.array(wb_img)
        self.color_hue_range = np.arange(0, 1, 0.0001)
        self.img_width = len(self.np_wb_img)
        self.img_height = len(self.np_wb_img[0])
        # selecting color for marking pixels as visited
        unique_wb_colors = np.unique(self.np_wb_img, return_counts=True)
        least_common_color = unique_wb_colors[0][np.argmin(unique_wb_colors[1], axis=0)]
        incr = 1 if least_common_color > 0 else -1
        for i in range(self.img_width):
            for j in range(self.img_height):
                if self.np_wb_img[i][j] == least_common_color:
                    self.np_wb_img[i][j] += incr
        self.wb_color_to_update = least_common_color       
        
        # calculate threshold
        self.threshold = self.calculate_threshold()
        self.hist_button.grid(column=0, row=4, sticky=tk.N, **self.padding)
        
        # detecting traces color 
        num_white_pixels = np.sum(self.np_wb_img >= self.threshold)
        num_black_pixels = np.sum(self.np_wb_img < self.threshold)
        if num_black_pixels >= num_white_pixels:
            self.start_color_range = [self.threshold, 255] # traces are white
        else:
            self.start_color_range = [0, self.threshold] # traces are black
        
        # flood-fill algorithm
        for i in range(self.img_width):
            for j in range(self.img_height):
                # check whether a pixel is a trace
                if not self.is_valid(i,j):
                    continue
                # generaing randomized color for a trace
                new_color = self.generate_color()
                self.flood_fill(i,j,new_color)


if __name__ == "__main__":
    app = App()
    app.mainloop()
