import random
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import numpy as np
import math
import shutil
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def all_label_location(label_string, background):

    bg_height, bg_width = background.shape[:2]
    # with open(txt_path,'r') as f:
    #         data = f.read()

    data_list = label_string[:-1].split("\n")
    x = []
    y = []
    w = []
    h = []

    xmax = []
    xmin = []
    ymax = []
    ymin = []
    name = []
    for i in range(len(data_list)):
            name.append(data_list[i].split(" ")[0])
            x.append(float(data_list[i].split(" ")[1]))
            y.append(float(data_list[i].split(" ")[2]))
            w.append(float(data_list[i].split(" ")[3]))
            h.append(float(data_list[i].split(" ")[4]))

    for i in range(len(x)):
          xmax.append(round((x[i]+w[i]/2)*bg_width)-1)
          xmin.append(round((x[i]-w[i]/2)*bg_width)-1)
          ymax.append(round((y[i]+h[i]/2)*bg_height)-1)
          ymin.append(round((y[i]-h[i]/2)*bg_height)-1)
          
    return xmax, xmin, ymax, ymin, name
def xymaxmin_bbox(image, txt_path):
    
    with open(txt_path,'r') as f:
        data = f.read()

    shape_list = data[:-2].split(' ')[1:5]
    imgh, imgw = image.shape[:2]
    
    x = float(shape_list[0])
    y = float(shape_list[1])
    w = float(shape_list[2])
    h = float(shape_list[3])

    xmin = int((x-w/2)*imgw)
    xmax = int((x+w/2)*imgw)
    ymin = int((y - h/2)*imgh)
    ymax = int((y + h/2)*imgh)
    
    point = [xmin, xmax, ymin, ymax, imgw, imgh]
    
    return point 
def read_or_write_txt(txt_path='', mode = '', appended_label = ''):
     
    with open(txt_path, mode) as file:
         if mode == 'r':
              data = file.read()
              return data
         elif mode == 'a':
              file.write(appended_label)

def cut_bbox(image, xmax, xmin, ymax, ymin, name, result_path, num):
    
    for i in range(len(xmax)):
        img2 = image[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        file_path = result_path+"_"+name[i]+"_"+str(num)+".png"
        cv2.imwrite(file_path, img2)
        num += 1
    
    return num
    


if __name__ == "__main__":
 
    print("go")
    file_initial = "examples/0731/sort_img/val/fan/"

    #file_result = "examples/0731/images/train/"

    img_label = "examples/0731/sort_label/val/fan/"

    #result_label = "examples/0731/labels/train/"

    img_names = []
    label_names = []
    [img_names.append(img_name) for img_name in Path(file_initial).glob('*.png')]
    [label_names.append(label_name) for label_name in Path(img_label).glob('*.txt')] 

    number = 1
    for i in range(len(img_names)):

        txt_data = read_or_write_txt(txt_path=label_names[i], mode='r')
        image_initial = cv2.imread(str(img_names[i]))

        image_copy = image_initial.copy()

        xmax0, xmin0, ymax0, ymin0, name0 = all_label_location(txt_data, image_copy)
        number = cut_bbox(image = image_copy, xmax = xmax0, xmin = xmin0, ymax = ymax0, ymin = ymin0, name = name0, result_path = "examples/cut/turbo/val/fan/lamp", num = number)




