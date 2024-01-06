import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
img_path = 'examples/0503T/FAN10_thermal.png'
txt_path = "examples/0503T/FAN10_thermal.txt"
output_path = "examples/FAN_cut/FAN"
def cut_bbox(image, txt_path):
    img = cv2.imread(image)
    with open(txt_path,'r') as f:
        data = f.read()

    shape_list = data[:-2].split(' ')[1:5]
    imgh, imgw = img.shape[:2]

    x = float(shape_list[0])
    y = float(shape_list[1])
    w = float(shape_list[2])
    h = float(shape_list[3])

    xmin = int((x-w/2)*imgw)
    xmax = int((x+w/2)*imgw)
    ymin = int((y - h/2)*imgh)
    ymax = int((y + h/2)*imgh)

    img2 = img[ymin:ymax, xmin:xmax]
    return img2

def all_label_location(txt_path, background):

    bg_height, bg_width = background.shape[:2]
    with open(txt_path,'r') as f:
            data = f.read()

    data_list = data[:-1].split("\n")
    x = []
    y = []
    w = []
    h = []

    xmax = []
    xmin = []
    ymax = []
    ymin = []
    
    for i in range(len(data_list)):
            x.append(float(data_list[i].split(" ")[1]))
            y.append(float(data_list[i].split(" ")[2]))
            w.append(float(data_list[i].split(" ")[3]))
            h.append(float(data_list[i].split(" ")[4]))

    for i in range(len(x)):
          xmax.append(round((x[i]+w[i]/2)*bg_width))
          xmin.append(round((x[i]-w[i]/2)*bg_width))
          ymax.append(round((y[i]+h[i]/2)*bg_height))
          ymin.append(round((y[i]-h[i]/2)*bg_height))
    
    return xmax, xmin, ymax, ymin

file_initial = "examples/FAN"

file_result = "examples/FAN_cut"

img_label = "examples/label"
counter = 1
img_names = []
label_names = []
[img_names.append(img_name) for img_name in Path(file_initial).glob('*.png')]
[label_names.append(label_name) for label_name in Path(img_label).glob('FAN*.txt')]

for i in range(len(img_names)):
    bg_img = cv2.imread(str(img_names[i]))
    xmax, xmin, ymax, ymin = all_label_location(label_names[i], bg_img)

    for j in range(len(xmax)):
         cut_img = bg_img[ymin[j]:ymax[j], xmin[j]:xmax[j]]
         cv2.imwrite(file_result+"/FAN"+str(counter)+"_cut.png", cut_img)
         counter += 1

