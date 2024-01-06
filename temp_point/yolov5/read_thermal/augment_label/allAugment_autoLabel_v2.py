import random
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import numpy as np
import math
import shutil
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def rotate_image(image, label_box_list=[], angle=90, color=(0, 0, 0), img_scale=1.0):
    
    height_ori, width_ori = image.shape[:2]
    x_center_ori, y_center_ori = (width_ori // 2, height_ori // 2)
 
    rotation_matrix = cv2.getRotationMatrix2D((x_center_ori, y_center_ori), angle, img_scale)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
 
    width_new = int((height_ori * sin) + (width_ori * cos))
    height_new = int((height_ori * cos) + (width_ori * sin))
 
    
    rotation_matrix[0, 2] += (width_new / 2) - x_center_ori
    rotation_matrix[1, 2] += (height_new / 2) - y_center_ori
    
    image_new = cv2.warpAffine(image, rotation_matrix, (width_new, height_new), borderValue=(20, 25, 20))
    return image_new

def scale_img(image, fx = 1.2, fy = 0.8):
    M = np.array([[fx,0,0],[0,fy,0]],dtype=float)
    #读取图像
    #img = cv2.imread(image)
    height,width = image.shape[:2]
    #将图片由BGR转为RGB
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #定义缩放后图片的大小
    scale_img = cv2.warpAffine(image,M,(int(width*fx),int(height*fy)))
    return scale_img

def flip(image, flip_type):
    #水平翻转
    horizontal_flip_img = cv2.flip(image,1)
    #垂直翻转
    vertical_flip_img = cv2.flip(image,0)
    #镜像翻转
    mirror_flip_img = cv2.flip(image,-1)
    if flip_type >= 0.6:
        return horizontal_flip_img
    elif flip_type >= 0.2:
        return vertical_flip_img
    elif flip_type <0.2:
        return mirror_flip_img
def translation_img(image, dx = 1, dy = 1):
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
    dsize = image.shape[:2][::-1]
    translation_img = cv2.warpAffine(image, M, dsize, borderValue=(0, 0, 0))
    return translation_img
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

def change_point_flip(img_width, img_height, xmax, xmin, ymax, ymin, name, ftype):
    
     
     

    for i in range(len(xmax)):
        if ftype >= 0.6:
            new_xmax = img_width-1-xmin[i]
            new_xmin = img_width-1-xmax[i]
            xmax[i] = new_xmax
            xmin[i] = new_xmin

            if xmax[i] >= img_width:
              print("xmax[i", xmax[i])

        elif ftype >= 0.2:
            new_ymax = img_height-1-ymin[i]
            new_ymin = img_height-1-ymax[i]
            ymax[i] = new_ymax
            ymin[i] = new_ymin

        elif ftype <0.2:
            new_xmax = img_width-1-xmin[i]
            new_xmin = img_width-1-xmax[i]
            new_ymax = img_height-1-ymin[i]
            new_ymin = img_height-1-ymax[i]
            xmax[i] = new_xmax
            xmin[i] = new_xmin
            ymax[i] = new_ymax
            ymin[i] = new_ymin
    return xmax, xmin, ymax, ymin, name
def change_point_translation(img_height, img_width, xmax, xmin, ymax, ymin , name, dx, dy):

    xmax_new = []
    xmin_new = []
    ymax_new = []
    ymin_new = []
    name_new = []

    for i in range(len(xmax)):
        xmax[i] += dx
        xmin[i] += dx
        ymax[i] += dy
        ymin[i] += dy

        if xmax[i] >= img_width:
            xmax[i] = img_width-1

        if xmax[i] < 0:
            xmax[i] = 0
        
        if xmin[i] >= img_width:
            xmin[i] = img_width-1

        if xmin[i] < 0:
            xmin[i] = 0

        if ymax[i] >= img_height:
            ymax[i] = img_height-1

        if ymax[i] < 0:
            ymax[i] = 0
        
        if ymin[i] >= img_height:
            ymin[i] = img_height-1

        if ymin[i] < 0:
            ymin[i] = 0

        if(xmax[i]!=xmin[i] and ymax[i]!=ymin[i]):
            xmax_new.append(xmax[i])
            xmin_new.append(xmin[i])
            ymax_new.append(ymax[i])
            ymin_new.append(ymin[i])
            name_new.append(name[i])
        
    return xmax_new, xmin_new, ymax_new, ymin_new, name_new
def change_point_rotate(img_height, img_width, xmax, xmin, ymax, ymin , name, angle_0):

    x_center_ori, y_center_ori = (img_width // 2, img_height // 2)
 
    rotation_matrix = cv2.getRotationMatrix2D((x_center_ori, y_center_ori), angle_0, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
 
    # compute the new bounding dimensions of the image
    width_new = int((img_height * sin) + (img_width * cos))
    height_new = int((img_height * cos) + (img_width * sin))
 
    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (width_new / 2) - x_center_ori
    rotation_matrix[1, 2] += (height_new / 2) - y_center_ori
    

    for i in range(len(xmax)):
        
        check_matrix = np.zeros((img_height, img_width, 3))
        
        check_matrix[ymin[i], xmin[i], 0] = 255
        check_matrix[ymax[i], xmin[i], 0] = 255
        check_matrix[ymax[i], xmax[i], 0] = 255
        check_matrix[ymin[i], xmax[i], 0] = 255
       

        
        matrix_new = cv2.warpAffine(check_matrix, rotation_matrix, (width_new, height_new))
       
       
        y_where, x_where= np.where(matrix_new[:, :, 0] != 0)
      

        
        xmax[i] = np.amax(x_where)
        xmin[i] = np.amin(x_where)
        ymax[i] = np.amax(y_where)
        ymin[i] = np.amin(y_where)

    return xmax, xmin, ymax, ymin , name
def hue_change2(image, saturation = (0.5, 1.2), constrast=(0.5, 1.2), brightness = (0.9, 1.09)):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transforms.ColorJitter(brightness = brightness)(image)
    image = transforms.ColorJitter(contrast = constrast)(image)
    image = transforms.ColorJitter(saturation = saturation)(image)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image


def create_label_file(img_height, img_width, xmax, xmin, ymax, ymin, name, result_path):

    if xmax == []:
        read_or_write_txt(txt_path = result_path, mode='a' , appended_label='')
    else:
        for i in range(len(xmax)):
            x = (round(float(xmax[i]+xmin[i])/2/img_width, 6))
            y = (round(float(ymax[i]+ymin[i])/2/img_height, 6))
            w = (round(float(xmax[i]-xmin[i])/img_width, 6))
            h = (round(float(ymax[i]-ymin[i])/img_height, 6))
            append_string = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(name[i], x, y, w, h)
            read_or_write_txt(txt_path = result_path, mode='a' , appended_label=append_string)

def add_background_randomly(image1, background, bg_label_str, img_filename):
    img_name_ = img_filename.split("_")[1]
    image = image1.copy()
    
    img_height, img_width = image.shape[:2]
    bg_height, bg_width = background.shape[:2]
    original_height = img_height
    original_width = img_width
    #貼風扇才調整?
    if original_height*original_width >= 180*180 :
        
        # resize image smaller to background
        # the image accounts for at least two-thirds and not more than four-fifths
        min_size = min(bg_height, bg_width) // 5
        max_size = min(bg_height, bg_width) // 4
        new_size = random.randint(min_size, max_size)
        resize_multiple = round(new_size / max(img_height, img_width), 4)
        # image = image.resize((int(img_width * resize_multiple), int(img_height * resize_multiple)), Image.ANTIALIAS)
        image = cv2.resize(image, (int(img_width * resize_multiple), int(img_height * resize_multiple)))
        
        img_height, img_width = image.shape[:2]

    background_0 = background.copy()
    # paste the image to the background
    xmax, xmin, ymax, ymin, _ = all_label_location(bg_label_str, background_0)
    #判斷貼不貼
    x_diff = np.array(xmax)-np.array(xmin)
    y_diff = np.array(ymax)-np.array(ymin)
    background_area = np.dot(x_diff, y_diff)
    img_area = img_height*img_width
    if background_area/(bg_height*bg_width) >= 0.4 or (background_area+img_area)/(bg_height*bg_width) >= 0.6:
        return background_0, bg_label_str
        


    print(background_area/(bg_height*bg_width))
    height_pos = random.randint(5, (bg_height-img_height-5))
    width_pos = random.randint(5, (bg_width-img_width-5))
    
   
    exit = 1
    while(exit == 1):
        checkList = np.zeros(len(xmax))
        
        for i in range(len(xmax)):
            if xmax[i] > width_pos and width_pos+img_width > xmin[i] and ymax[i] > height_pos and height_pos+img_height > ymin[i]:
        
                checkList[i] = 1
        
        if np.sum(checkList) == 0:
             break


        image = image1.copy()

        img_height, img_width = image.shape[:2]
        if original_height*original_width >= 180*180 :
            new_size = random.randint(min_size, max_size)
            resize_multiple = round(new_size / max(img_height, img_width), 4)
            # image = image.resize((int(img_width * resize_multiple), int(img_height * resize_multiple)), Image.ANTIALIAS)
            image = cv2.resize(image, (int(img_width * resize_multiple), int(img_height * resize_multiple)))
            img_height, img_width = image.shape[:2]

        img_area = img_height*img_width
        if background_area/(bg_height*bg_width) >= 0.6 or (background_area+img_area) >= bg_height*bg_width:
            return background_0, bg_label_str


        height_pos = random.randint(5, (bg_height-img_height-5))
        width_pos = random.randint(5, (bg_width-img_width-5))




    background_0[height_pos:(height_pos+img_height), width_pos:(width_pos+img_width)] = image

    x = round(float((2*width_pos+img_width))/2/bg_width, 6)
    y = round(float((2*height_pos+img_height))/2/bg_height, 6)
    w = round(float(img_width)/bg_width, 6)
    h = round(float(img_height)/bg_height, 6)
    
    append_string = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(img_name_, x, y, w, h)
    
    bg_label_str_new = bg_label_str + append_string
    # with open(bg_label_path, 'r') as file:
    #      label_data = file.read()

    # with open(, 'a') as file2:
    #      #file.write("0" + " "+ format(x, '.6f') + " " + format(y, '.6f') + " " + format(w, '.6f') + " " + format(h, '.6f') + "\n")
    #      file2.write(label_data)
    #      file2.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x, y, w, h))
    return background_0, bg_label_str_new    





if __name__ == "__main__":
 
    print("go")
    file_initial = "examples/0731/sort_img/val/breaker"
    file_result = "examples/0830/images/val/"
    img_label = "examples/0731/sort_label/val/breaker"
    result_label = "examples/0830/labels/val/"

    cut_path = "examples/cut/turbo/val/total/"

    img_names = []
    label_names = []
    cut_names = []


    [img_names.append(img_name) for img_name in Path(file_initial).glob('*.png')]
    [label_names.append(label_name) for label_name in Path(img_label).glob('*.txt')]
    [cut_names.append(cut_name) for cut_name in Path(cut_path).glob('*.png')]


    count = 1
    for z in range(1, 61):#11 #5
        for j in range(len(img_names)):
        #疊加
            # for i in range(3):
            #     image_initial = cv2.imread(str(img_names[j]))
            #     image = image_initial.copy()
                
            #     txt_data = read_or_write_txt(str(label_names[j]), mode = 'r')
            #     print(count)
                
            #     # image_height, image_width = image.shape[:2]
            #     # txt_data = read_or_write_txt(img_label, mode = 'r')
            #     # xmax0, xmin0, ymax0, ymin0, name0 = all_label_location(txt_data, image)
            #     for times in range(1+i):
            #         cut_name = str(cut_names[random.randint(0, len(cut_names)-1)])
            #         img = cv2.imread(cut_name)
            #         image, txt_data = add_background_randomly(img, image, txt_data, cut_name)


                image_initial = cv2.imread(str(img_names[j]))
                image = image_initial.copy()
                
                txt_data = read_or_write_txt(str(label_names[j]), mode = 'r')
                print(count)
                ftype = np.random.rand()
                dxx = np.random.randint(-40, 40)
                dyy = np.random.randint(-40, 40)
                ang = np.random.uniform(-10, 10)
                fxx = np.random.uniform(0.7, 1.5)
                fyy = np.random.uniform(0.7, 1.5)

                image = hue_change2(image)
                if np.random.rand() < 0.7:
                    image = scale_img(image, fx = fxx, fy = fyy)#對標註並無影響
                image_height, image_width = image.shape[:2]
                
                xmax0, xmin0, ymax0, ymin0, name0 = all_label_location(txt_data, image)
                
                if np.random.rand() < 0.7:
                    xmax0, xmin0, ymax0, ymin0, name0 = change_point_flip(img_width=image_width, img_height=image_height, xmax = xmax0, xmin = xmin0, ymax = ymax0, ymin = ymin0, name = name0, ftype=ftype)
                    image = flip(image, flip_type = ftype)
                if np.random.rand() < 0.6:
                    xmax0, xmin0, ymax0, ymin0, name0 = change_point_translation(img_width=image_width, img_height=image_height, xmax = xmax0, xmin = xmin0, ymax = ymax0, ymin = ymin0, name = name0, dx = dxx, dy = dyy)
                    image = translation_img(image, dx = dxx, dy = dyy)

                if np.random.rand() < 0.6:
                    xmax0, xmin0, ymax0, ymin0, name0 = change_point_rotate(img_height = image_height, img_width= image_width, xmax=xmax0, xmin=xmin0, ymax=ymax0, ymin=ymin0, name=name0, angle_0 = ang)
                    image = rotate_image(image, angle = ang)

                image_height_r, image_width_r = image.shape[:2]

                label_final = result_label+img_label.split("/")[-1]+str(count).zfill(2) + "_mix.txt"
                img_final = file_result+img_label.split("/")[-1]+str(count).zfill(2) + "_mix.png"

                create_label_file(img_height=image_height_r, img_width=image_width_r, xmax = xmax0, xmin = xmin0, ymax = ymax0, ymin = ymin0, name = name0, result_path=label_final)
                cv2.imwrite(img_final, image)
                
                count += 1