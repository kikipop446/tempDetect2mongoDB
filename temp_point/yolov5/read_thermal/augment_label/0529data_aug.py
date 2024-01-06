import random
import cv2
import math
import numpy as np
from pathlib import Path
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps


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

def add_background_randomly(image1, background, bg_label_str):

    """
    box_list = [(cls_type_0, rect_0), (cls_type_1, rect_1), ... , (cls_type_n, rect_n)]
    rect = [x0, y0, x1, y1, x2, y2, x3, y3]
    left_top = (x0, y0), right_top = (x1, y1), right_bottom = (x2, y2), left_bottom = (x3, y3)
    """
    image = image1.copy()
    image = random_rotate(image1, 15, 0.6)
    img_height, img_width = image.shape[:2]
    bg_height, bg_width = background.shape[:2]
    # resize image smaller to background
    # the image accounts for at least two-thirds and not more than four-fifths
    min_size = min(bg_height, bg_width) // 5
    max_size = min(bg_height, bg_width) // 3
    
    new_size = random.randint(min_size, max_size)
    resize_multiple = round(new_size / max(img_height, img_width), 4)
    # image = image.resize((int(img_width * resize_multiple), int(img_height * resize_multiple)), Image.ANTIALIAS)
    image = cv2.resize(image, (int(img_width * resize_multiple), int(img_height * resize_multiple)))
    img_height, img_width = image.shape[:2]

    background_0 = background.copy()
    # paste the image to the background
    # height_pos = random.randint((bg_height-img_height)//3, (bg_height-img_height)//3*2)
    # width_pos = random.randint((bg_width-img_width)//3, (bg_width-img_width)//3*2)
    xmax, xmin, ymax, ymin = all_label_location(bg_label_str, background_0)
    
    height_pos = random.randint(0, (bg_height-img_height))
    width_pos = random.randint(0, (bg_width-img_width))
    
   
    exit = 1
    while(exit == 1):
        checkList = np.zeros(len(xmax))
        
        for i in range(len(xmax)):
            if xmax[i] > width_pos and width_pos+img_width > xmin[i] and ymax[i] > height_pos and height_pos+img_height > ymin[i]:
        
                checkList[i] = 1
        
        if np.sum(checkList) == 0:
             break

        image = image1.copy()
        image = random_rotate(image1, 15, 0.6)
        img_height, img_width = image.shape[:2]
        new_size = random.randint(min_size, max_size)
        resize_multiple = round(new_size / max(img_height, img_width), 4)
        # image = image.resize((int(img_width * resize_multiple), int(img_height * resize_multiple)), Image.ANTIALIAS)
        image = cv2.resize(image, (int(img_width * resize_multiple), int(img_height * resize_multiple)))
        img_height, img_width = image.shape[:2]

        height_pos = random.randint(0, (bg_height-img_height))
        width_pos = random.randint(0, (bg_width-img_width))




    background_0[height_pos:(height_pos+img_height), width_pos:(width_pos+img_width)] = image

    x = round(float((2*width_pos+img_width))/2/bg_width, 6)
    y = round(float((2*height_pos+img_height))/2/bg_height, 6)
    w = round(float(img_width)/bg_width, 6)
    h = round(float(img_height)/bg_height, 6)
    
    append_string = '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x, y, w, h)
    bg_label_str_new = bg_label_str + append_string
    # with open(bg_label_path, 'r') as file:
    #      label_data = file.read()

    # with open(, 'a') as file2:
    #      #file.write("0" + " "+ format(x, '.6f') + " " + format(y, '.6f') + " " + format(w, '.6f') + " " + format(h, '.6f') + "\n")
    #      file2.write(label_data)
    #      file2.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x, y, w, h))
    return background_0, bg_label_str_new    

def cut_bbox(image, txt_path):

    
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

    img2 = image[ymin:ymax, xmin:xmax]
    return img2 



crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
	# 旋转角度的周期是360°
    angle %= 360
	
	# 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
	
	# 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

	# 如果需要裁剪去除黑边
    if crop:
	    # 对于裁剪角度的等效周期是180°
        angle_crop = angle % 180
		
		# 并且关于90°对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
			
		# 转化角度为弧度
        theta = angle_crop * np.pi / 180.0
		
		# 计算高宽比
        hw_ratio = float(h) / float(w)
		
		# 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
		
		# 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
		
		# 计算分母项
        denominator = r * tan_theta + 1
		
		# 计算最终的边长系数
        crop_mult = numerator / denominator
		
		# 得到裁剪区域
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated
def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)
def read_or_write_txt(txt_path='', mode = '', appended_label = ''):
     
    with open(txt_path, mode) as file:
         if mode == 'r':
              data = file.read()
              return data
         elif mode == 'a':
              file.write(appended_label)
      
def hue_change(image, saturation = (0.1, 2), constrast=(1, 2)):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if np.random.rand() < 0.3: image = transforms.ColorJitter(contrast = constrast)(image)
    if np.random.rand() < 0.3: image = transforms.ColorJitter(saturation = saturation)(image)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image

bg_label_dir = "examples/val/unaug_label"

background_dir = "examples/val/unaug_image"

image_dir = "examples/fan_cut_val/"

new_txt_dir = "examples/val/la/"

result_image_dir = "examples/val/img/"
bg_names = []
img_names = []
bg_label_txt = []

[img_names.append(img_name) for img_name in Path(image_dir).glob('*.png')]
[bg_label_txt.append(label_name) for label_name in Path(bg_label_dir).glob('FAN*.txt')]
[bg_names.append(bg_name) for bg_name in Path(background_dir).glob("*.png")]

#bg_names = bg_names[3:8]
#bg_label_txt = bg_label_txt[3:8]
print((bg_names))
counter = 1
for c in range(16):
    for j in range(len(bg_names)):
        for i in range(3):
            bg_img = cv2.imread(str(bg_names[j]))
            label_str = read_or_write_txt(txt_path = str(bg_label_txt[j]), mode='r')

            for times in range(i+1):
                
                img = cv2.imread(str(img_names[random.randint(0, len(img_names)-1)]))
                result_bg, result_string = add_background_randomly(img, bg_img, label_str)
                bg_img = result_bg
                label_str = result_string

                store_img_path = result_image_dir + "FAN" + str(counter).zfill(2) + "_aug.png"
                store_label_path = new_txt_dir + "FAN" + str(counter).zfill(2) + "_aug.txt"
            result_bg2 = hue_change(result_bg)
            cv2.imwrite(store_img_path, result_bg2)
            read_or_write_txt(txt_path = store_label_path, mode='a', appended_label = result_string)
            print(counter)
            counter += 1
        

            
     
     

# label_txt = read_or_write_txt(txt_path="examples/label_aug/FAN20_thermal.txt", mode='r')
# bg = cv2.imread("examples/FAN22.png")
# img = cv2.imread("examples/FAN_cut/FAN14_cut.png")

# result_bg, result_string = add_background_randomly(img, bg, label_txt)

# cv2.imwrite("examples/FAN22.png",result_bg)
# read_or_write_txt(txt_path = new_txt_path+"/FAN21_thermal.txt", mode = 'a', appended_label=result_string)


