import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import math
import shutil
import matplotlib.pyplot as plt
 
def add_background_randomly(image, background, image_label_path, bg_label_path, box_list=[]):
    """
    box_list = [(cls_type_0, rect_0), (cls_type_1, rect_1), ... , (cls_type_n, rect_n)]
    rect = [x0, y0, x1, y1, x2, y2, x3, y3]
    left_top = (x0, y0), right_top = (x1, y1), right_bottom = (x2, y2), left_bottom = (x3, y3)
    """
    image = cut_bbox(image, image_label_path)

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
    with open(bg_label_path,'r') as f:
            data = f.read()
    shape_list = data[:-2].split(' ')[1:5]
    imgh, imgw = background.shape[:2]
    x = float(shape_list[0])
    y = float(shape_list[1])
    w = float(shape_list[2])
    h = float(shape_list[3])
    xmin = int((x-w/2)*imgw)
    xmax = int((x+w/2)*imgw)
    ymin = int((y - h/2)*imgh)
    ymax = int((y + h/2)*imgh)
    height_pos = random.randint(0, (bg_height-img_height))
    width_pos = random.randint(0, (bg_width-img_width))
    
    while(xmax > width_pos and width_pos+img_width > xmin and ymax > height_pos and height_pos+img_height > ymin):
        height_pos = random.randint(0, (bg_height-img_height))
        width_pos = random.randint(0, (bg_width-img_width))
        
        
    
    


    background_0[height_pos:(height_pos+img_height), width_pos:(width_pos+img_width)] = image
    img_height, img_width = background.shape[:2]
    '''
    # calculate the boxes after adding background
    new_box_list = []
    for cls_type, rect in box_list:
        for coor_index in range(len(rect)//2):
            # resize
            rect[coor_index*2] = int(rect[coor_index*2] * resize_multiple)      # x
            rect[coor_index*2+1] = int(rect[coor_index*2+1] * resize_multiple)  # y
 
            # paste
            rect[coor_index*2] += width_pos                                     # x
            rect[coor_index*2+1] += height_pos                                  # y
 
            # limite
            rect[coor_index*2] = max(min(rect[coor_index*2], img_width), 0)     # x
            rect[coor_index*2+1] = max(min(rect[coor_index*2+1], img_height), 0)# y
        box = (cls_type, rect)
        new_box_list.append(box)
    image_with_boxes = [background, new_box_list]
    '''
    return background_0
 
 
def rotate_image(image, label_box_list=[], angle=90, color=(0, 0, 0), img_scale=1.0):
    """
    rotate with angle, background filled with color, default black (0, 0, 0)
    label_box = (cls_type, box)
    box = [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), 
    # then grab the sine and cosine (i.e., the rotation components of the matrix)
    # if angle < 0, counterclockwise rotation; if angle > 0, clockwise rotation
    # 1.0 - scale, to adjust the size scale (image scaling parameter), recommended 0.75
    height_ori, width_ori = image.shape[:2]
    x_center_ori, y_center_ori = (width_ori // 2, height_ori // 2)
 
    rotation_matrix = cv2.getRotationMatrix2D((x_center_ori, y_center_ori), angle, img_scale)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
 
    # compute the new bounding dimensions of the image
    width_new = int((height_ori * sin) + (width_ori * cos))
    height_new = int((height_ori * cos) + (width_ori * sin))
 
    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (width_new / 2) - x_center_ori
    rotation_matrix[1, 2] += (height_new / 2) - y_center_ori
 
    # perform the actual rotation and return the image
    # borderValue - color to fill missing background, default black, customizable
    image_new = cv2.warpAffine(image, rotation_matrix, (width_new, height_new), borderValue=(20, 25, 20))
    '''
    # each point coordinates
    angle = angle / 180 * math.pi
    box_rot_list = cal_rotate_box(label_box_list, angle, (x_center_ori, y_center_ori), (width_new//2, height_new//2))
    box_new_list = []
    for cls_type, box_rot in box_rot_list:
        for index in range(len(box_rot)//2):
            box_rot[index*2] = int(box_rot[index*2])
            box_rot[index*2] = max(min(box_rot[index*2], width_new), 0)
            box_rot[index*2+1] = int(box_rot[index*2+1])
            box_rot[index*2+1] = max(min(box_rot[index*2+1], height_new), 0)
        box_new_list.append((cls_type, box_rot))

    image_with_boxes = [image_new, box_new_list]
    '''
    return image_new
def cal_rotate_box(box_list, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    box_list_new = []
    for (cls_type, box) in box_list:
        box_new = []
        for index in range(len(box)//2):
            box_new.extend(cal_rotate_coordinate(box[index*2], box[index*2+1], angle, ori_center, new_center))
        label_box = (cls_type, box_new)
        box_list_new.append(label_box)
    return box_list_new
def cal_rotate_coordinate(x_ori, y_ori, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    x_0 = x_ori - ori_center[0]
    y_0 = ori_center[1] - y_ori
    x_new = x_0 * math.cos(angle) - y_0 * math.sin(angle) + new_center[0]
    y_new = new_center[1] - (y_0 * math.cos(angle) + x_0 * math.sin(angle))
    return (x_new, y_new)
 
 
def hue_change(image, brightness = 1.0, constrast = 1.0, saturation = 1.0, hue = 1.0):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if np.random.rand() < 0.8: image = transforms.ColorJitter(brightness = brightness)(image)
    if np.random.rand() < 0.5: image = transforms.ColorJitter(contrast = constrast)(image)
    if np.random.rand() < 0.5: image = transforms.ColorJitter(saturation = saturation)(image)
    if np.random.rand() < 0.5: image = transforms.ColorJitter(hue = hue)(image)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image

def hue_change2(image, saturation = (0.5, 1.2), constrast=(0.5, 1.2), brightness = (0.89, 1.29)):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transforms.ColorJitter(brightness = brightness)(image)
    image = transforms.ColorJitter(contrast = constrast)(image)
    image = transforms.ColorJitter(saturation = saturation)(image)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image

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
 
def perspective_tranform(image, perspective_rate=0.5, label_box_list=[]):
    # perspective transform
    img_height, img_width = image.shape[:2]
    # points_src = np.float32([[rect[0], rect[1]], [rect[2], rect[3]], [rect[4], rect[5]], [rect[6], rect[7]]])
    points_src = np.float32([[0, 0], [img_width-1, 0], [img_width-1, img_height-1], [0, img_height-1]])
    max_width = int(img_width * (1.0 + perspective_rate))
    max_height = int(img_height * (1.0 + perspective_rate))
    min_width = int(img_width * (1.0 - perspective_rate))
    min_height = int(img_height * (1.0 + perspective_rate))
    delta_width = (max_width - min_width) // 2
    delta_height = (max_height - min_height) // 2
    x0 = random.randint(0, delta_width)
    y0 = random.randint(0, delta_height)
    x1 = random.randint(delta_width + min_width, max_width)
    y1 = random.randint(0, delta_height)
    x2 = random.randint(delta_width + min_width, max_width)
    y2 = random.randint(delta_height + min_height, max_height)
    x3 = random.randint(0, delta_width)
    y3 = random.randint(delta_height + min_height, max_height)
    points_dst = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    # width_new = max(x0, x1, x2, x3) - min(x0, x1, x2, x3)
    # height_new = max(y0, y1, y2, y3) - min(y0, y1, y2, y3)
    M = cv2.getPerspectiveTransform(points_src, points_dst)
    image_res = cv2.warpPerspective(image, M, (max_width, max_height))
    # cut
    image_new = image_res[min(y0, y1):max(y2, y3), min(x0, x3):max(x1, x2)]
    '''
    # labels
    box_new_list = []
    for cls_type, box in label_box_list:
        # after transformation
        for index in range(len(box)//2):
            px = (M[0][0]*box[index*2] + M[0][1]*box[index*2+1] + M[0][2]) / ((M[2][0]*box[index*2] + M[2][1]*box[index*2+1] + M[2][2]))
            py = (M[1][0]*box[index*2] + M[1][1]*box[index*2+1] + M[1][2]) / ((M[2][0]*box[index*2] + M[2][1]*box[index*2+1] + M[2][2]))
            box[index*2] = int(px)
            box[index*2+1] = int(py)
            # cut
            box[index*2] -= min(x0, x3)
            box[index*2+1] -= min(y0, y1)
            box[index*2] = max(min(box[index*2], image_new.shape[1]), 0)
            box[index*2+1] = max(min(box[index*2+1], image_new.shape[0]), 0)
        box_new_list.append((cls_type, box))
 
    image_with_boxes = [image_new, box_new_list]
    '''
    return image_new

def show_compare_img(original_img,transform_img):
    _,axes = plt.subplots(1,2)
    #显示图像
    axes[0].imshow(original_img)
    axes[1].imshow(transform_img)
    #设置子标题
    axes[0].set_title("original image")
    axes[1].set_title("warpAffine transform image")
    plt.show()

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

def flip(image, flip_type):
    #水平翻转
    horizontal_flip_img = cv2.flip(image,1)
    #垂直翻转
    vertical_flip_img = cv2.flip(image,0)
    #镜像翻转
    mirror_flip_img = cv2.flip(image,-1)
    if flip_type == 1:
        return horizontal_flip_img
    elif flip_type == 0:
        return vertical_flip_img
    elif flip_type == -1:
        return mirror_flip_img

def translation_img(image, dx = 1, dy = 1):
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
    dsize = image.shape[:2][::-1]
    translation_img = cv2.warpAffine(image, M, dsize, borderValue=(0, 0, 0))
    return translation_img

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
if __name__ == "__main__":
 
  '''  # test
    img_test_path = os.path.join(test_path, file_name)
    points = np.array([[rect[0],rect[1]], [rect[2],rect[3]], [rect[4],rect[5]], [rect[6],rect[7]]], np.int32)
    image_rect = cv2.polylines(image_res, pts=[points], isClosed=True, color=(0,0,255), thickness=3)
    cv2.imwrite(img_test_path, image_res)
    # print("")'''
  print("go")
  file_initial = "examples//0712/sort_img/test/resistance/"

  file_result = "examples/0712/images/test/"

  #img_label = "examples/label"

  img_names = []
  #label_names = []
  [img_names.append(img_name) for img_name in Path(file_initial).glob('*.png')]
  #[label_names.append(label_name) for label_name in Path(img_label).glob('FAN*.txt')]
  
#   bg_names = img_names[6:]
#   bg_label = label_names[6:]
#   for i in bg_names:
#       print(i)
count = 1
for i in range(1, 61):
  for img in img_names:
    image_initial = cv2.imread(str(img))
    image = image_initial.copy()
    #旋轉-----每張3次
    
    #mix
    # lable_point_list = xymaxmin_bbox(image, str(label_names[img_names.index(img)]))
    # label_weight = lable_point_list[1]-lable_point_list[0]
    # label_height = lable_point_list[3]-lable_point_list[2]
    # ratio = label_height*label_weight/(lable_point_list[4]*lable_point_list[5])

    image = hue_change2(image)

    if np.random.rand() < 0.9:
     image = flip(image, flip_type = np.random.randint(-1, 1))

    if np.random.rand() < 0.8 :
      image = translation_img(image, dx = np.random.randint(-50, 50), dy = np.random.randint(-50, 50))

    if np.random.rand() < 0.7:
      image = rotate_image(image, angle = np.random.uniform(-10, 10))
      
    if np.random.rand() < 0.6:
      image = scale_img(image, fx = np.random.uniform(0.7, 1.2), fy = np.random.uniform(0.7, 1.2))
    
    store_img_path = file_result + "resistance" + str(count).zfill(2) + "_mix.png"
    cv2.imwrite(store_img_path, image)
    print(count)
    count += 1
    '''
    for rand_angle in np.random.uniform(0, 360, 3):
        rot_img = rotate_image(image, angle = rand_angle)
        cv2.imwrite((str(img).replace("FAN", 'FAN_rot')).replace(".png", "_rot%d.png"%count), rot_img)
        count = count+1
    '''
    #縮放
    '''
    for count_scale in range(1,4):
        fx_rand = np.random.uniform(0.5, 3)
        fy_rand = np.random.uniform(0.5, 3)
        scaleImg = scale_img(image, fx = fx_rand, fy = fy_rand)
        cv2.imwrite((str(img).replace("FAN", 'FAN_scale')).replace(".png", "_scale%d.png"%count_scale), scaleImg)
    '''
    #翻轉
    
    '''
    flip_type = -1
    for count_flip in range(1,4):
        
        flip_img = flip(image, flip_type = flip_type)
        cv2.imwrite((str(img).replace("FAN", 'FAN_flip')).replace(".png", "_flip%d.png"%count_flip), flip_img)
        flip_type = flip_type+1
    '''
    #色彩變化
    '''
    for count_hue in range(1, 4):
        hue = np.random.uniform(0.1, 0.9, 3)
        hue_rand = np.random.uniform(0., 0.5)
        hue_img = hue_change(image, brightness = hue[0], constrast = hue[1], saturation = hue[2], hue = hue_rand )
        cv2.imwrite((str(img).replace("FAN", 'FAN_hue')).replace(".png", "_hue%d.png"%count_hue), hue_img)
    '''
    #平移
    '''
    for count_translation in range(1, 4):
        dx = np.random.randint(-160, 160)
        dy = np.random.randint(-120, 120)
        tran_img = translation_img(image, dx = dx, dy = dy)
        cv2.imwrite((str(img).replace("FAN", 'FAN_tran')).replace(".png", "_tran%d.png"%count_translation), tran_img)
    '''
    #透射變換
    '''
    for count_perspec in range(1, 4):
        print(count_perspec)
        perspec_img = perspective_tranform(image)
        print(perspec_img)
    
        cv2.imwrite((str(img).replace("FAN", 'FAN_perspec')).replace(".png", "_perspec%d.png"%count_perspec), perspec_img)
  '''
    #貼背景
    
    # for bg in bg_names:
        
    #     bg_img = cv2.imread(str(bg))
    #     for count_bg in range(1, 3):
            
    #         result_img = add_background_randomly(image = image, background = bg_img, image_label_path = str(label_names[img_names.index(img)]), bg_label_path = str(bg_label[bg_names.index(bg)]))
            
    #         cv2.imwrite((str(img).replace("FAN", 'FAN_bg')).replace(".png", "_bg%d_%d.png"%(bg_names.index(bg), count_bg)), result_img)
        
    '''
  img = cv2.imread("examples/0503T/FAN04_thermal.png")
  bg = cv2.imread("examples/0503T/FAN19_thermal.png")
  img1 = Image.open("examples/0503T/FAN04_thermal.png")
  img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  result = add_background_randomly(img, bg)
  cv2.imshow("zxc", result)
  cv2.waitKey (0)
  #result.show()

  cv2.imwrite("examples/data_aug/aug.png", result)
  cv2.imshow("zxc", result)
  cv2.waitKey (0)
  '''
 