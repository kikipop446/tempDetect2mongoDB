# --*--coding: utf-8 --*...
import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms as transforms
 
import cv2
import numpy as np
import math
import shutil
import matplotlib.pyplot as plt

def show_compare_img(original_img,transform_img):
    _,axes = plt.subplots(1,2)
    #显示图像
    axes[0].imshow(original_img)
    axes[1].imshow(transform_img)
    #设置子标题
    axes[0].set_title("original image")
    axes[1].set_title("warpAffine transform image")
    plt.show()


def translation_img(image):
    # 定义一个图像平移矩阵
    # x向左平移(负数向左,正数向右)200个像素
    # y向下平移(负数向上,正数向下)500个像素
    M = np.array([[1, 0, -128], [0, 1, 96]], dtype=float)
    # 读取需要平移的图像
    img = cv2.imread(image)
    # 将图片由BGR转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 定义平移后图像的大小,保持和原图大小一致
    dsize = img.shape[:2][::-1]
    # 便于大家观察这里采用绿色来填充边界
    translation_img = cv2.warpAffine(img, M, dsize, borderValue=(0, 0, 0))
    # 显示图像
    #show_compare_img(img, translation_img)
    translation_img = cv2.cvtColor(translation_img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("examples/data_aug/aug01.png", translation_img)

def flip(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #水平翻转
    horizontal_flip_img = cv2.flip(img,1)
    #垂直翻转
    vertical_flip_img = cv2.flip(img,0)
    #镜像翻转
    mirror_flip_img = cv2.flip(img,-1)

def scale_img(image):
    fx = 0.8
    fy = 0.3
    
    M = np.array([[fx,0,0],[0,fy,0]],dtype=float)
    #读取图像
    img = cv2.imread(image)
    height,width = img.shape[:2]
    #将图片由BGR转为RGB
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #定义缩放后图片的大小
    scale_img = cv2.warpAffine(img,M,(int(width*fx),int(height*fy)))
    return scale_img

# cv2.imshow("zxc", scale_img("examples/0503T/FAN01_thermal.png"))
# cv2.waitKey (0)
def rotate_img_original(theta):
   #将角度转换为弧度制
   radian_theta = theta/180 * np.pi
   #定义围绕原点旋转的变换矩阵
   M = np.array([[np.cos(radian_theta),np.sin(radian_theta),0],
                 [-np.sin(radian_theta),np.cos(radian_theta),0]])
   # 读取图像
   img = cv2.imread("examples/0503T/FAN04_thermal.png")
   #定义旋转后图片的宽和高
   height,width = img.shape[:2]
   # 将图片由BGR转为RGB
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #围绕原点逆时针旋转\theta度
   rotate_img = cv2.warpAffine(img,M,(width,height))
   horizontal_flip_img = cv2.flip(rotate_img,1)
   #显示图像
   show_compare_img(img,horizontal_flip_img)
#rotate_img_original(15)

def rotate_img_point(point_x,point_y,theta,img,is_completed=False):
    #将角度转换为弧度制
    radian_theta = theta / 180 * np.pi
    #定义围绕任意点旋转的变换矩阵
    M = np.array([[np.cos(radian_theta), np.sin(radian_theta),
                   (1-np.cos(radian_theta))*point_x-point_y*np.sin(radian_theta)],
                  [-np.sin(radian_theta), np.cos(radian_theta),
                   (1-np.cos(radian_theta))*point_y+point_x*np.sin(radian_theta)]])
    # 将图片由BGR转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 定义旋转后图片的宽和高
    height, width = img.shape[:2]
    #判断旋转之后的图片是否需要保持完整
    if is_completed:
        #增大旋转之后图片的宽和高,防止被裁剪掉
        new_height = height * np.cos(radian_theta) + width * np.sin(radian_theta)
        new_width = height * np.sin(radian_theta) + width * np.cos(radian_theta)
        #增大变换矩阵的平移参数
        M[0, 2] += (new_width - width) * 0.5
        M[1, 2] += (new_height - height) * 0.5
        height = int(np.round(new_height))
        width = int(np.round(new_width))
    # 围绕原点逆时针旋转\theta度
    rotate_img = cv2.warpAffine(img, M, (width, height))
    # 显示图像
    show_compare_img(img, rotate_img)
'''
img = cv2.imread("examples/0503T/FAN04_thermal.png")
height,width = img.shape[:2]
#定义围绕图片的中心旋转
point_x,point_y = int(width/2),int(height/2)
rotate_img_point(point_x,point_y,45,img,True)'''
print( np.random.randint(-1, 1))




