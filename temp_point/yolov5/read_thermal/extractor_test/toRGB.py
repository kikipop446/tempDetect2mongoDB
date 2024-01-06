import flir_image_extractor
import cv2
from pathlib import Path
import shutil
import os
import numpy as np
from PIL import Image, ExifTags
def rotate_image(image, label_box_list=[], angle=-90, color=(0, 0, 0), img_scale=1.0):
    
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
def exif_transpose(img):
        if not img:
            return img

        exif_orientation_tag = 274

        # Check for EXIF data (only present on some files)
        if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
            exif_data = img._getexif()
            orientation = exif_data[exif_orientation_tag]
            
            # Handle EXIF Orientation
            if orientation == 1:
                # Normal image - nothing to do!
                pass
            elif orientation == 2:
                # Mirrored left to right
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotated 180 degrees
                img = img.rotate(180)
            elif orientation == 4:
                # Mirrored top to bottom
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                # Mirrored along top-left diagonal
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                # Rotated 90 degrees
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                # Mirrored along top-right diagonal
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                # Rotated 270 degrees
                img = img.rotate(90, expand=True)

        return img


file_source = 'examples/images/breaker/breaker05.jpg'
img_names = [file_source]
#[img_names.append(img_name) for img_name in Path(file_source).glob('*.jpg')]
fir = flir_image_extractor.FlirImageExtractor()
# for img_name in img_names:

#     fir.process_image(str(img_name))
#     fir.save_images()#原圖轉成可視影像和熱影像(無其他字擋住

#     print(img_name)







# file_destination ='examples/0503I'

# for file in Path(file_source).glob('*_image.jpg'):
#     shutil.move(file,file_destination)


fir = flir_image_extractor.FlirImageExtractor()
fir.process_image(file_source)
fir.save_images()

# img = Image.open(file_source)
# img = exif_transpose(img)
# img.save(file_source)