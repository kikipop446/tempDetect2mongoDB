import flir_image_extractor
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import os
from matplotlib import cm
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

def hue_change(image, brightness = 1.0, constrast = 1.0, saturation = 1.0, hue = (-0.1, 0.1)):
    #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #if np.random.rand() < 0.8: image = transforms.ColorJitter(brightness = brightness)(image)
    image = transforms.ColorJitter(contrast = constrast)(image)
    image = transforms.ColorJitter(saturation = saturation)(image)
    image = transforms.ColorJitter(hue = hue)(image)
    #image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image



img_path = "examples/images/breaker/breaker10.jpg"
result_path = "examples/images/breaker/breaker000.jpg" 
fir = flir_image_extractor.FlirImageExtractor()
fir.process_image(img_path)






thermal_np1= fir.get_thermal_np()
thermal_np = thermal_np1
# x, y = np.shape(thermal_np)
# part = (np.amax(thermal_np)-np.amin(thermal_np))/3
# min_ = np.amin(thermal_np)
# for i in range(x):
#     for j in range(y):
#         if thermal_np[i][j] <= min_+part:
#             thermal_np[i][j] = thermal_np[i][j]*0.8

#         elif thermal_np [i][j]<= min_+2*part:
#             thermal_np [i][j]= thermal_np[i][j]*1.2

#         else:
#             thermal_np [i][j] = thermal_np[i][j]*1.8

thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))



a = np.uint8(cm.inferno(thermal_normalized) * 255)
img_thermal = Image.fromarray(a)

thermal_suffix = "_bright.png"
img_thermal = hue_change(img_thermal, saturation = (0.8, 1.5), constrast=(0.8, 1.5))
fn_prefix, _ = os.path.splitext(result_path)
thermal_filename = fn_prefix + thermal_suffix
print(thermal_filename)
img_thermal.save(thermal_filename)