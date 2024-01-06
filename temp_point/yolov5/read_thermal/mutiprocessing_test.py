from flir_image_extractor_time_test import FlirImageExtractor
import numpy as np
from pathlib import Path
from time import sleep, time
from multiprocessing import Pool

def img_processing(ir_path, i, save_path):
    print("{} start process".format(1+i))
    fir = FlirImageExtractor()
    fir.process_image(ir_path)
    yolo_path = fir.save_images(save_path)
    temp_np = fir.get_thermal_np() 
    return yolo_path, temp_np



if __name__=="__main__":
    #取得圖片路徑存list
    path_base = "temp_point\\yolov5\\read_thermal\\examples\\test_image"
    img_names = []
    [img_names.append(img_name) for idx, img_name in enumerate(Path(path_base).glob('*.jpg'))]
    TempNp = np.zeros(len(img_names)).tolist()
    img_names = sorted(img_names)
    print(img_names)

    result_list = []
    time1 = time()
    with Pool(processes=8) as pool:#process=開幾個進程

        for idx, img_name in enumerate(img_names):

            result = pool.apply_async(img_processing, args = (img_name, idx, path_base,))
            result_list.append(result)
        
        for i, res in enumerate(result_list):
            TempNp[i] = res.get()[1]
            yolo_detect_path = res.get()[0]
    time2 = time()
    print("path: ", yolo_detect_path)
    print(TempNp)
    print("time", time2-time1)





