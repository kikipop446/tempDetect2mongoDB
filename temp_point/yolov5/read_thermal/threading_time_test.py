from flir_image_extractor_time_test import FlirImageExtractor
import time
import queue 
import threading
import numpy as np
from pathlib import Path

class worker(threading.Thread):
    def __init__(self, queue, lock, fir, num):
        threading.Thread.__init__(self)
        self.queue = queue
        self.lock = lock
        self.fir = fir
        self.num = num
        self.yolo_detect_path = ""

    def run(self):
        while self.queue.qsize() > 0:
            IRimg_path, IR_num = self.queue.get()

            self.fir.process_image(IRimg_path) #時間主要都花在這裡 == 4.5秒?

            self.yolo_detect_path = self.fir.save_images(path_base)#原圖轉成可視影像和熱影像(無其他字擋住
            #LOCK??
            TempNp[IR_num] = self.fir.get_thermal_np() #這個順序變很奇怪????
            print(str(IRimg_path), self.num)
            #print("worker{} finish??".format(i))
    def get_path(self):
        return self.yolo_detect_path
    

if __name__ == "__main__":
    #取得圖片路徑存list
    path_base = "temp_point\\yolov5\\read_thermal\\examples\\test_image"
    img_names = []
    [img_names.append(img_name) for img_name in Path(path_base).glob('*.jpg')]
    TempNp = np.zeros(len(img_names)).tolist()
    img_names = sorted(img_names)
    print(img_names)
    img_gueue = queue.Queue()
    #將 (待處理圖片路徑和其編號) 加入佇列
    for i in range(len(img_names)):  
        img_gueue.put([str(img_names[i]), i])#將待處理路徑和其編號(length = 2的list)加入佇列

    counter = 0
    yolo_detect_path = ""
    lock = threading.Lock()
    fir = []
    workers = []
    t1 = time.time()
    print("start generating turbo img...")
    #開10個執行緒
    for i in range(5):
        fir.append(FlirImageExtractor())
        workers.append(worker(queue = img_gueue, lock = lock, fir = fir[i], num = i))
        workers[i].start()
        print("worker[{}] start".format(i))

    for i in range(5):
        workers[i].join()
    # fir = FlirImageExtractor()
    # for i in img_names:
    #     fir.process_image(str(i))
    #     yolo_detect_path = fir.save_images(path_base)
    #     TempNp[counter] = fir.get_thermal_np()
    #     counter += 1
    t2 = time.time()
    print("threading Done time:", (t2-t1))
    yolo_detect_path = workers[0].get_path()