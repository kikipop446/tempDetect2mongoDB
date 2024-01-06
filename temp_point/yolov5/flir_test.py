from read_thermal import flir_image_extractor
import time

img_path = "temp_point/yolov5/data/images/breaker01.jpg"

fir = flir_image_extractor.FlirImageExtractor()

for i in range(20):
    start = time.time()
    fir.process_image(img_path)
    end1 = time.time()
    yolo_detect_path = fir.save_images("test000")#原圖轉成可視影像和熱影像(無其他字擋住
    end2 = time.time()    
    print("process time",format(end1-start))
    print("save time", format(end2-end1))