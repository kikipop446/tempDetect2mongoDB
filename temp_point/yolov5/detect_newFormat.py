# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
from read_thermal import flir_image_extractor
import argparse
import os
import platform
import sys
from pathlib import Path

from datetime import datetime
import numpy as np
from PIL import Image
from matplotlib import cm
from pymongo import MongoClient
import time

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        tempList = []
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    counter = 0

    # all_image_dic = {}
    # time_now = datetime.now()
    # all_image_dic["time"] = f'{time_now:%Y/%m/%d %H:%M:%S}'

    for path, im, im0s, vid_cap, s in dataset:

        img_number = "{:03d}".format(counter+1)
        


        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

           
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                str_ = ""

                equip_number = 1
                
                
                for *xyxy, conf, cls in reversed(det):#det代表該圖片有幾個物件框
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                        str_ += ('%g ' * len(line)).rstrip() % line + '\n'

                        equip_dictionary = {}
                        time_now = datetime.now()
                        equip_dictionary['time'] = f'{time_now:%Y/%m/%d %H:%M:%S}'
                        equip_dictionary['image'] = str(p)
                        equip_dictionary["category"] = which_equip(str(int(cls.cpu().item())))

                        xmin = int(xyxy[0].cpu().item()-1)
                        ymin = int(xyxy[1].cpu().item()-1)
                        xmax = int(xyxy[2].cpu().item()-1)
                        ymax = int(xyxy[3].cpu().item()-1)

                        equip_dictionary["coordinate"] = [xmin, ymin, xmax, ymax]
                        thermal_np = tempList[counter]
                        thermalCut = thermal_np[ymin:ymax, xmin:xmax]
                        equip_dictionary['result'] = condition(equip = str(int(cls.cpu().item())), temp_cut = thermalCut)
                        tempMax = round(np.max(thermalCut), 4)
                        tempMin = round(np.min(thermalCut), 4)
                        tempAvg = round(np.mean(thermalCut), 4)
                        equip_dictionary['temp'] = {"max" : tempMax, "min" : tempMin, "avg" : tempAvg}

                        collection.insert_one(equip_dictionary)

                        
                        equip_number += 1
                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                #print(str_)
        #print(all_image_dic)
        counter += 1   
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'data/run_weight/paste/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/turbo_img', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/IR_detect.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default = True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default = True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default = False, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
def main(opt, tempList):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt), tempList = tempList)
    
def which_equip(equip):
    if equip == "0":
        return "斷路器"
    elif equip == "1":
        return "電容"
    elif equip == "2":
        return "電磁接觸器"
    elif equip == "3":
        return "錶頭"
    elif equip == "4":
        return "風扇"
    elif equip == "5":
        return "熔絲"
    elif equip == "6":
        return "指示燈"
    elif equip == "7":
        return "馬達"
    elif equip == "8":
        return "PVC電纜"
    elif equip == "9":
        return "電阻"
    elif equip == "10":
        return "變壓器"
    elif equip == "11":
        return "電驛"
def condition(equip, temp_cut):
    if equip == "0":
        if np.max(temp_cut) >= 105:
            return "危險"
        elif np.max(temp_cut) >= 75:
            return "異常"
        elif np.max(temp_cut) >= 60:
            return "注意"
        else:
            return "正常"
        #return "斷路器"
    elif equip == "1":
        if np.max(temp_cut) >= 145:
            return "危險"
        elif np.max(temp_cut) >= 120:
            return "異常"
        elif np.max(temp_cut) >= 85:
            return "注意"
        else:
            return "正常"
        #return "電子電容"
    elif equip == "2":
        if np.max(temp_cut) >= 110:
            return "危險"
        elif np.max(temp_cut) >= 95:
            return "異常"
        elif np.max(temp_cut) >= 80:
            return "注意"
        else:
            return "正常"
        #return "電磁接觸器(封閉式)"
    elif equip == "3":
        if np.max(temp_cut) >= 90:
            return "危險"
        elif np.max(temp_cut) >= 70:
            return "異常"
        elif np.max(temp_cut) >= 55:
            return "注意"
        else:
            return "正常"
        #return "錶頭"
    elif equip == "4":
        if np.max(temp_cut) >= 90:
            return "危險"
        elif np.max(temp_cut) >= 75:
            return "異常"
        elif np.max(temp_cut) >= 65:
            return "注意"
        else:
            return "正常"
        #return "風扇"
    elif equip == "5":
        if np.max(temp_cut) >= 105:
            return "危險"
        elif np.max(temp_cut) >= 60:
            return "異常"
        elif np.max(temp_cut) >= 45:
            return "注意"
        else:
            return "正常"
        #return "熔絲"
    elif equip == "6":
        if np.max(temp_cut) >= 75:
            return "危險"
        elif np.max(temp_cut) >= 60:
            return "異常"
        elif np.max(temp_cut) >= 45:
            return "注意"
        else:
            return "正常"
        #return "指示燈(比壓器降壓)"
    elif equip == "7":
        if np.max(temp_cut) >= 115:
            return "危險"
        elif np.max(temp_cut) >= 90:
            return "異常"
        elif np.max(temp_cut) >= 75:
            return "注意"
        else:
            return "正常"
        #return "馬達"
    elif equip == "8":
        if np.max(temp_cut) >= 75:
            return "危險"
        elif np.max(temp_cut) >= 55:
            return "異常"
        elif np.max(temp_cut) >= 45:
            return "注意"
        else:
            return "正常"
        #return "PVC電纜"
    elif equip == "9":
        if np.max(temp_cut) >= 240:
            return "危險"
        elif np.max(temp_cut) >= 140:
            return "異常"
        elif np.max(temp_cut) >= 95:
            return "注意"
        else:
            return "正常"
        #return "電阻"
    elif equip == "10":
        if np.max(temp_cut) >= 100:
            return "危險"
        elif np.max(temp_cut) >= 85:
            return "異常"
        elif np.max(temp_cut) >= 65:
            return "注意"
        else:
            return "正常"
        #return "變壓器(線圈)"
    # elif equip == "11":
    #     #return "電驛"
if __name__ == '__main__':
    client = MongoClient("mongodb://140.118.172.141:27017/")
    db = client.TSMC_test
    collection = db.detect_result
    collection2 = db.if_detect
    collection3 = db.Unidentifield

    while(True):
        start_num = collection2.find_one()
        if start_num["number"] == 1:
            ir_img_path = r"C:\Users\ki\Desktop\python\temp_point\yolov5\data\images"
            img_names = []
            [img_names.append(img_name) for img_name in Path(ir_img_path).absolute().glob('*.jpg')]
            TempNp = []
            
            fir = flir_image_extractor.FlirImageExtractor()
            for img_name in img_names:
                print("==")
                fir.process_image(str(img_name))
                fir.save_images()#原圖轉成可視影像和熱影像(無其他字擋住
                TempNp.append(fir.get_thermal_np())
                
            # thermal_np = TempNp[0]
            # thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
            # a = np.uint8(cm.inferno(thermal_normalized) * 255)
            # img_thermal = Image.fromarray(a)    
            # img_thermal.show()

            opt = parse_opt()
            main(opt, tempList=TempNp)


            
            
            

            query = {"_id" : start_num["_id"]}
            update = {"$set": {"number": 0}}
            collection2.update_one(query, update)
        
        time.sleep(1)
        print("ping")


