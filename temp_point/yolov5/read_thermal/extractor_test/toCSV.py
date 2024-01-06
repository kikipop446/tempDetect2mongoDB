import flir_image_extractor
from pathlib import Path
import numpy as np
import cv2
def to_csv(gen):
    fir = flir_image_extractor.FlirImageExtractor()
    for names in gen:
        name1 = str(names)
        fir.process_image(name1)
        csv_name = name1.replace("jpg","csv")
        fir.export_thermal_to_csv(csv_name)

dir_path = Path('examples/')
jpg_namnes = dir_path.glob('FAN*.jpg')


to_csv(jpg_namnes)
