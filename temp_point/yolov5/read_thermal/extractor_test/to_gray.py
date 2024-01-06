import numpy as np
import cv2
from pathlib import Path

def homo_filter(image_in, high, low, c, sigma):
    img = np.array(image_in, dtype=np.float64)
    height, width = img.shape
    center_x = width // 2
    center_y = height // 2
    log_img = np.log(img + 1)
    log_fft = np.fft.fft2(log_img)
    log_fft = np.fft.fftshift(log_fft)
    
    h = np.zeros_like(img)
    for y in range(height):
        for x in range(width):
            dist = (x - center_x)**2 + (y - center_y)**2
            h[y, x] = (high - low) * (1 - np.exp(-c * (dist / (2 * sigma * sigma)))) + low
    
    log_fft = h * log_fft
    log_fft = np.fft.ifftshift(log_fft)
    log_fft = np.fft.ifft2(log_fft)
    out = np.exp(log_fft) - 1

    out_normalized = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255
    image_out = np.array(out_normalized, dtype=np.uint8)

    return image_out


result_path = 'examples/0731/sort_img_homo/val/breaker/'
img_names = []
[img_names.append(img_name) for img_name in Path(result_path).glob('*.png')]



for img_name in img_names:
    print(img_name)

    srcImg = cv2.imread(str(img_name))
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(srcImg)

    cv2.imwrite(str(img_name), clahe_img)



for img_name in img_names:
    High_value = 2
    Low_value = 0.2
    C_value = 0.1
    Sigma_value = 150

    input_image = cv2.imread(str(img_name), cv2.IMREAD_GRAYSCALE)

    Height, Width = input_image.shape[:2]
    Sigma_value = max(Width, Height)
    # Example usage
    # Assuming you have an input grayscale image 'input_image.png' and you want to apply the HomoFilter

    output_image = homo_filter(input_image, High_value, Low_value, C_value, Sigma_value)

    cv2.imwrite(str(img_name), output_image)