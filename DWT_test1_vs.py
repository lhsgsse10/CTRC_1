import numpy as np
import pywt
from PIL import Image

def insert_watermark(original_image, watermark_image):
    original_image_data = np.array(original_image)
    watermark_image_data = np.array(watermark_image)

    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    watermark_resized = np.resize(watermark_image_data, LL.shape)
    LL_w = LL + 0.01 * watermark_resized

    coeffs2_w = LL_w, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    return Image.fromarray(np.uint8(watermarked_image_data)), LL

def extract_watermark(watermarked_image, original_LL, watermark_size):
    watermarked_image_data = np.array(watermarked_image)

    coeffs2_w = pywt.dwt2(watermarked_image_data, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2_w

    watermark_extracted = (LL_w - original_LL) / 0.01
    watermark_extracted_resized = np.resize(watermark_extracted, watermark_size)

    return Image.fromarray(np.uint8(watermark_extracted_resized))
