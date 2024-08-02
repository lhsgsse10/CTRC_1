import numpy as np
import pywt
from PIL import Image

def insert_watermark(original_image, watermark_image):
    # 이미지 로드 및 변환
    original_image_data = np.array(original_image)
    watermark_image_data = np.array(watermark_image)

    # DWT 적용
    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # 워터마크 삽입 (LL 성분에 추가)
    watermark_resized = np.resize(watermark_image_data, LL.shape)
    LL_w = LL + 0.01 * watermark_resized

    # 역 DWT 적용
    coeffs2_w = LL_w, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    return Image.fromarray(np.uint8(watermarked_image_data)), LL

def extract_watermark(watermarked_image, original_LL, watermark_size):
    # 이미지 로드 및 변환
    watermarked_image_data = np.array(watermarked_image)

    # DWT 적용
    coeffs2_w = pywt.dwt2(watermarked_image_data, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2_w

    # 워터마크 추출
    watermark_extracted = (LL_w - original_LL) / 0.01

    # 워터마크 리사이즈
    watermark_extracted_resized = np.resize(watermark_extracted, watermark_size)

    return Image.fromarray(np.uint8(watermark_extracted_resized))

# 원본 이미지와 워터마크 이미지 로드
original_image_path = "original_image.png"
watermark_image_path = "watermarked_image.png"

original_image = Image.open(original_image_path).convert('L')
watermark_image = Image.open(watermark_image_path).convert('L')

# 워터마크 삽입 및 LL 계수 반환
watermarked_image, original_LL = insert_watermark(original_image, watermark_image)
watermarked_image.save("watermarked_image.png")

# 워터마크 추출
extracted_watermark = extract_watermark(watermarked_image, original_LL, watermark_image.size)
extracted_watermark.save("extracted_watermark.png")
