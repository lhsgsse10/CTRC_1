import numpy as np
import pywt
from PIL import Image

def insert_watermark(original_image_path, watermark_image_path, output_image_path):
    # 이미지 로드
    original_image = Image.open(original_image_path).convert('L')  # 흑백 이미지로 변환
    watermark_image = Image.open(watermark_image_path).convert('L')

    original_image_data = np.array(original_image)
    watermark_image_data = np.array(watermark_image)

    # DWT (Discrete Wavelet Transform) 적용
    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # 워터마크 크기 조정
    watermark_resized = np.resize(watermark_image_data, LL.shape)
    
    # 워터마크 삽입
    LL_w = LL + 0.01 * watermark_resized
    coeffs2_w = LL_w, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    # 결과 저장
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    return LL

def extract_watermark(watermarked_image_path, original_LL, watermark_size, output_watermark_path):
    # 이미지 로드
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    watermarked_image_data = np.array(watermarked_image)

    # DWT (Discrete Wavelet Transform) 적용
    coeffs2_w = pywt.dwt2(watermarked_image_data, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2_w

    # 워터마크 추출
    watermark_extracted = (LL_w - original_LL) / 0.01
    watermark_extracted_resized = np.resize(watermark_extracted, watermark_size)

    # 결과 저장
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)

# Example usage
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

# 삽입
original_LL = insert_watermark(original_image_path, watermark_image_path, watermarked_image_path)

# 추출
extract_watermark(watermarked_image_path, original_LL, Image.open(watermark_image_path).size, extracted_watermark_path)
