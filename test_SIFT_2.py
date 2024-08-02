import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 경로 설정
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def extract_sift_keypoints(image):
    image_cv = np.array(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    return keypoints, descriptors

def insert_watermark_sift_region(original_image_path, watermark_image_path, output_image_path, region_size=32, alpha=0.1):
    original_image = Image.open(original_image_path).convert('L')
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    keypoints, _ = extract_sift_keypoints(original_image)
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermark_image_data = np.array(watermark_image, dtype=np.float32)
    
    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # 워터마크를 삽입할 영역 선택
        x_start = max(x - region_size // 2, 0)
        y_start = max(y - region_size // 2, 0)
        x_end = min(x_start + region_size, LL.shape[1])
        y_end = min(y_start + region_size, LL.shape[0])
        
        # 영역 크기가 0이 아닌지 확인
        if x_end > x_start and y_end > y_start:
            break
    else:
        raise ValueError("No valid region found for watermark insertion.")

    watermark_resized = np.array(Image.fromarray(watermark_image_data).resize((x_end - x_start, y_end - y_start), Image.LANCZOS))
    
    # LL 영역에 워터마크 삽입
    LL[y_start:y_end, x_start:x_end] += alpha * watermark_resized
    
    coeffs2_w = LL, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    return x_start, y_start, x_end, y_end, kp

def extract_watermark_sift_region(original_image_path, watermarked_image_path, x_start, y_start, x_end, y_end, kp, output_watermark_path, alpha=0.1):
    original_image = Image.open(original_image_path).convert('L')
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)
    
    coeffs2_orig = pywt.dwt2(original_image_data, 'haar')
    LL_orig, (LH_orig, HL_orig, HH_orig) = coeffs2_orig
    
    coeffs2_wat = pywt.dwt2(watermarked_image_data, 'haar')
    LL_wat, (LH_wat, HL_wat, HH_wat) = coeffs2_wat
    
    # LL 영역에서 워터마크 추출
    watermark_extracted = (LL_wat[y_start:y_end, x_start:x_end] - LL_orig[y_start:y_end, x_start:x_end]) / alpha
    
    watermark_extracted = np.clip(watermark_extracted, 0, 255)
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted))
    extracted_watermark.save(output_watermark_path)
    
    return extracted_watermark

# 워터마크 삽입
x_start, y_start, x_end, y_end, kp = insert_watermark_sift_region(original_image_path, watermark_image_path, watermarked_image_path)

# 워터마크 추출
extracted_watermark = extract_watermark_sift_region(original_image_path, watermarked_image_path, x_start, y_start, x_end, y_end, kp, extracted_watermark_path)

# 결과 이미지 로드 및 표시
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

plt.figure(figsize=(20, 8))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()
