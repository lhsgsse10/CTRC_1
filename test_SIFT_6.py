import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

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

def resize_image_to_match(img1, img2):
    """이미지 크기를 일치시키는 함수"""
    if img1.shape != img2.shape:
        new_size = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_AREA)
    return img1, img2

def insert_watermark_sift_region(original_image_path, watermark_image_path, output_image_path, region_size=32, alpha=0.15):
    start_time = time.time()  # 시작 시간 측정
    
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
    
    elapsed_time = time.time() - start_time  # 걸린 시간 계산
    print(f"워터마크 삽입 시간: {elapsed_time:.10f} seconds")
    
    # 삽입된 워터마크 이미지와 원본 이미지를 반환
    return x_start, y_start, x_end, y_end, kp, watermarked_image_data, original_image_data

def extract_watermark_sift_region(original_image_path, watermarked_image_path, x_start, y_start, x_end, y_end, kp, output_watermark_path, alpha=0.1):
    start_time = time.time()  # 시작 시간 측정
    
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
    
    elapsed_time = time.time() - start_time  # 걸린 시간 계산
    print(f"워터마크 추출 시간: {elapsed_time:.10f} seconds")
    
    return extracted_watermark

def calculate_psnr(original_image_data, watermarked_image_data):
    # 이미지 크기 일치
    original_image_data, watermarked_image_data = resize_image_to_match(original_image_data, watermarked_image_data)
    
    mse = np.mean((original_image_data - watermarked_image_data) ** 2)
    if mse == 0:  # MSE가 0이면 두 이미지가 완전히 동일함을 의미
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 워터마크 삽입
x_start, y_start, x_end, y_end, kp, watermarked_image_data, original_image_data = insert_watermark_sift_region(original_image_path, watermark_image_path, watermarked_image_path)

# 워터마크 추출
extracted_watermark = extract_watermark_sift_region(original_image_path, watermarked_image_path, x_start, y_start, x_end, y_end, kp, extracted_watermark_path)

# PSNR 계산 (회색조로 변환된 원본 이미지와 워터마크 삽입된 이미지 비교)
psnr_value = calculate_psnr(original_image_data, watermarked_image_data)
print(f"PSNR 계산: {psnr_value:.2f} dB")

# 원본 이미지에 특징점을 표시
original_image = Image.open(original_image_path)
original_image_cv = np.array(original_image.convert('RGB'))
keypoints_img = cv2.drawKeypoints(original_image_cv, [kp], None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# 결과 이미지 로드 및 표시
watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
extracted_watermark = Image.open(extracted_watermark_path)
logo_image = Image.open(watermark_image_path).convert('L')

plt.figure(figsize=(20, 8))

# 1행
plt.subplot(2, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(keypoints_img)
plt.title('SIFT Keypoints')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(logo_image, cmap='gray')
plt.title('Watermark Image')
plt.axis('off')

# 2행
plt.subplot(2, 4, 5)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()
