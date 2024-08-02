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

def extract_orb_keypoints(image):
    image_cv = np.array(image)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(image_cv, None)
    return keypoints, descriptors

def insert_watermark_orb(original_image_path, watermark_image_path, output_image_path, num_keypoints=300, alpha=0.3):
    start_time = time.time()
    
    original_image = Image.open(original_image_path).convert('L')
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    keypoints, _ = extract_orb_keypoints(original_image)
    
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:num_keypoints]
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermark_image_data = np.array(watermark_image, dtype=np.float32)

    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    watermark_resized = np.array(Image.fromarray(watermark_image_data).resize(LL.shape[::-1], Image.LANCZOS))
    
    LL_w = LL.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x < LL.shape[1] and y < LL.shape[0]:
            LL_w[y, x] += alpha * watermark_resized[y, x]
    
    coeffs2_w = LL_w, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 삽입 시간: {elapsed_time:.10f} seconds")
    
    return LL, keypoints

def extract_watermark_orb(watermarked_image_path, original_LL, keypoints, watermark_size, output_watermark_path, alpha=0.3):
    start_time = time.time()
    
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)

    coeffs2_w = pywt.dwt2(watermarked_image_data, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2_w

    watermark_extracted = np.zeros_like(original_LL)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x < LL_w.shape[1] and y < LL_w.shape[0]:
            watermark_extracted[y, x] = (LL_w[y, x] - original_LL[y, x]) / alpha
    
    watermark_extracted_resized = np.array(Image.fromarray(watermark_extracted).resize(watermark_size[::-1], Image.LANCZOS))
    
    watermark_extracted_resized = np.clip(watermark_extracted_resized, 0, 255)
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 추출 시간: {elapsed_time:.10f} seconds")

# 워터마크 삽입
original_LL, selected_keypoints = insert_watermark_orb(original_image_path, watermark_image_path, watermarked_image_path, num_keypoints=300, alpha=0.3)

# 워터마크 추출
extract_watermark_orb(watermarked_image_path, original_LL, selected_keypoints, Image.open(watermark_image_path).size, extracted_watermark_path, alpha=0.3)

# 결과 이미지 로드 및 표시
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

# 키포인트가 표시된 원본 이미지 로드
original_image_cv = np.array(original_image.convert('RGB'))
keypoints_img = cv2.drawKeypoints(original_image_cv, selected_keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# 이미지 표시
plt.figure(figsize=(20, 8))

plt.subplot(1, 5, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(keypoints_img)
plt.title('Original Image with Keypoints')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.subplot(1, 5, 4)
watermark_image = Image.open(watermark_image_path).convert('L')
plt.imshow(watermark_image, cmap='gray')
plt.title('Watermark Image')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()
