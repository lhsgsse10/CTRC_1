import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

# Updated paths
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def extract_sift_keypoints(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    
    return keypoints, descriptors

def insert_watermark(original_image_path, watermark_image_path, output_image_path):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')  # Convert to grayscale
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    # Extract SIFT keypoints
    keypoints, _ = extract_sift_keypoints(original_image)
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermark_image_data = np.array(watermark_image, dtype=np.float32)

    # Apply DWT (Discrete Wavelet Transform)
    coeffs2 = pywt.dwt2(original_image_data, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Resize watermark to fit the LL band using LANCZOS interpolation
    watermark_resized = np.array(Image.fromarray(watermark_image_data).resize(LL.shape[::-1], Image.LANCZOS))
    
    # Insert watermark at keypoints
    alpha = 0.05  # Adjusted scaling factor
    LL_w = LL.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x < LL.shape[1] and y < LL.shape[0]:  # Ensure keypoints are within bounds
            LL_w[y, x] += alpha * watermark_resized[y, x]
    
    coeffs2_w = LL_w, (LH, HL, HH)
    watermarked_image_data = pywt.idwt2(coeffs2_w, 'haar')

    # Save the result
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)  # Clip values
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 삽입 시간: {elapsed_time:.10f} seconds")
    
    return LL, keypoints

def extract_watermark(watermarked_image_path, original_LL, keypoints, watermark_size, output_watermark_path):
    start_time = time.time()
    
    # Load image
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)

    # Apply DWT (Discrete Wavelet Transform)
    coeffs2_w = pywt.dwt2(watermarked_image_data, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2_w

    # Extract watermark
    alpha = 0.05  # Adjusted scaling factor
    watermark_extracted = np.zeros_like(original_LL)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x < LL_w.shape[1] and y < LL_w.shape[0]:  # Ensure keypoints are within bounds
            watermark_extracted[y, x] = (LL_w[y, x] - original_LL[y, x]) / alpha
    
    watermark_extracted_resized = np.array(Image.fromarray(watermark_extracted).resize(watermark_size[::-1], Image.LANCZOS))

    # Save the result
    watermark_extracted_resized = np.clip(watermark_extracted_resized, 0, 255)  # Clip values
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 추출 시간: {elapsed_time:.10f} seconds")

# Insert watermark
original_LL, keypoints = insert_watermark(original_image_path, watermark_image_path, watermarked_image_path)

# Extract watermark
extract_watermark(watermarked_image_path, original_LL, keypoints, Image.open(watermark_image_path).size, extracted_watermark_path)

# Load images for display
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

# Convert original image to OpenCV format to draw keypoints
original_image_cv = np.array(original_image.convert('RGB'))
keypoints_img = cv2.drawKeypoints(original_image_cv, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# Display the images side by side
plt.figure(figsize=(20, 8))  # Adjust figure size if needed

# Original Image
plt.subplot(1, 5, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Original Image with Keypoints
plt.subplot(1, 5, 2)
plt.imshow(keypoints_img)
plt.title('Original Image with Keypoints')
plt.axis('off')


# Watermarked Image
plt.subplot(1, 5, 3)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

# Watermark Image
plt.subplot(1, 5, 4)
watermark_image = Image.open(watermark_image_path).convert('L')
plt.imshow(watermark_image, cmap='gray')
plt.title('Watermark Image')
plt.axis('off')

# Extracted Watermark
plt.subplot(1, 5, 5)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()
