import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pywt
import time

# Updated paths
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def apply_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    return coeffs

def apply_idwt(coeffs):
    return pywt.idwt2(coeffs, 'haar')

def extract_sift_keypoints(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    
    # Sort keypoints based on their response (strength), take the top 300 keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:300]
    
    return keypoints, descriptors

def embed_watermark_dwt(dwt_coeffs, watermark_image, keypoints, alpha=1.0):
    LL, (LH, HL, HH) = dwt_coeffs
    watermark_resized = np.array(watermark_image.resize((LL.shape[1], LL.shape[0]), Image.LANCZOS), dtype=np.float32)
    
    LL_w = LL.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 2 <= x < LL.shape[1] - 2 and 2 <= y < LL.shape[0] - 2:  # Ensure keypoints are within bounds
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    LL_w[y + dy, x + dx] += alpha * watermark_resized[y + dy, x + dx]
    
    return LL_w, (LH, HL, HH)

def extract_watermark_dwt(dwt_watermarked, dwt_original, keypoints, alpha=1.0):
    LL_w, (LH_w, HL_w, HH_w) = dwt_watermarked
    LL, (LH, HL, HH) = dwt_original
    watermark_extracted = np.zeros_like(LL)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 2 <= x < LL_w.shape[1] - 2 and 2 <= y < LL_w.shape[0] - 2:  # Ensure keypoints are within bounds
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    watermark_extracted[y + dy, x + dx] = (LL_w[y + dy, x + dx] - LL[y + dy, x + dx]) / alpha
    
    return watermark_extracted

def embed_watermark(original_image_path, watermark_image_path, output_image_path, alpha=1.0):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    
    # Extract SIFT keypoints
    keypoints, _ = extract_sift_keypoints(original_image)
    
    # Apply DWT
    dwt_image = apply_dwt(original_image_data)
    
    # Embed watermark in DWT domain
    dwt_watermarked = embed_watermark_dwt(dwt_image, watermark_image, keypoints, alpha)
    
    # Apply inverse DWT
    watermarked_image_data = apply_idwt(dwt_watermarked)
    
    # Save the result
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Watermark embedding time: {elapsed_time:.10f} seconds")
    
    return keypoints

def extract_watermark(watermarked_image_path, original_image_path, keypoints, watermark_size, output_watermark_path, alpha=1.0):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)
    
    # Apply DWT
    dwt_original = apply_dwt(original_image_data)
    dwt_watermarked = apply_dwt(watermarked_image_data)
    
    # Extract watermark
    watermark_extracted = extract_watermark_dwt(dwt_watermarked, dwt_original, keypoints, alpha)
    
    # Resize and clip the extracted watermark
    watermark_extracted_resized = np.array(Image.fromarray(np.uint8(watermark_extracted)).resize(watermark_size, Image.LANCZOS))
    watermark_extracted_resized = np.clip(watermark_extracted_resized, 0, 255)
    
    # Save the result
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Watermark extraction time: {elapsed_time:.10f} seconds")

# Embed watermark
keypoints = embed_watermark(original_image_path, watermark_image_path, watermarked_image_path, alpha=1.0)

# Extract watermark
extract_watermark(watermarked_image_path, original_image_path, keypoints, Image.open(watermark_image_path).size, extracted_watermark_path, alpha=1.0)

# Load images for display
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

# Convert original image to OpenCV format to draw keypoints
original_image_cv = np.array(original_image.convert('RGB'))
keypoints_img = cv2.drawKeypoints(original_image_cv, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# Display the images side by side
plt.figure(figsize=(20, 8))

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
