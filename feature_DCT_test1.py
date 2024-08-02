import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

# Updated paths
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def apply_dct(image):
    return cv2.dct(np.float32(image))

def apply_idct(dct_image):
    return cv2.idct(dct_image)

def extract_sift_keypoints(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    
    # Sort keypoints based on their response (strength), take the top 200 keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:200]
    
    return keypoints, descriptors

def embed_watermark_dct(dct_image, watermark_image, keypoints, alpha=0.5):
    watermark_resized = np.array(watermark_image.resize((dct_image.shape[1], dct_image.shape[0]), Image.LANCZOS), dtype=np.float32)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 8 <= x < dct_image.shape[1] - 8 and 8 <= y < dct_image.shape[0] - 8:  # Ensure keypoints are within bounds
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    dct_image[y + dy, x + dx] += alpha * watermark_resized[y + dy, x + dx]
    
    return dct_image

def extract_watermark_dct(dct_watermarked, dct_original, keypoints, alpha=0.5):
    watermark_extracted = np.zeros_like(dct_original)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 8 <= x < dct_watermarked.shape[1] - 8 and 8 <= y < dct_watermarked.shape[0] - 8:  # Ensure keypoints are within bounds
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    watermark_extracted[y + dy, x + dx] = (dct_watermarked[y + dy, x + dx] - dct_original[y + dy, x + dx]) / alpha
    
    return watermark_extracted

def embed_watermark(original_image_path, watermark_image_path, output_image_path, alpha=0.5):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    
    # Extract SIFT keypoints
    keypoints, _ = extract_sift_keypoints(original_image)
    
    # Apply DCT
    dct_image = apply_dct(original_image_data)
    
    # Embed watermark in DCT domain
    dct_watermarked = embed_watermark_dct(dct_image, watermark_image, keypoints, alpha)
    
    # Apply inverse DCT
    watermarked_image_data = apply_idct(dct_watermarked)
    
    # Save the result
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Watermark embedding time: {elapsed_time:.10f} seconds")
    
    return keypoints

def extract_watermark(watermarked_image_path, original_image_path, keypoints, watermark_size, output_watermark_path, alpha=0.5):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)
    
    # Apply DCT
    dct_original = apply_dct(original_image_data)
    dct_watermarked = apply_dct(watermarked_image_data)
    
    # Extract watermark
    watermark_extracted = extract_watermark_dct(dct_watermarked, dct_original, keypoints, alpha)
    
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
keypoints = embed_watermark(original_image_path, watermark_image_path, watermarked_image_path, alpha=0.5)

# Extract watermark
extract_watermark(watermarked_image_path, original_image_path, keypoints, Image.open(watermark_image_path).size, extracted_watermark_path, alpha=0.5)

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
