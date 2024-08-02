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

def extract_sift_keypoints(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    
    # Sort keypoints based on their response (strength), take the top 50 keypoints
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:50]
    
    return keypoints, descriptors

def svd_watermark_embed(U, S, V, watermark_image, alpha):
    # Resize the watermark image to the shape of the singular values vector
    wm = np.array(watermark_image.resize((S.shape[0],), Image.LANCZOS), dtype=np.float32)
    S_w = S + alpha * wm
    return U @ np.diag(S_w) @ V

def svd_watermark_extract(S_w, original_S, alpha, watermark_size):
    S_extracted = (S_w - original_S) / alpha
    return S_extracted

def insert_watermark(original_image_path, watermark_image_path, output_image_path, alpha=0.05):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')  # Convert to grayscale
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    # Extract SIFT keypoints
    keypoints, _ = extract_sift_keypoints(original_image)
    
    original_image_data = np.array(original_image, dtype=np.float32)

    # Apply SVD
    U, S, V = np.linalg.svd(original_image_data, full_matrices=False)

    # Embed watermark using SVD
    watermarked_image_data = svd_watermark_embed(U, S, V, watermark_image, alpha)

    # Save the result
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)  # Clip values
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 삽입 시간: {elapsed_time:.10f} seconds")
    
    return S, keypoints

def extract_watermark(watermarked_image_path, original_S, watermark_size, output_watermark_path, alpha=0.05):
    start_time = time.time()
    
    # Load image
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)

    # Apply SVD
    U_w, S_w, V_w = np.linalg.svd(watermarked_image_data, full_matrices=False)

    # Extract watermark using SVD
    extracted_watermark_data = svd_watermark_extract(S_w, original_S, alpha, watermark_size)
    
    # Resize and clip the extracted watermark
    extracted_watermark_resized = np.array(Image.fromarray(extracted_watermark_data.reshape((1, -1))).resize(watermark_size, Image.LANCZOS))
    extracted_watermark_resized = np.clip(extracted_watermark_resized, 0, 255)

    # Save the result
    extracted_watermark = Image.fromarray(np.uint8(extracted_watermark_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 추출 시간: {elapsed_time:.10f} seconds")

# Insert watermark
original_S, keypoints = insert_watermark(original_image_path, watermark_image_path, watermarked_image_path, alpha=0.05)

# Extract watermark
extract_watermark(watermarked_image_path, original_S, Image.open(watermark_image_path).size, extracted_watermark_path, alpha=0.05)

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
