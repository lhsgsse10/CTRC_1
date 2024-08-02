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

def embed_watermark_dct(original_image_path, watermark_image_path, output_image_path, alpha=0.1):
    start_time = time.time()
    
    # Load images
    original_image = Image.open(original_image_path).convert('L')  # Convert to grayscale
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    watermark_image_data = np.array(watermark_image, dtype=np.float32)

    # Apply DCT
    dct_image = apply_dct(original_image_data)
    
    # Resize watermark to fit the DCT coefficients
    watermark_resized = np.array(Image.fromarray(watermark_image_data).resize(dct_image.shape[::-1], Image.LANCZOS))
    
    # Embed watermark
    dct_watermarked = dct_image + alpha * watermark_resized
    
    # Apply inverse DCT
    watermarked_image_data = apply_idct(dct_watermarked)

    # Save the result
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)  # Clip values
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 삽입 시간: {elapsed_time:.10f} seconds")

def extract_watermark_dct(watermarked_image_path, original_image_path, watermark_size, output_watermark_path, alpha=0.1):
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
    watermark_extracted = (dct_watermarked - dct_original) / alpha
    
    # Resize and clip the extracted watermark
    watermark_extracted_resized = np.array(Image.fromarray(watermark_extracted).resize(watermark_size, Image.LANCZOS))
    watermark_extracted_resized = np.clip(watermark_extracted_resized, 0, 255)

    # Save the result
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"워터마크 추출 시간: {elapsed_time:.10f} seconds")

# Embed watermark
embed_watermark_dct(original_image_path, watermark_image_path, watermarked_image_path, alpha=0.1)

# Extract watermark
extract_watermark_dct(watermarked_image_path, original_image_path, Image.open(watermark_image_path).size, extracted_watermark_path, alpha=0.1)

# Load images for display
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

# Display the images side by side
plt.figure(figsize=(20, 8))  # Adjust figure size if needed

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Watermarked Image
plt.subplot(1, 4, 2)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

# Watermark Image
plt.subplot(1, 4, 3)
watermark_image = Image.open(watermark_image_path).convert('L')
plt.imshow(watermark_image, cmap='gray')
plt.title('Watermark Image')
plt.axis('off')

# Extracted Watermark
plt.subplot(1, 4, 4)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()
