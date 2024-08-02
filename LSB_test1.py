import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 경로 설정
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def embed_watermark_lsb(original_image_path, watermark_image_path, output_image_path):
    # Load images
    original_image = cv2.imread(original_image_path)
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark to fit the original image
    watermark_resized = cv2.resize(watermark_image, (original_image.shape[1], original_image.shape[0]))

    # Normalize watermark to binary values (0 or 1)
    _, watermark_binary = cv2.threshold(watermark_resized, 128, 1, cv2.THRESH_BINARY)

    # Embed watermark into the LSB of the original image
    watermarked_image = original_image.copy()
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            for k in range(original_image.shape[2]):
                watermarked_image[i, j, k] = (original_image[i, j, k] & 0xFE) | watermark_binary[i, j]

    # Save the result
    cv2.imwrite(output_image_path, watermarked_image)

def extract_watermark_lsb(watermarked_image_path, output_watermark_path):
    # Load watermarked image
    watermarked_image = cv2.imread(watermarked_image_path)

    # Extract watermark from the LSB of the watermarked image
    extracted_watermark = np.zeros((watermarked_image.shape[0], watermarked_image.shape[1]), dtype=np.uint8)
    for i in range(watermarked_image.shape[0]):
        for j in range(watermarked_image.shape[1]):
            extracted_watermark[i, j] = watermarked_image[i, j, 0] & 1

    # Scale extracted watermark to visible range (0 or 255)
    extracted_watermark = extracted_watermark * 255

    # Save the result
    cv2.imwrite(output_watermark_path, extracted_watermark)

# Embed watermark
embed_watermark_lsb(original_image_path, watermark_image_path, watermarked_image_path)

# Extract watermark
extract_watermark_lsb(watermarked_image_path, extracted_watermark_path)

# Load images for display
original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

# Display the images side by side
plt.figure(figsize=(20, 8))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

# Watermarked Image
plt.subplot(1, 4, 2)
plt.imshow(watermarked_image)
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
