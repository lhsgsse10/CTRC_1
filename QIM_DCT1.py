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

def extract_sift_keypoints(image, num_keypoints=50):
    image_cv = np.array(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:num_keypoints]
    return keypoints, descriptors

def qim_embed(value, watermark_bit, delta):
    return (np.floor(value / delta) + 0.5 + watermark_bit) * delta

def qim_extract(value, delta):
    return np.round((value % delta) / delta)

def embed_watermark_dct_qim(dct_image, watermark_image, keypoints, delta=10):
    watermark_resized = np.array(watermark_image.resize((dct_image.shape[1], dct_image.shape[0]), Image.LANCZOS), dtype=np.float32)
    watermark_bits = np.round(watermark_resized / 255).astype(int)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 8 <= x < dct_image.shape[1] - 8 and 8 <= y < dct_image.shape[0] - 8:
            dct_block = dct_image[y-8:y+8, x-8:x+8]
            watermark_block = watermark_bits[y-8:y+8, x-8:x+8]
            dct_block += delta * watermark_block
    
    return dct_image

def extract_watermark_dct_qim(dct_watermarked, keypoints, delta=10):
    watermark_extracted = np.zeros(dct_watermarked.shape)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 8 <= x < dct_watermarked.shape[1] - 8 and 8 <= y < dct_watermarked.shape[0] - 8:
            dct_block = dct_watermarked[y-8:y+8, x-8:x+8]
            watermark_block = np.round((dct_block % delta) / delta)
            watermark_extracted[y-8:y+8, x-8:x+8] = watermark_block
    
    return watermark_extracted

def embed_watermark(original_image_path, watermark_image_path, output_image_path, delta=10, num_keypoints=50):
    start_time = time.time()
    
    original_image = Image.open(original_image_path).convert('L')
    watermark_image = Image.open(watermark_image_path).convert('L')
    
    original_image_data = np.array(original_image, dtype=np.float32)
    
    keypoints, _ = extract_sift_keypoints(original_image, num_keypoints)
    
    dct_image = apply_dct(original_image_data)
    
    dct_watermarked = embed_watermark_dct_qim(dct_image, watermark_image, keypoints, delta)
    
    watermarked_image_data = apply_idct(dct_watermarked)
    
    watermarked_image_data = np.clip(watermarked_image_data, 0, 255)
    watermarked_image = Image.fromarray(np.uint8(watermarked_image_data))
    watermarked_image.save(output_image_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Watermark embedding time: {elapsed_time:.10f} seconds")
    
    return keypoints

def extract_watermark(watermarked_image_path, keypoints, watermark_size, output_watermark_path, delta=10):
    start_time = time.time()
    
    watermarked_image = Image.open(watermarked_image_path).convert('L')
    watermarked_image_data = np.array(watermarked_image, dtype=np.float32)
    
    dct_watermarked = apply_dct(watermarked_image_data)
    
    watermark_extracted = extract_watermark_dct_qim(dct_watermarked, keypoints, delta)
    
    watermark_extracted_resized = np.array(Image.fromarray(np.uint8(watermark_extracted * 255)).resize(watermark_size, Image.LANCZOS))
    watermark_extracted_resized = np.clip(watermark_extracted_resized, 0, 255)
    
    extracted_watermark = Image.fromarray(np.uint8(watermark_extracted_resized))
    extracted_watermark.save(output_watermark_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Watermark extraction time: {elapsed_time:.10f} seconds")

keypoints = embed_watermark(original_image_path, watermark_image_path, watermarked_image_path, delta=20, num_keypoints=200)

extract_watermark(watermarked_image_path, keypoints, Image.open(watermark_image_path).size, extracted_watermark_path, delta=20)

original_image = Image.open(original_image_path)
watermarked_image = Image.open(watermarked_image_path)
extracted_watermark = Image.open(extracted_watermark_path)

original_image_cv = np.array(original_image.convert('RGB'))
keypoints_img = cv2.drawKeypoints(original_image_cv, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

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
