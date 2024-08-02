import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 경로 설정
original_image_path = 'original_image.png'
watermark_image_path = 'logo.png'
watermarked_image_path = 'watermarked_image.png'
extracted_watermark_path = 'extracted_watermark.png'

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def flann_feature_matching(desc1, desc2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def embed_watermark_sift(original_image_path, watermark_image_path, output_image_path, alpha=0.1):
    original_image = cv2.imread(original_image_path)
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark to fit the original image
    watermark_resized = cv2.resize(watermark_image, (original_image.shape[1], original_image.shape[0]))

    # Extract SIFT features
    kp1, desc1 = extract_sift_features(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY))
    kp2, desc2 = extract_sift_features(watermark_resized)

    # Match features
    matches = flann_feature_matching(desc1, desc2)

    # Embed watermark based on matched keypoints
    watermarked_image = original_image.copy()
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        if 0 <= x1 < original_image.shape[1] and 0 <= y1 < original_image.shape[0] and 0 <= x2 < watermark_resized.shape[1] and 0 <= y2 < watermark_resized.shape[0]:
            for k in range(3):
                watermarked_image[y1, x1, k] = (1 - alpha) * original_image[y1, x1, k] + alpha * watermark_resized[y2, x2]

    cv2.imwrite(output_image_path, watermarked_image)

def extract_watermark_sift(watermarked_image_path, original_image_path, watermark_image_path, output_watermark_path, alpha=0.1):
    original_image = cv2.imread(original_image_path)
    watermarked_image = cv2.imread(watermarked_image_path)
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark to fit the original image
    watermark_resized = cv2.resize(watermark_image, (original_image.shape[1], original_image.shape[0]))

    # Extract SIFT features
    kp1, desc1 = extract_sift_features(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY))
    kp2, desc2 = extract_sift_features(watermark_resized)

    # Match features
    matches = flann_feature_matching(desc1, desc2)

    # Extract watermark based on matched keypoints
    extracted_watermark = np.zeros_like(watermark_resized, dtype=np.float32)
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        if 0 <= x1 < watermarked_image.shape[1] and 0 <= y1 < watermarked_image.shape[0] and 0 <= x2 < watermark_resized.shape[1] and 0 <= y2 < watermark_resized.shape[0]:
            extracted_watermark[y2, x2] = (watermarked_image[y1, x1, 0] - original_image[y1, x1, 0]) / alpha

    extracted_watermark = np.clip(extracted_watermark, 0, 255)
    extracted_watermark_image = Image.fromarray(np.uint8(extracted_watermark))
    extracted_watermark_image.save(output_watermark_path)

# Embed watermark
embed_watermark_sift(original_image_path, watermark_image_path, watermarked_image_path, alpha=0.1)

# Extract watermark
extract_watermark_sift(watermarked_image_path, original_image_path, watermark_image_path, extracted_watermark_path, alpha=0.1)

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
