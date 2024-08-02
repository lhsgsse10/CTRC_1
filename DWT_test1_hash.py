import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 원본 이미지 로드 및 DWT 수행
original_image_path = "original_image.png"
original_image = Image.open(original_image_path).convert('L')
original_image_data = np.array(original_image)
original_coeffs2 = pywt.dwt2(original_image_data, 'haar')
LL_o, (LH_o, HL_o, HH_o) = original_coeffs2

# 워터마크된 이미지 로드 및 DWT 수행
watermarked_image_path = "watermarked_image.png"
watermarked_image = Image.open(watermarked_image_path).convert('L')
watermarked_image_data = np.array(watermarked_image)
watermarked_coeffs2 = pywt.dwt2(watermarked_image_data, 'haar')
LL_w, (LH_w, HL_w, HH_w) = watermarked_coeffs2

# 원본 이미지의 DWT 결과 시각화
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs[0, 0].imshow(LL_o, cmap='gray')
axs[0, 0].set_title('Original LL')
axs[0, 1].imshow(LH_o, cmap='gray')
axs[0, 1].set_title('Original LH')
axs[0, 2].imshow(HL_o, cmap='gray')
axs[0, 2].set_title('Original HL')
axs[0, 3].imshow(HH_o, cmap='gray')
axs[0, 3].set_title('Original HH')

# 워터마크된 이미지의 DWT 결과 시각화
axs[1, 0].imshow(LL_w, cmap='gray')
axs[1, 0].set_title('Watermarked LL')
axs[1, 1].imshow(LH_w, cmap='gray')
axs[1, 1].set_title('Watermarked LH')
axs[1, 2].imshow(HL_w, cmap='gray')
axs[1, 2].set_title('Watermarked HL')
axs[1, 3].imshow(HH_w, cmap='gray')
axs[1, 3].set_title('Watermarked HH')

for ax in axs.flat:
    ax.label_outer()

plt.show()
