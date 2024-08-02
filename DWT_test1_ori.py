import pywt
import numpy as np
from PIL import Image
import hashlib
import matplotlib.pyplot as plt

# 원본 이미지 로드
original_image_path = "original_image.png"
original_image = Image.open(original_image_path).convert('L')  # 그레이스케일로 변환
original_image_data = np.array(original_image)

# 원본 이미지의 SHA-256 해시 계산
original_hash_sha256 = hashlib.sha256(original_image_data.tobytes()).hexdigest()
print(f"SHA-256 hash of the original image: {original_hash_sha256}")

# 원본 이미지의 DWT 수행
original_coeffs2 = pywt.dwt2(original_image_data, 'haar')
LL, (LH, HL, HH) = original_coeffs2

# 원본 이미지의 DWT 결과 시각화
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(LL, cmap='gray')
axs[0, 0].set_title('LL')
axs[0, 1].imshow(LH, cmap='gray')
axs[0, 1].set_title('LH')
axs[1, 0].imshow(HL, cmap='gray')
axs[1, 0].set_title('HL')
axs[1, 1].imshow(HH, cmap='gray')
axs[1, 1].set_title('HH')

for ax in axs.flat:
    ax.label_outer()

plt.show()
