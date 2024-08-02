import pywt
import numpy as np
from PIL import Image
import hashlib
import matplotlib.pyplot as plt

# 워터마크된 이미지 로드
watermarked_image_path = "watermarked_image.png"
watermarked_image = Image.open(watermarked_image_path).convert('L')  # 그레이스케일로 변환
watermarked_image_data = np.array(watermarked_image)

# 워터마크된 이미지의 SHA-256 해시 계산
watermarked_hash_sha256 = hashlib.sha256(watermarked_image_data.tobytes()).hexdigest()
print(f"SHA-256 hash of the watermarked image: {watermarked_hash_sha256}")

# 워터마크된 이미지의 DWT 수행
watermarked_coeffs2 = pywt.dwt2(watermarked_image_data, 'haar')
LL, (LH, HL, HH) = watermarked_coeffs2

# 워터마크된 이미지의 DWT 결과 시각화
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
