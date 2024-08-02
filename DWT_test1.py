import numpy as np
import pywt
from PIL import Image
import hashlib


def dwt2(image):
    """2차원 DWT 변환을 수행하여 LL, LH, HL, HH 대역으로 분해"""
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD


def idwt2(cA, cH, cV, cD):
    """LL, LH, HL, HH 대역을 사용하여 IDWT 변환을 수행하여 원본 이미지를 복원"""
    coeffs = (cA, (cH, cV, cD))
    image = pywt.idwt2(coeffs, 'haar')
    return image


def text_to_binary(text):
    """텍스트를 바이너리 비트 시퀀스로 변환"""
    return ''.join(format(ord(char), '08b') for char in text)


def insert_watermark_LL(cA, binary_hash, alpha=0.01):
    """LL 대역에 해시 값을 삽입"""
    flat_cA = cA.flatten()
    hash_len = len(binary_hash)

    for i in range(hash_len):
        if i < len(flat_cA):
            flat_cA[i] += alpha * int(binary_hash[i])

    cA = flat_cA.reshape(cA.shape)
    return cA


# 원본 이미지 로드 및 회색조 변환
image = Image.open('original_image.png').convert('L')
image = np.array(image, dtype=np.float32)

# 해시 값 생성 (예: 이미지 파일의 SHA-256 해시)
hash_value = hashlib.sha256(image.tobytes()).hexdigest()
print('해쉬값 : ',hash_value)

# 해시 값을 바이너리 비트 시퀀스로 변환
binary_hash = text_to_binary(hash_value)
print('이진값 : ',binary_hash)

# DWT 변환
cA, cH, cV, cD = dwt2(image)

# LL 대역에 해시 값 삽입
cA = insert_watermark_LL(cA, binary_hash)

# IDWT 변환을 통해 워터마크가 삽입된 이미지 생성
watermarked_image = idwt2(cA, cH, cV, cD)
watermarked_image = np.clip(watermarked_image, 0, 255)
watermarked_image = Image.fromarray(watermarked_image.astype(np.uint8))

# 워터마크가 삽입된 이미지 저장
watermarked_image.save('watermarked_image.png')