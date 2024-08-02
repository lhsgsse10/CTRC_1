import numpy as np
import pywt
from PIL import Image

def dwt2(image):
    """2차원 DWT 변환을 수행하여 LL, LH, HL, HH 대역으로 분해"""
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def extract_watermark_LL(cA, binary_length, alpha=0.01):
    """LL 대역에서 워터마크 비트를 추출"""
    flat_cA = cA.flatten()
    extracted_bits = []

    for i in range(binary_length):
        if i < len(flat_cA):
            # 원본 비트와의 차이를 계산하여 추출
            value = flat_cA[i] % alpha
            bit = int(value / alpha)
            extracted_bits.append(str(bit))
            # 디버깅 출력
            print(f'Index {i}: flat_cA[i] = {flat_cA[i]}, value = {value}, bit = {bit}')
        else:
            extracted_bits.append('0')  # 부족한 부분을 채우기 위한 기본값

    return ''.join(extracted_bits)

def binary_to_text(binary_str):
    """이진 비트 시퀀스를 텍스트로 변환"""
    chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
    return ''.join(chars)

def text_to_binary(text):
    """텍스트를 바이너리 비트 시퀀스로 변환"""
    return ''.join(format(ord(char), '08b') for char in text)

# 워터마크가 삽입된 이미지 읽기
watermarked_image = Image.open('watermarked_image.png').convert('L')
watermarked_image = np.array(watermarked_image, dtype=np.float32)

# DWT 변환
cA, cH, cV, cD = dwt2(watermarked_image)

# 바이너리 해시 길이 지정 (원본 해시 길이에 맞게 조정)
hash_value = 'a83336cfd0b1d814a8d7ed99ec2f48186e87d8617205eeced11c89b3cc0f89c3'  # 실제 해시 값을 넣어야 합니다
hash_length = len(text_to_binary(hash_value))  # 기존 해시 값의 길이와 일치해야 함

# LL 대역에서 워터마크 비트 추출
extracted_binary = extract_watermark_LL(cA, hash_length)

# 이진 비트를 텍스트로 변환
extracted_text = binary_to_text(extracted_binary)

print('추출된 해시값 : ', extracted_text)
print('추출된 이진값 : ', extracted_binary)  # 디버깅용 출력
