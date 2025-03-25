import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 원본 이미지 로드
image = cv.imread('../img/coffee cup.JPG')  # 예제 이미지 사용
if image is None:
    raise FileNotFoundError("이미지를 로드할 수 없습니다.")

# 초기 사각형 영역 설정 (이미지 전체)
height, width = image.shape[:2]
rect = (50, 50, width - 100, height - 100)  

# GrabCut 알고리즘에 필요한 초기화
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 알고리즘 실행
cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 마스크 값 변환 (cv.GC_BGD, cv.GC_PR_BGD -> 0: 배경 / cv.GC_FGD, cv.GC_PR_FGD -> 1: 전경)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 배경 제거된 이미지 생성
result = image * mask2[:, :, np.newaxis]

# 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Mask Image")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Result Image")
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
