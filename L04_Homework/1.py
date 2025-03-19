import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv.imread('../img/soccer.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 변환한 사진을 오츄 알고리즘으로 이진화
t, binary_img = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 그레이스케일 이미지의 히스토그램 계산
gray_hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

# 이진화된 이미지의 히스토그램 계산
binary_hist = cv.calcHist([binary_img], [0], None, [256], [0, 256])

# 결과 출력
# 서브플롯 생성
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(gray_hist, color = 'r', linewidth = 1)
plt.title('Histogram_gray')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(binary_hist, color = 'b', linewidth = 1)
plt.title('Histogram_bin')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.legend()
plt.tight_layout()
plt.show()
