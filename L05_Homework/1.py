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

# 소벨 연산자(상하,좌우 화소에 가중치 부여, 에지강도와 방향때문에 자료형은 32비트 실수형인 cv.CV_32F)
gray_x = cv.Sobel(gray_image,cv.CV_64F,1,0,ksize=3)
gray_y = cv.Sobel(gray_image,cv.CV_64F,0,1,ksize=3)

# 절대값을 취해 양수 영상으로 변환
sobel_x = cv.convertScaleAbs(gray_x)
sobel_y = cv.convertScaleAbs(gray_y)

# 에지 강도 계산
magnitude = cv.magnitude(np.float64(sobel_x), np.float64(sobel_y))

# 에지 강도를 uint8로 변환
magnitude_uint8 = cv.convertScaleAbs(magnitude)

# 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# 에지 강도 이미지
plt.subplot(1, 2, 2)
plt.title('Edge Magnitude')
plt.imshow(magnitude_uint8, cmap='gray')
plt.axis('off')

plt.show()