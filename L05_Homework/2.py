import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv.imread('../img/DaboTower.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


canny = cv.Canny(gray_image,100,200)

# 허프 변환을 사용하여 선 검출
lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

# 선을 그린 이미지 복사본 생성
image_with_lines = image.copy()

# 검출된 선을 이미지에 그리기
if lines is not None:
    for i in lines:
        for x1, y1, x2, y2 in i:
            cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis('off')

# 직선이 그려진 이미지
plt.subplot(1, 2, 2)
plt.title('Image with Lines')
plt.imshow(cv.cvtColor(image_with_lines, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
