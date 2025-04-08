import cv2 as cv
import numpy as np

# 이미지 불러오기
image = cv.imread('../img/rose.png')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

height, width = image.shape[:2]

center = (width/ 2, height / 2) # 회전 축 좌표정의
angle = 45  # 45도 회전
scale = 1.5  # 1.5배 확대

# 회전 변환 행렬 생성
rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
rotated_image = cv.warpAffine(image, rotation_matrix, (int(width*1.5), int(height*1.5)), flags=cv.INTER_LINEAR)

# 결과 출력
cv.imshow('Original Image', image)
cv.imshow('Result Image', rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
