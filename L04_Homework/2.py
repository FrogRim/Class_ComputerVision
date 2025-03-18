import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv.imread('../img/JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 변환한 사진을 오츄 알고리즘으로 이진화
t, binary_img = cv.threshold(image[:,:,3], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 커널 생성 수정
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

# 모폴로지 연산 적용
dilation = cv.morphologyEx(binary_img, cv.MORPH_DILATE, kernel) # 팽창 연산
erosion = cv.morphologyEx(binary_img, cv.MORPH_ERODE, kernel) # 침식 연산
opening = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel) # 열림 연산
closing = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel) # 닫힘 연산

# 모든 이미지를 3채널로 변환하여 연결
binary_3channel = cv.cvtColor(binary_img, cv.COLOR_GRAY2BGR)
erosion_3channel = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
dilation_3channel = cv.cvtColor(dilation, cv.COLOR_GRAY2BGR)
opening_3channel = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
closing_3channel = cv.cvtColor(closing, cv.COLOR_GRAY2BGR)

# 이미지 가로로 연결
combined_image = np.hstack((binary_3channel, erosion_3channel, dilation_3channel, 
                          opening_3channel, closing_3channel))

# 결과 출력
cv.imshow('Result', combined_image)
cv.waitKey(0)  # 아무 키나 누를 때까지 대기
cv.destroyAllWindows()
