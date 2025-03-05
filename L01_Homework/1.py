import cv2 as cv
import numpy as np

# 이미지 불러오기
image = cv.imread('soccer.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 그레이스케일 이미지를 3채널로 변환 (np.hstack을 위해)
gray_3channel = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)

# 두 이미지를 가로로 연결
combined_image = np.hstack((image, gray_3channel))


# 결과 출력
cv.imshow('Result', combined_image)
cv.waitKey(0)  # 아무 키나 누를 때까지 대기
cv.destroyAllWindows()
