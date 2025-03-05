## L01_1. 이미지 불러오기 및 그레이스케일 변환
### 단계
1. 이미지를 로드
2. 그레이스케일로 변환
3. 원본이미지와 붙여서 출력
4. 결과 출력및 아무거나 클릭시 창이 닫히게하기

### 결과
![Image](https://github.com/user-attachments/assets/50b7ae46-e5b8-48b4-811a-2916574e25cc)
### 코드
```python
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
```

## L01_2. 웹캡 영상에서 에지 검출
### 단계
1. 웹캠 영상을 로드
2. 각 프레임을 그레이스케일로 변환 후 에지 검출 수행
3. 원본영상과 에지영상을 붙여서 출력
4. Q키를 누르면 영상창이 종료

### 결과

### 코드
```python
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
```


## L01_3. 마우스로 영역 선택 및 ROI 추출
### 단계
1. 이미지를 불러오고 화면에 출력
2. 마우스 이벤트를 처리
3. 사용자가 클릭한 시작점에서 드래그하여 영역 선택
4. R키를 누르면 영역 선택 리셋, S키를 누르면 선택한 영역을 이미지로 저장

### 결과
![Image](https://github.com/user-attachments/assets/6ffba2fb-2d10-4f83-887e-cdb2584a9399)

### 코드
```python
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
```
