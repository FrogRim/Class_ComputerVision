## 1. 이진화 및 히스토그램 구하기

이 코드는 다음과 같은 과정으로 이미지를 처리합니다:

1. 이미지를 로드합니다.
2. 컬러 이미지를 그레이스케일로 변환합니다.
3. 오츄(Otsu) 알고리즘을 사용하여 자동으로 최적의 임계값을 찾아 이진화합니다.
4. 원본 그레이스케일 이미지의 히스토그램을 계산하고 시각화합니다.

```python
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

```

### 기술적 설명

- **cv.cvtColor()**: BGR 컬러 이미지를 그레이스케일로 변환합니다.
- **cv.threshold()**: 오츄(OTSU) 알고리즘을 사용하여 이미지를 이진화합니다. 이 알고리즘은 이미지의 히스토그램을 분석하여 최적의 임계값을 자동으로 계산합니다.
- **cv.calcHist()**: 이미지의 히스토그램을 계산합니다. 여기서는 그레이스케일 이미지의, 0 채널(그레이스케일은 채널이 하나만 있음), 마스크 없이, 256개의 빈을 사용하여 0-255 범위의 값을 가진 히스토그램을 계산합니다.
- **plt.plot()**: 계산된 히스토그램을 시각화합니다.
## 결과
![Image](https://github.com/user-attachments/assets/b5cc9c47-69d5-47ad-b8b3-7ae232ddf68d)


## 2. 모폴로지 연산 적용하기

이 코드는 이미지에 다양한 모폴로지 연산을 적용하는 과정을 보여줍니다:

1. 투명도 채널이 있는 PNG 이미지를 로드합니다.
2. 알파 채널(투명도)을 사용하여 이진화합니다.
3. 5x5 크기의 사각형 커널을 생성합니다.
4. 팽창(Dilation), 침식(Erosion), 열림(Opening), 닫힘(Closing) 연산을 적용합니다.
5. 모든 결과 이미지를 가로로 나란히 배치하여 시각화합니다.

```python
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
 
# 이미지 불러오기 
image = cv.imread('../img/JohnHancocksSignature.png', cv.IMREAD_UNCHANGED) 
 
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
```

### 기술적 설명

- **cv.IMREAD_UNCHANGED**: 투명도 채널을 포함하여 이미지를 로드합니다.
- **image[:,:,3]**: PNG 이미지의 알파 채널(투명도 정보)에 접근합니다.
- **cv.getStructuringElement()**: 모폴로지 연산에 사용할 5x5 크기의 사각형 구조 요소(커널)를 생성합니다.
- **cv.morphologyEx()**: 다양한 모폴로지 연산을 수행합니다.
  - **MORPH_DILATE**: 팽창 연산으로, 밝은 영역을 확장시킵니다.
  - **MORPH_ERODE**: 침식 연산으로, 밝은 영역을 축소시킵니다.
  - **MORPH_OPEN**: 열림 연산으로, 침식 후 팽창을 수행해 작은 노이즈를 제거합니다.
  - **MORPH_CLOSE**: 닫힘 연산으로, 팽창 후 침식을 수행해 작은 구멍을 메웁니다.
- **cv.cvtColor(_, cv.COLOR_GRAY2BGR)**: 그레이스케일 이미지를 3채널 BGR 이미지로 변환합니다.
- **np.hstack()**: 여러 이미지를 수평으로 연결하여 하나의 이미지로 만듭니다.

  ## 결과
![Image](https://github.com/user-attachments/assets/a4c642ba-3655-4f0a-a036-427a72a47185)



## 3. 기하 연산 및 선형 보간 적용하기

이 코드는 이미지에 기하학적 변환과 선형 보간을 적용하는 과정을 보여줍니다:

1. 이미지를 로드합니다.
2. 이미지의 중심점을 중심으로 45도 회전하는 변환 행렬을 생성합니다.
3. 선형 보간법(Bilinear Interpolation)을 적용하여 이미지를 회전하고 1.5배 확대합니다.
4. 원본 이미지와 변환된 이미지를 시각적으로 비교합니다.

```python
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
```

### 기술적 설명

- **image.shape[:2]**: 이미지의 높이와 너비를 가져옵니다.
- **cv.getRotationMatrix2D()**: 회전 변환 행렬을 생성합니다. 매개변수는 회전 중심점, 회전 각도(반시계 방향), 크기 조정 비율입니다.
- **cv.warpAffine()**: 
  - 생성한 변환 행렬을 사용하여 이미지에 아핀 변환(회전, 크기 조정)을 적용합니다.
  - 원본 이미지보다 1.5배 큰 출력 이미지 크기를 지정합니다.
  - **cv.INTER_LINEAR** 플래그를 사용하여 양선형 보간법을 적용합니다. 이는 회전이나 크기 조정 시 발생하는 픽셀 간 불일치를 부드럽게 처리하여 더 자연스러운 결과를 제공합니다.

 ## 결과
 ### 원본
![Image](https://github.com/user-attachments/assets/a976f64f-4d17-4a5e-b071-46d7dd876c07)
 ### 확대된 이미지
![Image](https://github.com/user-attachments/assets/bbe7b98f-8e16-428d-96bb-837367d7d26c)
(출력이미지 크기도 1.5배 해놓았기에 원본과 크기차이가 없어보입니다.)
