

## L05_1. 소벨(Sobel) 에지 검출

### 단계
1. 이미지를 `cv.imread()`로 불러옵니다.  
2. `cv.cvtColor()`로 그레이스케일로 변환합니다.  
3. 소벨 연산자를 X방향과 Y방향으로 각각 적용하여 에지 정보를 구합니다.  
4. 두 방향의 결과를 이용해 에지 강도(`cv.magnitude()`)를 계산합니다.  
5. 최종 에지 강도를 8비트(unsigned int)로 변환(`cv.convertScaleAbs()`)합니다.  
6. Matplotlib을 이용해 원본 이미지와 에지 강도 이미지를 시각화합니다.

### 결과
![Image](https://github.com/user-attachments/assets/2b979aed-6709-4bc2-a777-1370f596fc1d)

### 코드
```Python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv.imread('../img/edgeDetectionImage.jpg')

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
```

---

## L05_2. 허프(Hough) 변환을 이용한 직선 검출

### 단계
1. 이미지를 `cv.imread()`로 불러옵니다.  
2. 그레이스케일 변환(`cv.cvtColor()`) 후, 캐니(Canny) 에지 검출(`cv.Canny()`)을 수행합니다.  
3. `cv.HoughLinesP()` 함수를 이용해 에지 이미지에서 직선을 검출합니다.  
4. 검출된 직선을 원본 이미지 위에 빨간 선(`cv.line()`)으로 표시합니다.  
5. Matplotlib으로 원본 이미지와 직선이 그려진 이미지를 비교 시각화합니다.

### 결과
![Image](https://github.com/user-attachments/assets/649afe76-8d00-4d4f-950f-2f610b847a37)

### 코드
```Python
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
``` 

---

## L05_3. GrabCut을 이용한 전경(Foreground) 분리

### 단계
1. 이미지를 `cv.imread()`로 불러옵니다.  
2. 전경과 배경을 구분할 대략적인 사각형 영역(`rect`)을 지정합니다.  
3. 초기 마스크(`mask`), 배경/전경 모델(`bgdModel`, `fgdModel`)을 준비합니다.  
4. `cv.grabCut()` 함수를 5회 반복(`iterCount=5`) 적용하여 배경과 전경을 점진적으로 구분합니다.  
5. 최종적으로 전경(GC_FGD, GC_PR_FGD)에 해당하는 픽셀만 남도록 마스크(`mask2`)를 재구성합니다.  
6. 전경 분리 결과 이미지를 원본과 함께 시각화합니다.

### 결과
![Image](https://github.com/user-attachments/assets/61c5950d-1e32-4364-859c-6b65d6ccdbc7)

### 코드
```Python
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
``` 

---
