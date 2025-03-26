

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

# 소벨 연산자(상하,좌우 화소에 가중치 부여, 에지강도와 방향때문에 자료형은 64비트 실수형인 cv.CV_64F)
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

## L05_3. GrabCut을 이용한 전경(Foreground) 분리(대화식)

### 단계
1. 이미지를 `cv.imread()`로 불러옵니다.  
2. 마우스로 ROI(Region of Interest) 사각형 영역을 드래그하여 설정합니다.
3. 초기 마스크(`mask`), 배경/전경 모델(`bgdModel`, `fgdModel`)을 준비합니다.  
4. `cv.grabCut()` 함수를 'GC_INIT_WITH_RECT' 모드로 반복 적용하여 배경과 전경을 점진적으로 분리합니다.  
5. 최종적으로 전경(GC_FGD, GC_PR_FGD)에 해당하는 픽셀만 남도록 마스크(`mask2`)를 재구성합니다.  
6. 원본 이미지와 마스크, 그리고 최종 전경 분리 결과 이미지를 Matplotlib 서브플롯으로 함께 시각화합니다.

### 결과
![Image](https://github.com/user-attachments/assets/3d009bb2-2597-4a96-a391-250b32e76df2)

### 코드
```Python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


drawing = False          # 마우스 드래그 상태 여부
ix, iy = -1, -1          # 드래그 시작점
rect = (0, 0, 0, 0)      # GrabCut에 쓸 사각형 (x, y, w, h)


image = cv.imread('../img/coffee cup.JPG')  
if image is None:
    raise FileNotFoundError("이미지를 로드할 수 없습니다.")
(h, w) = image.shape[:2]

# 화면에 표시할 임시 복사본 (사각형 드래그 결과 표시용)
preview = image.copy()

def on_mouse(event, x, y, flags, param):
    global drawing, ix, iy, rect, preview
    
   
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
  
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
           
            preview = image.copy()
            cv.rectangle(preview, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('Select ROI', preview)
    
    
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
       
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        
        preview = image.copy()
        cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow('Select ROI', preview)


# 윈도우 생성 및 마우스 콜백 설정
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', on_mouse)
cv.imshow('Select ROI', preview)


# 키 입력에 따라 GrabCut 실행
while True:
    key = cv.waitKey(1) & 0xFF

    if key == 13:  # Enter: GrabCut 실행
        if rect[2] > 0 and rect[3] > 0:
            # GrabCut 초기화
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # GrabCut 알고리즘 (사각형 모드)
            cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
            
            # 마스크 값 변환(GC_BGD, GC_PR_BGD=0 / GC_FGD, GC_PR_FGD=1) -> 전경/배경을 1과 0으로 구분
            mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
            
            # 배경 제거된 이미지(전경만 남김)
            result = image * mask2[:, :, np.newaxis]
            
           
            plt.figure(figsize=(12, 4))

            # (1) 원본 이미지
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            plt.axis('off')

            # (2) 마스크(흑백)
            plt.subplot(1, 3, 2)
            plt.title("Mask Image")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

            # (3) 결과(배경제거)
            plt.subplot(1, 3, 3)
            plt.title("Result Image")
            plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("사각형 영역이 올바르지 않습니다. 다시 드래그하세요.")

    elif key == 27:  # ESC: 종료
        break

cv.destroyAllWindows()

``` 

---
