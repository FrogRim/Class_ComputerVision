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
![Image](https://github.com/user-attachments/assets/abce71a1-b58f-4663-b0aa-5776753597e6)
### 코드
```python
import cv2 as cv
import numpy as np

# 웹캠 열기
cap = cv.VideoCapture(0) # 0은 기본 웹캠

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 캡처
    ret, frame = cap.read()
    
    # 프레임이 제대로 캡처되었는지 확인
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 그레이스케일로 변환
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Canny 에지 검출 적용
    edges = cv.Canny(gray_frame, 100, 200)  # 하한 임계값 100, 상한 임계값 200
    
    # 에지 이미지를 3채널로 변환 (np.hstack을 위해)
    edges_3channel = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    
    # 두 이미지를 가로로 연결
    combined_frame = np.hstack((frame, edges_3channel))
    
    # 결과 출력
    cv.imshow('Combined_frame2edges', combined_frame)
    
    # q 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
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

# 전역 변수
drawing = False  # 마우스가 클릭된 상태 여부
ix, iy = -1, -1  # 시작점 좌표
roi = None  # 선택된 ROI

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, roi, img_copy
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
    
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 드래그 종료, 사각형 그리기
        cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # ROI 추출 (좌표값 정렬을 위해 시작점과 종료점 조정)
        x_start = min(ix, x)
        x_end = max(ix, x)
        y_start = min(iy, y)
        y_end = max(iy, y)
        
        # ROI 추출
        roi = img[y_start:y_end, x_start:x_end]
        
        # ROI 표시
        if roi.size > 0:  # ROI가 유효한지 확인
            cv.imshow('Selected ROI', roi)

# 메인 함수
def main():
    global img, img_copy, roi
    
    # 이미지 불러오기
    img = cv.imread('soccer.jpg')
    
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    img_copy = img.copy()
    
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_rectangle)
    
    while True:
        cv.imshow('image', img_copy)
        key = cv.waitKey(1) & 0xFF
        
        # r 키를 누르면 리셋
        if key == ord('r'):
            img_copy = img.copy()
            cv.destroyWindow('Selected ROI')
        
        # s 키를 누르면 ROI 저장
        elif key == ord('s'):
            if roi is not None and roi.size > 0:
                cv.imwrite('selected_roi.jpg', roi)
                print("ROI가 'selected_roi.jpg'로 저장되었습니다.")
        
        # q 키를 누르면 종료
        elif key == ord('q'):
            break
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
```
