import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ===== [1] 초기 세팅 및 함수 정의 =====
drawing = False          # 마우스 드래그 상태 여부
ix, iy = -1, -1          # 드래그 시작점
rect = (0, 0, 0, 0)      # GrabCut에 쓸 사각형 (x, y, w, h)

# 원본 이미지 로드
image = cv.imread('../img/coffee cup.JPG')  # 예제 이미지
if image is None:
    raise FileNotFoundError("이미지를 로드할 수 없습니다.")
(h, w) = image.shape[:2]

# 화면에 표시할 임시 복사본 (사각형 드래그 결과 표시용)
preview = image.copy()

def on_mouse(event, x, y, flags, param):
    global drawing, ix, iy, rect, preview
    
    # 왼쪽 버튼 누른 순간
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    # 드래그 중
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 미리보기 이미지를 매번 초기화 후, 현재 드래그 사각형 표시
            preview = image.copy()
            cv.rectangle(preview, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('Select ROI', preview)
    
    # 왼쪽 버튼에서 손을 뗀 순간(드래그 종료)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 드래그로 만들어진 실제 사각형 좌표/크기
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        # 최종 사각형을 preview 상에 표시
        preview = image.copy()
        cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow('Select ROI', preview)


# ===== [2] 윈도우 생성 & 마우스 콜백 설정 =====
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', on_mouse)
cv.imshow('Select ROI', preview)

print("원하는 영역을 마우스로 드래그한 뒤, Enter 키를 누르면 GrabCut을 수행합니다.")
print("드래그를 다시 하면 사각형이 바뀝니다. ESC 키로 종료합니다.")

# ===== [3] 키 입력에 따라 GrabCut 실행 → Matplotlib 시각화 =====
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
            
            # 마스크 값 변환(GC_BGD, GC_PR_BGD=0 / GC_FGD, GC_PR_FGD=1)
            mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
            
            # 배경 제거된 이미지
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
