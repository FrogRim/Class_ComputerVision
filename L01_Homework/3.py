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
