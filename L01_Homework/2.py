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
    cv.imshow('원본 및 에지 검출', combined_frame)
    
    # q 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv.destroyAllWindows()
