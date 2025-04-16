import cv2 as cv
import mediapipe as mp

# MediaPipe의 face_mesh 솔루션을 읽어 mp_mesh에 저장
mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # 결과를 그리는데 사용하는 drawing_utils 모듈
mp_styles = mp.solutions.drawing_styles  # 스타일을 지정하는 drawing_styles 모듈

# FaceMesh 초기화
mesh = mp_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# MediaPipe를 이용해 비디오에서 사람 인식
while True:
    ret, frame = cap.read()
    if not ret:
        print("캡처된 화면이 없거나 프레임을 읽을 수 없습니다.")
        break

    # 실제 그물망 검출을 수행하고 결과를 res에 저장
    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.multi_face_landmarks:  # 검출된 얼굴이 있는지 확인
        for face_landmarks in res.multi_face_landmarks:
            h, w, _ = frame.shape  # 이미지 크기 가져오기
            for idx, landmark in enumerate(face_landmarks.landmark):
                # 정규화된 랜드마크 좌표를 픽셀 좌표로 변환
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                # OpenCV의 circle 함수로 랜드마크를 점으로 표시
                cv.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)

    cv.imshow("MediaPipe Face Mesh", cv.flip(frame, 1))  # 좌우 반전

    if cv.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv.destroyAllWindows()