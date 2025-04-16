# Dynamic Vision Homework

---

## 1. `1.py`: YOLOv4를 활용한 객체 검출 및 SORT 기반 객체 추적

### 주요 기능
- **YOLOv4 객체 검출**: 사전 학습된 YOLOv4-tiny 모델을 사용하여 입력 프레임에서 객체를 검출합니다.
- **SORT 추적기**: 검출된 객체를 SORT 알고리즘을 사용하여 추적하며, 각 객체에 고유 ID를 부여합니다.
- **결과 시각화**: 검출된 객체의 경계 상자와 ID를 실시간으로 비디오 프레임에 표시합니다.

### 주요 함수 및 로직
#### 1. `construct_yolo_v3()`
- **기능**: YOLOv4-tiny 모델을 초기화하고, 출력 레이어와 클래스 이름을 반환합니다.
- **로직**:
  - `cv.dnn.readNet()`을 사용하여 YOLO 모델과 구성 파일을 로드합니다.
  - 모델의 출력 레이어 이름을 추출합니다.
  - 클래스 이름(`classes.txt`)을 읽어 리스트로 반환합니다.

#### 2. `yolo_detect(img, yolo_model, out_layers)`
- **기능**: YOLO 모델을 사용하여 입력 이미지에서 객체를 검출합니다.
- **로직**:
  - 입력 이미지를 YOLO 모델에 맞게 전처리(`cv.dnn.blobFromImage`)합니다.
  - YOLO 모델의 출력 레이어를 통해 객체를 검출합니다.
  - 신뢰도가 50% 이상인 객체만 필터링하고, 비최대 억제(NMS)를 적용하여 최종 객체를 반환합니다.

#### 3. 메인 루프
- **기능**: YOLO 모델로 객체를 검출하고 SORT 추적기를 사용하여 객체를 추적합니다.
- **로직**:
  - YOLO를 통해 검출된 객체 중 사람(`class_id == 0`)만 필터링합니다.
  - SORT 추적기를 사용하여 객체를 추적하고, 각 객체에 고유 ID를 부여합니다.
  - OpenCV를 사용하여 경계 상자와 ID를 비디오 프레임에 표시합니다.
### 코드 전문
```python
import numpy as np
import cv2 as cv
import sys
sys.path.append('C:\\Users\\user\\Class_ComputerVision\\L08_Homework\\sort')
from sort import Sort

# 사전 학습 모델을 읽어 YOLO 구성
def construct_yolo_v3():
    f = open('classes.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    f.close()

    model = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

# YOLO 모델
def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[:2]
    test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)

    yolo_model.setInput(test_img)
    output = yolo_model.forward(out_layers)

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 번호
    for out in output:
        for vec85 in out:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 신뢰도가 50% 이상일 경우에만 처리
                center_x, center_y = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                box.append([center_x - w // 2, center_y - h // 2, w, h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [(box[i], conf[i], id[i]) for i in range(len(box)) if i in ind]

    return objects

model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(100, 3))  # 100개의 클래스 트랙 구분


sort = Sort()  # Sort 클래스의 sort 객체 생성

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit("카메라 연결 실패")  # 카메라 연결 실패 시 종료

# 사람 추적
while True:
    ret, frame = cap.read()
    if not ret: sys.exit("캡처된 화면이 없거나 프레임을 읽을 수 없습니다.")

    res = yolo_detect(frame, model, out_layers)  # YOLO v3로 검출
    persons = [
        [box[0], box[1], box[0] + box[2], box[1] + box[3], conf]
        for (box, conf, class_id) in res if class_id == 0  # 사람 클래스만 필터링
    ]

    if len(persons) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(persons))

    for i in range(len(tracks)):
        x1, y1, x2, y2, track_id = tracks[i].astype(int)
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % 100], 2)
        cv.putText(frame, f"ID:{int(track_id)}", (x1, y1 - 10), cv.FONT_HERSHEY_PLAIN, 1, colors[track_id % 100], 2)

    cv.imshow("Person tracking by SORT", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()
```
### 결과 이미지
![Image](https://github.com/user-attachments/assets/c068457d-7ac2-4d2d-b762-15d114f111c9)


---

## 2. `2.py`: MediaPipe FaceMesh를 활용한 얼굴 랜드마크 검출

### 주요 기능
- **얼굴 랜드마크 검출**: MediaPipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화 한 뒤, 얼굴의 468개 랜드마크를 검출합니다.
- **결과 시각화**: 검출된 랜드마크를 실시간 비디오 프레임에 점으로 표시합니다.

### 주요 함수 및 로직
#### 1. MediaPipe FaceMesh 초기화
- **기능**: MediaPipe의 FaceMesh 객체를 초기화합니다.
- **로직**:
  - `mp.solutions.face_mesh.FaceMesh`를 사용하여 최대 2개의 얼굴을 검출하도록 설정합니다.
  - 랜드마크 검출 신뢰도(`min_detection_confidence`)와 추적 신뢰도(`min_tracking_confidence`)를 각각 0.5로 설정합니다.

#### 2. 랜드마크 검출 및 시각화
- **기능**: 얼굴 랜드마크를 검출하고, OpenCV를 사용하여 각 랜드마크를 점으로 표시합니다.
- **로직**:
  - `mesh.process()`를 사용하여 입력 프레임에서 얼굴 랜드마크를 검출합니다.
  - 검출된 랜드마크의 정규화된 좌표(`x`, `y`)를 이미지 크기에 맞게 변환합니다.
  - OpenCV의 `cv.circle()`을 사용하여 각 랜드마크를 점으로 표시합니다.

#### 3. 메인 루프
- **기능**: 실시간으로 얼굴 랜드마크를 검출하고 시각화합니다.
- **로직**:
  - OpenCV를 사용하여 웹캠에서 프레임을 캡처합니다.
  - MediaPipe FaceMesh를 통해 얼굴 랜드마크를 검출합니다.
  - 검출된 랜드마크를 비디오 프레임에 시각화하고 ESC 키를 누르면 종료합니다.

---

## 주요 기술 및 라이브러리
1. **OpenCV**:
   - 비디오 캡처 및 프레임 처리.
   - 객체 검출 결과와 랜드마크를 시각화.
2. **MediaPipe**:
   - FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출.
3. **YOLOv4-tiny**:
   - 사전 학습된 YOLO 모델을 사용하여 객체 검출.
4. **SORT**:
   - 객체 추적 알고리즘으로, 검출된 객체를 추적하고 고유 ID를 부여.

### 코드 전문
```python
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
```
### 결과 이미지
![Image](https://github.com/user-attachments/assets/3fe6b2b7-e871-4b29-af27-8d42e54d2a9c)
---

## 실행 방법
1. **필요한 파일 준비**:
   - `1.py` 실행을 위해 다음 파일이 필요합니다:
     - `yolov4-tiny.weights`
     - `yolov4-tiny.cfg`
     - `classes.txt`
   - `2.py` 실행을 위해 추가 파일은 필요하지 않습니다.

2. **필요한 패키지 설치**:
   ```bash
   pip install numpy opencv-python mediapipe
