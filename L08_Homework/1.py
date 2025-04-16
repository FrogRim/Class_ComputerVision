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