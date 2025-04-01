# Computer Vision - L06 Homework


## 파일 구조
- `1.py`: SIFT 특징점 검출 및 시각화
- `2.py`: SIFT 특징점 매칭
- `3.py`: 호모그래피를 이용한 이미지 정합
- `README.md`: 프로젝트 설명 문서

## 과제 상세 설명

### 1. SIFT 특징점 검출 및 시각화 (1.py)

#### 목표
주어진 이미지에서 SIFT 알고리즘을 사용하여 특징점을 검출하고 시각화합니다.

#### 주요 기능
- OpenCV의 `cv.SIFT_create()`를 사용하여 SIFT 객체 생성
- `detectAndCompute()` 메소드로 특징점 및 설명자 추출
- `cv.drawKeypoints()`를 사용하여 검출된 특징점 시각화
- Matplotlib을 활용하여 원본 이미지와 특징점 시각화 이미지 비교 출력
- SIFT 파라미터(nfeatures) 조정을 통한 특징점 개수 제한 기능 제공

#### 결과 이미지
![Image](https://github.com/user-attachments/assets/b10fdcd4-00ff-408b-a961-8d40fbf3da8a)
![Image](https://github.com/user-attachments/assets/9e9fa418-a40f-41e8-8005-932b3d30fe49)
![Image](https://github.com/user-attachments/assets/396f3c11-fb2a-4cf7-a77b-9fa37f93a2c8)




#### 전체 코드
```python
import cv2 as cv  
import matplotlib.pyplot as plt  

def main():
    # 이미지 불러오기
    img = cv.imread('../img/mot_color70.jpg') 
    
    # 이미지가 제대로 로드되었는지 확인
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 그레이스케일 변환 (SIFT는 그레이스케일 이미지에서 동작)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # SIFT 객체 생성 (Scale-Invariant Feature Transform)
    sift = cv.SIFT_create()  # 기본 파라미터로 SIFT 객체 생성
    
    # 특징점 검출 및 설명자 계산
    # kp: 키포인트(위치, 크기, 방향 등의 정보), des: 설명자(특징점 주변 패턴 정보)
    kp, des = sift.detectAndCompute(gray, None)  # None은 마스크 없이 전체 이미지 사용
    
    print(f"검출된 특징점 개수: {len(kp)}")  # 검출된 특징점 수 출력
    
    # 특징점 시각화
    # flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 특징점의 크기와 방향 정보 포함
    img_keypoints = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 이미지를 matplotlib을 이용하여 출력
    # OpenCV는 BGR 순서로 색상을 다루므로 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 원본 이미지 RGB 변환
    img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)  # 특징점 이미지 RGB 변환
    
    # 원본 이미지와 특징점 시각화 이미지를 나란히 출력
    plt.figure(figsize=(12, 6))  # 그림 크기 설정
    
    # 1행 2열, 첫 번째 subplot에 원본 이미지 출력
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')  # 축 눈금 제거
    
    # 1행 2열, 두 번째 subplot에 특징점 이미지 출력
    plt.subplot(1, 2, 2)
    plt.title('SIFT Keypoints Image')
    plt.imshow(img_keypoints_rgb)
    plt.axis('off')
    
    plt.tight_layout()  # 그래프 간격 자동 조정
    plt.show()  # 그래프 표시
    
    # SIFT 매개변수 변경 예시: 특징점 수 제한
    # 특징점이 너무 많을 경우 nfeatures 값을 조정하여 개수 제한
    sift_limited = cv.SIFT_create(nfeatures=100)  # 최대 100개 특징점으로 제한
    kp_limited, des_limited = sift_limited.detectAndCompute(gray, None)
    
    print(f"제한된 특징점 개수: {len(kp_limited)}")
    
    # 제한된 특징점 시각화
    img_keypoints_limited = cv.drawKeypoints(gray, kp_limited, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_keypoints_limited_rgb = cv.cvtColor(img_keypoints_limited, cv.COLOR_BGR2RGB)
    
    # 원본 특징점과 제한된 특징점 비교
    plt.figure(figsize=(12, 6))
    
    # 첫 번째 subplot에 모든 특징점 이미지 출력
    plt.subplot(1, 2, 1)
    plt.title('SIFT Keypoints Keypoints')
    plt.imshow(img_keypoints_rgb)
    plt.axis('off')
    
    # 두 번째 subplot에 제한된 특징점 이미지 출력
    plt.subplot(1, 2, 2)
    plt.title('Limited SIFT Keypoints(n = 100)')
    plt.imshow(img_keypoints_limited_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()

```


### 2. SIFT 특징점 매칭 (2.py)

#### 목표
두 개의 이미지에서 SIFT 특징점을 검출하고 이를 매칭하여 결과를 시각화합니다.

#### 주요 기능
- 두 이미지(mot_color70.jpg, mot_color83.jpg)에서 SIFT 특징점 추출
- BFMatcher와 FLANN 기반 매처 두 가지 방법으로 특징점 매칭 구현
- 거리 비율 테스트(Lowe's ratio test)를 통한 매칭 정확도 향상
- `cv.drawMatches()`를 사용한 매칭 결과 시각화
- 사용자 정의 함수를 통한 개선된 매칭 시각화 제공

#### 매칭 알고리즘 비교
1. **BFMatcher(Brute-Force Matcher)**
   - 모든 특징점을 비교하는 직관적인 방법
   - `cv.NORM_L2` 거리 측정 및 `crossCheck=True`로 상호 매칭 확인

2. **FLANN(Fast Library for Approximate Nearest Neighbors)**
   - 대규모 데이터셋에 더 효율적인 매칭 방법
   - KNN 매칭과 ratio test를 통한 매칭 품질 향상

#### 결과 이미지
![Image](https://github.com/user-attachments/assets/45084936-c904-4feb-8055-7685087c36a6)
![Image](https://github.com/user-attachments/assets/0966567b-e032-4397-b080-7bd569db5edf)


#### 전체 코드
```python
import cv2 as cv 
import matplotlib.pyplot as plt  
import numpy as np 

def main():
    
    img1 = cv.imread('../img/mot_color70.jpg')  
    img2 = cv.imread('../img/mot_color83.jpg')  
    
    # 이미지가 제대로 로드되었는지 확인
    if img1 is None or img2 is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 그레이스케일 변환 (특징점 검출용)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지 그레이스케일 변환
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # 두 번째 이미지 그레이스케일 변환
    
    # SIFT 객체 생성
    sift = cv.SIFT_create()  # 기본 파라미터로 SIFT 객체 생성
    
    # 각 이미지에서 특징점과 설명자 추출
    kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지 특징점 및 설명자
    kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지 특징점 및 설명자
    
    print(f"첫 번째 이미지 특징점 개수: {len(kp1)}")
    print(f"두 번째 이미지 특징점 개수: {len(kp2)}")
    
    #---------------------------------------------------------------------
    # 방법 1: BFMatcher(Brute-Force Matcher)를 사용한 매칭
    #---------------------------------------------------------------------
    
    # cv.NORM_L2: 유클리드 거리 사용, crossCheck=True: 상호 매칭 확인 
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # 모든 특징점 간 매칭 수행
    
    # 거리에 따라 매치 정렬 (거리가 작을수록 유사도가 높음)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 상위 50개 매치만 시각화 (가장 유사도가 높은 매칭)
    good_matches = matches[:50]
    
    # 매칭 결과 시각화
    # DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS: 매칭된 점만 표시
    img_matches_bf = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    #---------------------------------------------------------------------
    # 방법 2: FLANN 기반 매처 사용
    #---------------------------------------------------------------------
    
    # FLANN 매개변수 설정 (Fast Library for Approximate Nearest Neighbors)
    FLANN_INDEX_KDTREE = 1  # KD-트리 인덱스 사용
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 5개의 트리 사용
    search_params = dict(checks=50)  # 50회 검색 반복
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    # knnMatch를 사용하여 최근접 이웃 찾기 (k=2: 각 특징점에 대해 가장 유사한 2개 찾기)
    knn_matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe의 ratio test를 적용하여 좋은 매치 선택
    # 첫 번째 매치가 두 번째 매치보다 충분히 좋은 경우만 선택
    ratio_thresh = 0.7  # 비율 임계값 (0.7 권장)
    good_knn_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:  # 첫 번째 매치가 두 번째의 70% 이하 거리
            good_knn_matches.append(m)
    
    print(f"BFMatcher로 매칭된 특징점 개수: {len(good_matches)}")
    print(f"FLANN + ratio test 후 매칭된 특징점 개수: {len(good_knn_matches)}")
    
    # FLANN 매칭 결과 시각화
    img_matches_flann = cv.drawMatches(img1, kp1, img2, kp2, good_knn_matches, None, 
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 이미지를 matplotlib을 이용하여 출력 (BGR에서 RGB로 변환)
    img_matches_bf_rgb = cv.cvtColor(img_matches_bf, cv.COLOR_BGR2RGB)
    img_matches_flann_rgb = cv.cvtColor(img_matches_flann, cv.COLOR_BGR2RGB)
    
    # 결과 시각화 - BFMatcher와 FLANN 비교
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 1, 1)
    plt.title('Match : BFMatcher')
    plt.imshow(img_matches_bf_rgb)
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title('Match : FLANN + Ratio Test')
    plt.imshow(img_matches_flann_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()

```

---

### 3. 호모그래피를 이용한 이미지 정합 (3.py)

#### 목표
SIFT 특징점 매칭을 기반으로 두 이미지 간의 호모그래피를 계산하고, 이를 이용하여 한 이미지를 다른 이미지의 시점으로 변환(정합)합니다.

#### 주요 기능
- SIFT 특징점 검출 및 FLANN 기반 매칭
- ratio test를 통한 정확한 대응점 선별
- `cv.findHomography()`와 RANSAC을 사용한 강인한 호모그래피 계산
- `cv.warpPerspective()`를 통한 이미지 변환
- 다양한 시각화 방법(경계 박스, 이미지 합성 등)을 통한 결과 검증

#### 호모그래피 및 이미지 정합 과정
1. 두 이미지에서 특징점 추출 및 매칭
2. 매칭된 특징점에서 좌표 추출
3. RANSAC 방법으로 호모그래피 행렬 계산
4. 첫 번째 이미지를 호모그래피를 이용하여 변환
5. 변환된 이미지와 두 번째 이미지를 합성하여 정합 결과 확인

#### 결과 이미지
![Image](https://github.com/user-attachments/assets/99d073d4-4f43-417b-9d07-d8420e0b0fa1)
![Image](https://github.com/user-attachments/assets/728509b8-b5c5-4ec5-bab1-75f904d5181e)
#### 전체 코드
```python
import cv2 as cv 
import numpy as np  
import matplotlib.pyplot as plt  

def main():
    
    img1 = cv.imread('../img/img1.jpg')  
    img2 = cv.imread('../img/img2.jpg')  
    
    # 이미지가 제대로 로드되었는지 확인
    if img1 is None or img2 is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 그레이스케일 변환 (특징점 검출용)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  
    
    # SIFT 객체 생성
    sift = cv.SIFT_create()  # 기본 파라미터로 SIFT 객체 생성
    
    # 각 이미지에서 특징점과 설명자 추출
    kp1, des1 = sift.detectAndCompute(gray1, None)  
    kp2, des2 = sift.detectAndCompute(gray2, None)  
    
    print(f"첫 번째 이미지 특징점 개수: {len(kp1)}")
    print(f"두 번째 이미지 특징점 개수: {len(kp2)}")
    
    # bf 기반 매처 설정
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)  # crossCheck=False: 양방향 매칭 비활성화
    
    # 매칭 수행 (모든 특징점 간 매칭)
    matches = bf.match(des1, des2)
    
    # knnMatch를 사용하여 각 특징점마다 가장 가까운 2개의 매칭점 찾기
    knn_matches = bf.knnMatch(des1, des2, 2)  # k=2: 각 특징점에 대해 상위 2개 매칭
    
    # Lowe의 ratio test를 적용하여 좋은 매치 선택
    # 첫 번째 매치가 두 번째 매치보다 충분히 좋은 경우만 선택
    ratio = 0.7  # 비율 임계값 (0.7 권장)
    good_match = []
    for nearest1, nearest2 in knn_matches:
        if nearest1.distance < nearest2.distance * ratio:  # 첫 번째 매치가 두 번째의 70% 이하 거리
            good_match.append(nearest1)
    
    print(f"좋은 매치 개수: {len(good_match)}")
    
    # 매칭점에서 좌표 추출
    # queryIdx: 첫 번째 이미지에서의 특징점 인덱스, trainIdx: 두 번째 이미지에서의 특징점 인덱스
    points1 = np.float32([kp1[match.queryIdx].pt for match in good_match])  # 첫 번째 이미지 대응점
    points2 = np.float32([kp2[match.trainIdx].pt for match in good_match])  # 두 번째 이미지 대응점
    
    # RANSAC을 사용하여 호모그래피 계산
    # RANSAC(Random Sample Consensus): 이상치(outlier)에 강인한 모델 추정 알고리즘
    H, _ = cv.findHomography(points1, points2, cv.RANSAC)  # H: 호모그래피 행렬
    
    # 이미지 크기 가져오기
    h1, w1 = img1.shape[:2]  # 첫 번째 이미지 높이, 너비
    h2, w2 = img2.shape[:2]  # 두 번째 이미지 높이, 너비
    
    # 변환 후 이미지 크기 계산 (두 번째 이미지 크기로 설정)
    # 첫 번째 이미지의 네 모서리 좌표
    box1 = np.float32([(0,0), (0,h1-1), (w1-1,h1-1), (w1-1,0)]).reshape(4,1,2)
    # 호모그래피를 적용하여 첫 번째 이미지의 모서리가 두 번째 이미지에서 어디에 위치하는지 계산
    box2 = cv.perspectiveTransform(box1, H)
    
    # 두 번째 이미지 위에 첫 번째 이미지 변환하여 합치기
    # 호모그래피로 계산된 영역을 녹색 선으로 표시
    img2_warped = cv.polylines(img2.copy(), [np.int32(box2)], True, (0,255,0), 3)
    
    # 첫 번째 이미지를 호모그래피를 사용하여 변환
    # 두 이미지를 나란히 배치하고 매칭 라인 그리기
    img_match = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
    cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 결과 이미지를 matplotlib으로 시각화
    plt.figure(figsize=(15, 10))
    
    # 매칭 결과 표시
    plt.subplot(2, 2, 1)
    plt.title('Matches and Homography')
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 두 번째 이미지에 변환 영역 표시
    plt.subplot(2, 2, 2)
    plt.title('Projected Region on Image 2')
    plt.imshow(cv.cvtColor(img2_warped, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 첫 번째 이미지를 두 번째 이미지 공간으로 변환
    # 호모그래피를 적용하여 첫 번째 이미지를 두 번째 이미지의 크기와 시점으로 변환
    warped_img = cv.warpPerspective(img1, H, (w2, h2))
    
    plt.subplot(2, 2, 3)
    plt.title('Image 1 Warped to Image 2 Space')
    plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 변환된 이미지와 두 번째 이미지 합성
    # 두 영상을 반투명하게 합성하여 정합 결과 확인
    alpha = 0.5  # 첫 번째 이미지 투명도
    beta = 1.0 - alpha  # 두 번째 이미지 투명도
    blended = cv.addWeighted(warped_img, alpha, img2, beta, 0.0)
    
    plt.subplot(2, 2, 4)
    plt.title('Blended Result (Alignment Verification)')
    plt.imshow(cv.cvtColor(blended, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    


# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()


```


