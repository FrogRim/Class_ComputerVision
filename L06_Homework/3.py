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
    
    # FLANN 기반 매처 설정
    flann = cv.FlannBasedMatcher()
    
    # knnMatch를 사용하여 각 특징점마다 가장 가까운 2개의 매칭점 찾기
    knn_matches = flann.knnMatch(des1, des2, 2)  # k=2: 각 특징점에 대해 상위 2개 매칭
    
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
    
    #---------------------------------------------------------------------
    # 추가: 대체 시각화 방법 (색상으로 구분)
    #---------------------------------------------------------------------
    
    # 원본 컬러 채널 유지를 위해 다른 방식으로 시각화
    result = img2.copy()
    
    # 변환된 이미지의 마스크 생성 (검은색 부분 제외)
    # 변환된 이미지에서 실제 콘텐츠가 있는 부분만 마스크로 생성
    gray_warped = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_warped, 1, 255, cv.THRESH_BINARY)  # 임계값 1 이상인 부분만 255로 설정
    
    # 마스크를 사용하여 원본 이미지 일부 영역을 빨간색으로 변경
    red_region = np.zeros_like(img2)  # 두 번째 이미지와 같은 크기의 검은 이미지
    red_region[:,:,2] = mask  # 빨간색 채널에 마스크 적용 (B=0, G=0, R=mask)
    
    # 두 이미지 합성
    result = cv.addWeighted(result, 1.0, red_region, 0.5, 0.0)  # 원본 + 빨간 마스크
    
    # 결과 시각화
    plt.figure(figsize=(10, 8))
    plt.title('Alignment Result (Red: Warped Image Region)')
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
