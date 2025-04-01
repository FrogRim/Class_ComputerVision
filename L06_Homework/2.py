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
    
    #---------------------------------------------------------------------
    # 개선된 매칭 결과 시각화 (더 많은 정보 표시)
    #---------------------------------------------------------------------
    
    # 두 이미지를 나란히 배치하고 매칭 라인 그리기 위한 사용자 정의 함수
    def draw_matches_with_info(img1, kp1, img2, kp2, matches):
        # 이미지 크기 가져오기
        h1, w1 = img1.shape[:2]  # 첫 번째 이미지 높이, 너비
        h2, w2 = img2.shape[:2]  # 두 번째 이미지 높이, 너비
        
        # 결과 이미지 생성 (두 이미지를 나란히 배치할 빈 캔버스)
        combined_height = max(h1, h2)  # 두 이미지 중 더 큰 높이 사용
        result = np.zeros((combined_height, w1 + w2, 3), dtype=np.uint8)  # 검은색 배경
        
        # 두 이미지 복사
        result[:h1, :w1] = img1  # 왼쪽에 첫 번째 이미지 배치
        result[:h2, w1:w1+w2] = img2  # 오른쪽에 두 번째 이미지 배치
        
        # BGR에서 RGB로 변환 (matplotlib 표시용)
        result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        
        # 매칭 라인 그리기
        for match in matches:
            # 첫 번째 이미지의 키포인트 위치
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            # 두 번째 이미지의 키포인트 위치 (x 좌표에 첫 번째 이미지 너비 추가)
            pt2 = (int(kp2[match.trainIdx].pt[0] + w1), int(kp2[match.trainIdx].pt[1]))
            
            # 매칭 라인 그리기 (파란색)
            cv.line(result_rgb, pt1, pt2, (255, 0, 0), 1)
            # 키포인트 표시 (초록색 원)
            cv.circle(result_rgb, pt1, 5, (0, 255, 0), 2)
            cv.circle(result_rgb, pt2, 5, (0, 255, 0), 2)
            
        return result_rgb
    
    # 개선된 매칭 시각화 (상위 30개 매칭만 표시)
    improved_matches = draw_matches_with_info(img1, kp1, img2, kp2, good_knn_matches[:30])
    
    plt.figure(figsize=(16, 8))
    plt.title('Improved Match : (FLANN + Ratio Test, top 30)')
    plt.imshow(improved_matches)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
