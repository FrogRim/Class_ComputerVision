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
