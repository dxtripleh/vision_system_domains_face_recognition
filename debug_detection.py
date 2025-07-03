#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 검출 디버깅 스크립트
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def debug_face_detection(image_path: str):
    """얼굴 검출 디버깅"""
    print(f"\n=== {image_path} 디버깅 ===")
    
    # 1. 이미지 로드 확인
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 이미지 로드 실패")
        return
    
    print(f"✅ 이미지 로드 성공: {img.shape}")
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✅ 그레이스케일 변환: {gray.shape}")
    
    # 3. 히스토그램 평활화
    gray_eq = cv2.equalizeHist(gray)
    print(f"✅ 히스토그램 평활화 완료")
    
    # 4. Cascade 파일 로드 확인
    cascade_path = 'models/weights/haarcascade_frontalface_default.xml'
    if not Path(cascade_path).exists():
        print(f"❌ Cascade 파일 없음: {cascade_path}")
        return
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Cascade 로드 실패")
        return
    
    print("✅ Cascade 로드 성공")
    
    # 5. 다양한 설정으로 검출 시도
    test_configs = [
        {'scale': 1.1, 'neighbors': 3, 'name': '기본 설정'},
        {'scale': 1.05, 'neighbors': 2, 'name': '민감한 설정'},
        {'scale': 1.02, 'neighbors': 1, 'name': '매우 민감한 설정'},
        {'scale': 1.1, 'neighbors': 1, 'name': '매우 민감한 설정 2'},
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} 테스트 ---")
        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=config['scale'],
            minNeighbors=config['neighbors'],
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"검출 결과: {len(faces)}개 얼굴")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  얼굴 {i+1}: ({x}, {y}, {w}x{h})")
    
    # 6. 원본 이미지와 평활화된 이미지 비교
    print(f"\n--- 원본 vs 평활화 비교 ---")
    
    # 원본으로 검출
    faces_original = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(20, 20)
    )
    print(f"원본 이미지 검출: {len(faces_original)}개")
    
    # 평활화된 이미지로 검출
    faces_eq = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(20, 20)
    )
    print(f"평활화 이미지 검출: {len(faces_eq)}개")
    
    # 7. 이미지 품질 분석
    print(f"\n--- 이미지 품질 분석 ---")
    print(f"평균 밝기: {np.mean(gray):.1f}")
    print(f"표준편차: {np.std(gray):.1f}")
    print(f"최소값: {np.min(gray)}")
    print(f"최대값: {np.max(gray)}")
    
    # 8. 얼굴 유사 영역 검색
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    skin_ratio = skin_pixels / total_pixels
    
    print(f"피부색 유사 픽셀: {skin_pixels}/{total_pixels} ({skin_ratio:.3f})")
    
    # 9. 결과 시각화 (첫 번째 검출 결과가 있다면)
    if len(faces_eq) > 0:
        result_img = img.copy()
        for (x, y, w, h) in faces_eq:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        output_path = f"debug_{Path(image_path).stem}_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"✅ 결과 이미지 저장: {output_path}")

def main():
    """메인 함수"""
    # 실제 이미지들 테스트
    image_paths = [
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10535.jpg",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10540.jpg",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10551.jpg"
    ]
    
    for image_path in image_paths:
        if Path(image_path).exists():
            debug_face_detection(image_path)
        else:
            print(f"❌ 파일 없음: {image_path}")

if __name__ == "__main__":
    main() 