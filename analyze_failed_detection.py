#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실패한 얼굴 검출 분석 및 개선 방안 테스트
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def analyze_failed_image(image_path: str):
    """실패한 이미지 분석"""
    print(f"\n=== {image_path} 실패 원인 분석 ===")
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 이미지 로드 실패")
        return
    
    print(f"✅ 이미지 로드 성공: {img.shape}")
    
    # 1. 기본 품질 분석
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n--- 기본 품질 분석 ---")
    print(f"평균 밝기: {np.mean(gray):.1f}")
    print(f"표준편차: {np.std(gray):.1f}")
    print(f"대비: {np.max(gray) - np.min(gray)}")
    print(f"최소값: {np.min(gray)}")
    print(f"최대값: {np.max(gray)}")
    
    # 2. 다양한 전처리 방법 테스트
    preprocessing_methods = {
        'original': gray,
        'histogram_eq': cv2.equalizeHist(gray),
        'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
        'gaussian_blur': cv2.GaussianBlur(gray, (3,3), 0),
        'median_blur': cv2.medianBlur(gray, 3),
        'bilateral_filter': cv2.bilateralFilter(gray, 9, 75, 75),
        'adaptive_threshold': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    }
    
    # Cascade 로드
    cascade_path = 'models/weights/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    print(f"\n--- 다양한 전처리 방법 테스트 ---")
    for method_name, processed_img in preprocessing_methods.items():
        faces = face_cascade.detectMultiScale(
            processed_img,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )
        print(f"{method_name}: {len(faces)}개 얼굴 검출")
    
    # 3. 다양한 검출 파라미터 테스트
    print(f"\n--- 다양한 검출 파라미터 테스트 (CLAHE 적용) ---")
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    param_configs = [
        {'scale': 1.02, 'neighbors': 1, 'name': '매우 민감'},
        {'scale': 1.05, 'neighbors': 1, 'name': '민감'},
        {'scale': 1.05, 'neighbors': 2, 'name': '기본'},
        {'scale': 1.1, 'neighbors': 1, 'name': '민감 2'},
        {'scale': 1.02, 'neighbors': 0, 'name': '극도 민감'},
    ]
    
    for config in param_configs:
        faces = face_cascade.detectMultiScale(
            clahe_img,
            scaleFactor=config['scale'],
            minNeighbors=config['neighbors'],
            minSize=(15, 15),  # 더 작은 얼굴도 검출
            maxSize=(300, 300)  # 너무 큰 얼굴 제외
        )
        print(f"{config['name']}: {len(faces)}개 얼굴 검출")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  얼굴 {i+1}: ({x}, {y}, {w}x{h})")
    
    # 4. 다중 Cascade 테스트
    print(f"\n--- 다중 Cascade 테스트 ---")
    cascade_files = [
        'models/weights/haarcascade_frontalface_default.xml',
        'models/weights/haarcascade_frontalface_alt.xml',
        'models/weights/haarcascade_frontalface_alt2.xml',
        'models/weights/haarcascade_frontalface_alt_tree.xml'
    ]
    
    for cascade_file in cascade_files:
        if Path(cascade_file).exists():
            cascade = cv2.CascadeClassifier(cascade_file)
            if not cascade.empty():
                faces = cascade.detectMultiScale(
                    clahe_img,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(20, 20)
                )
                print(f"{Path(cascade_file).stem}: {len(faces)}개 얼굴 검출")
        else:
            print(f"{Path(cascade_file).stem}: 파일 없음")
    
    # 5. 이미지 리사이징 테스트
    print(f"\n--- 이미지 리사이징 테스트 ---")
    resize_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for factor in resize_factors:
        if factor != 1.0:
            new_width = int(img.shape[1] * factor)
            new_height = int(img.shape[0] * factor)
            resized_img = cv2.resize(img, (new_width, new_height))
            resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            resized_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(resized_gray)
        else:
            resized_clahe = clahe_img
        
        faces = face_cascade.detectMultiScale(
            resized_clahe,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )
        print(f"리사이즈 {factor}x: {len(faces)}개 얼굴 검출")

def main():
    """메인 함수"""
    # 실패한 이미지 분석
    failed_image = "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10540.jpg"
    
    if Path(failed_image).exists():
        analyze_failed_image(failed_image)
    else:
        print(f"❌ 파일 없음: {failed_image}")

if __name__ == "__main__":
    main() 