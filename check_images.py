#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실제 이미지 내용 확인 스크립트
"""

import cv2
import numpy as np
from pathlib import Path

def check_image_content(image_path: str):
    """이미지 내용 확인"""
    print(f"\n=== {image_path} 분석 ===")
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 이미지 로드 실패")
        return
    
    # 기본 정보
    height, width, channels = img.shape
    print(f"크기: {width}x{height}, 채널: {channels}")
    
    # 픽셀 값 통계
    mean_color = np.mean(img, axis=(0, 1))
    std_color = np.std(img, axis=(0, 1))
    print(f"평균 색상 (BGR): {mean_color}")
    print(f"표준편차 (BGR): {std_color}")
    
    # 밝기 분석
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    print(f"평균 밝기: {mean_brightness:.1f}")
    
    # 대비 분석
    contrast = np.std(gray)
    print(f"대비: {contrast:.1f}")
    
    # 얼굴 유사 영역 검색 (간단한 색상 기반)
    # 피부색 범위 (HSV에서)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 피부색 마스크 (대략적인 범위)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = width * height
    skin_ratio = skin_pixels / total_pixels
    
    print(f"피부색 유사 픽셀 비율: {skin_ratio:.3f} ({skin_pixels}/{total_pixels})")
    
    # 얼굴 검출 시도 (OpenCV)
    face_cascade = cv2.CascadeClassifier('models/weights/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        print(f"✅ OpenCV 얼굴 검출: {len(faces)}개")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  얼굴 {i+1}: ({x}, {y}, {w}x{h})")
    else:
        print("❌ OpenCV 얼굴 검출 실패")
    
    # 이미지 미리보기 (첫 100x100 픽셀)
    preview = img[:100, :100]
    print(f"미리보기 (100x100): 평균 색상 {np.mean(preview, axis=(0, 1))}")

def main():
    """메인 함수"""
    # 실제 이미지들 확인
    image_paths = [
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10535.jpg",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10540.jpg",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10551.jpg"
    ]
    
    for image_path in image_paths:
        if Path(image_path).exists():
            check_image_content(image_path)
        else:
            print(f"❌ 파일 없음: {image_path}")

if __name__ == "__main__":
    main() 