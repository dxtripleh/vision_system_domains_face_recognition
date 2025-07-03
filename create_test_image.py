#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실제 얼굴이 있는 테스트 이미지 생성 스크립트
"""

import cv2
import numpy as np
from datetime import datetime
import random
import os

def create_test_face_image():
    """실제 얼굴이 있는 테스트 이미지 생성"""
    
    # 현재 시간을 기반으로 고유한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 밀리초까지
    filename = f"cap_{timestamp}.jpg"
    
    # 640x480 크기의 이미지 생성
    height, width = 480, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 50  # 어두운 배경
    
    # 여러 개의 가상 얼굴 영역 생성
    face_regions = [
        (150, 100, 120, 150),  # (x, y, width, height)
        (400, 200, 100, 120),
        (250, 280, 130, 160)
    ]
    
    for i, (x, y, w, h) in enumerate(face_regions):
        # 얼굴 형태의 타원 생성
        center = (x + w//2, y + h//2)
        axes = (w//2, h//2)
        
        # 피부색 계열의 색상
        skin_colors = [
            (180, 150, 120),  # 밝은 피부
            (160, 130, 100),  # 중간 피부
            (140, 110, 80)    # 어두운 피부
        ]
        skin_color = random.choice(skin_colors)
        
        # 얼굴 타원 그리기
        cv2.ellipse(image, center, axes, 0, 0, 360, skin_color, -1)
        
        # 눈 그리기
        eye_y = y + h//3
        left_eye_x = x + w//3
        right_eye_x = x + 2*w//3
        cv2.circle(image, (left_eye_x, eye_y), 8, (0, 0, 0), -1)
        cv2.circle(image, (right_eye_x, eye_y), 8, (0, 0, 0), -1)
        
        # 코 그리기
        nose_x = x + w//2
        nose_y = y + h//2
        cv2.circle(image, (nose_x, nose_y), 5, (100, 100, 100), -1)
        
        # 입 그리기
        mouth_y = y + 2*h//3
        cv2.ellipse(image, (x + w//2, mouth_y), (w//4, h//8), 0, 0, 180, (0, 0, 0), 2)
    
    # 파일 저장
    cv2.imwrite(filename, image)
    print(f"테스트 이미지 생성: {filename}")
    
    return filename

if __name__ == "__main__":
    create_test_face_image() 