#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 검출 디버깅 스크립트

이 스크립트는 얼굴 검출이 잘 안되는 문제를 진단합니다.
"""

import os
import sys
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

def test_opencv_detection():
    """OpenCV 얼굴 검출 테스트"""
    print("=== OpenCV 얼굴 검출 테스트 ===")
    
    # OpenCV 검출기 설정
    config = {
        'cascade_path': 'models/weights/haarcascade_frontalface_default.xml',
        'confidence_threshold': 0.3,
        'scale_factor': 1.05,
        'min_neighbors': 2,
        'min_size': (20, 20)
    }
    
    try:
        detector = OpenCVDetectionEngine(config)
        print("✓ OpenCV 검출기 로딩 성공")
    except Exception as e:
        print(f"✗ OpenCV 검출기 로딩 실패: {e}")
        return
    
    # 테스트 이미지들
    test_images = [
        "data/domains/face_recognition/raw_input/uploads/image01.png",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10535.jpg",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10540.jpg"
    ]
    
    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"⚠️ 이미지 파일 없음: {img_path}")
            continue
            
        print(f"\n--- 테스트 이미지: {img_path} ---")
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"✗ 이미지 로드 실패: {img_path}")
            continue
        
        print(f"이미지 크기: {image.shape}")
        
        # 얼굴 검출
        try:
            detections = detector.detect(image)
            print(f"검출된 얼굴 수: {len(detections)}")
            
            if detections:
                for i, detection in enumerate(detections):
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('confidence', 0)
                    print(f"  얼굴 {i+1}: bbox={bbox}, confidence={confidence:.3f}")
            else:
                print("  얼굴이 검출되지 않음")
                
        except Exception as e:
            print(f"✗ 검출 실패: {e}")

def test_onnx_detection():
    """ONNX 얼굴 검출 테스트"""
    print("\n=== ONNX 얼굴 검출 테스트 ===")
    
    try:
        import onnxruntime as ort
        print("✓ ONNX Runtime 사용 가능")
    except ImportError:
        print("✗ ONNX Runtime 설치되지 않음")
        return
    
    # ONNX 모델 경로
    model_path = "models/weights/face_detection_retinaface_resnet50.onnx"
    
    if not Path(model_path).exists():
        print(f"✗ ONNX 모델 파일 없음: {model_path}")
        return
    
    try:
        # ONNX 세션 생성
        providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 2
        
        session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
        print(f"✓ ONNX 모델 로딩 성공: {model_path}")
        
        # 입력 정보
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"입력 이름: {input_name}")
        print(f"입력 형태: {input_shape}")
        
    except Exception as e:
        print(f"✗ ONNX 모델 로딩 실패: {e}")
        return
    
    # 테스트 이미지
    test_image = "data/domains/face_recognition/raw_input/uploads/image01.png"
    
    if not Path(test_image).exists():
        print(f"⚠️ 테스트 이미지 없음: {test_image}")
        return
    
    print(f"\n--- ONNX 테스트 이미지: {test_image} ---")
    
    # 이미지 로드 및 전처리
    image = cv2.imread(test_image)
    if image is None:
        print(f"✗ 이미지 로드 실패: {test_image}")
        return
    
    print(f"원본 이미지 크기: {image.shape}")
    
    # 전처리 (640x640으로 리사이즈)
    target_size = 640
    h, w = image.shape[:2]
    
    # 비율 유지하면서 리사이즈
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # 패딩으로 정사각형 만들기
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # BGR to RGB 변환
    input_tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    print(f"전처리된 텐서 형태: {input_tensor.shape}")
    
    # 추론 실행
    try:
        outputs = session.run(None, {input_name: input_tensor})
        print(f"출력 개수: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"출력 {i} 형태: {output.shape}")
            if len(output.shape) == 3:
                print(f"출력 {i} 샘플값: {output[0, :5, :5]}")
        
    except Exception as e:
        print(f"✗ 추론 실패: {e}")
        import traceback
        traceback.print_exc()

def analyze_image_quality():
    """이미지 품질 분석"""
    print("\n=== 이미지 품질 분석 ===")
    
    test_images = [
        "data/domains/face_recognition/raw_input/uploads/image01.png",
        "data/domains/face_recognition/raw_input/captured/20250703/cap_20250703_10535.jpg"
    ]
    
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\n--- 이미지 품질 분석: {img_path} ---")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        print(f"해상도: {w}x{h}")
        
        # 밝기 분석
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        print(f"평균 밝기: {mean_brightness:.1f}")
        print(f"밝기 표준편차: {std_brightness:.1f}")
        
        # 대비 분석
        contrast = np.std(gray)
        print(f"대비: {contrast:.1f}")
        
        # 블러 분석
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"블러 정도: {laplacian_var:.1f}")
        
        if laplacian_var < 100:
            print("⚠️ 이미지가 흐릿할 수 있습니다")
        if mean_brightness < 50:
            print("⚠️ 이미지가 어두울 수 있습니다")
        if mean_brightness > 200:
            print("⚠️ 이미지가 너무 밝을 수 있습니다")

def main():
    """메인 함수"""
    print("얼굴 검출 디버깅 시작")
    
    # 1. OpenCV 검출 테스트
    test_opencv_detection()
    
    # 2. ONNX 검출 테스트
    test_onnx_detection()
    
    # 3. 이미지 품질 분석
    analyze_image_quality()
    
    print("\n=== 디버깅 완료 ===")

if __name__ == "__main__":
    main() 