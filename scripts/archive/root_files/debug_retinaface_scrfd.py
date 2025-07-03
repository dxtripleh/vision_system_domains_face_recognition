#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetinaFace와 SCRFD 모델 출력 분석
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_retinaface_output():
    """RetinaFace 모델 출력 분석"""
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    model_path = Path('models/weights/face_detection_retinaface_resnet50.onnx')
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 존재하지 않음: {model_path}")
        return
    
    try:
        # 모델 로드
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print(f"✅ RetinaFace 모델 로드 성공")
        
        # RetinaFace 전처리
        blob = cv2.dnn.blobFromImage(test_image, 1.0, (640, 640), (104, 117, 123), True, False)
        
        print(f"📊 입력 blob shape: {blob.shape}")
        
        # 추론
        net.setInput(blob)
        outputs = net.forward()
        
        print(f"📊 출력 개수: {len(outputs)}")
        
        # 각 출력 분석
        for i, output in enumerate(outputs):
            print(f"\n출력 {i}:")
            print(f"  shape: {output.shape}")
            print(f"  dtype: {output.dtype}")
            print(f"  min: {output.min()}")
            print(f"  max: {output.max()}")
            print(f"  mean: {output.mean()}")
            
            # 첫 몇 개 값 출력
            if len(output.shape) == 1:
                print(f"  첫 10개 값: {output[:10]}")
            elif len(output.shape) == 2:
                print(f"  첫 3행: {output[:3]}")
        
    except Exception as e:
        print(f"❌ RetinaFace 분석 실패: {e}")

def analyze_scrfd_output():
    """SCRFD 모델 출력 분석"""
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    model_path = Path('models/weights/face_detection_scrfd_10g_20250628.onnx')
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 존재하지 않음: {model_path}")
        return
    
    try:
        # 모델 로드
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print(f"✅ SCRFD 모델 로드 성공")
        
        # SCRFD 전처리
        blob = cv2.dnn.blobFromImage(test_image, 1.0, (640, 640), (0, 0, 0), True, False)
        
        print(f"📊 입력 blob shape: {blob.shape}")
        
        # 추론
        net.setInput(blob)
        outputs = net.forward()
        
        print(f"📊 출력 개수: {len(outputs)}")
        
        # 각 출력 분석
        for i, output in enumerate(outputs):
            print(f"\n출력 {i}:")
            print(f"  shape: {output.shape}")
            print(f"  dtype: {output.dtype}")
            print(f"  min: {output.min()}")
            print(f"  max: {output.max()}")
            print(f"  mean: {output.mean()}")
            
            # 첫 몇 개 값 출력
            if len(output.shape) == 1:
                print(f"  첫 10개 값: {output[:10]}")
            elif len(output.shape) == 2:
                print(f"  첫 3행: {output[:3]}")
        
    except Exception as e:
        print(f"❌ SCRFD 분석 실패: {e}")

if __name__ == "__main__":
    print("🔍 RetinaFace 모델 분석")
    print("="*50)
    analyze_retinaface_output()
    
    print("\n\n🔍 SCRFD 모델 분석")
    print("="*50)
    analyze_scrfd_output() 