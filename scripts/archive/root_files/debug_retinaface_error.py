#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetinaFace 모델 오류 분석
"""

import cv2
import numpy as np
from pathlib import Path

def debug_retinaface_error():
    """RetinaFace 모델 오류 디버깅"""
    
    model_path = "models/weights/face_detection_retinaface_resnet50.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # 테스트 이미지
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 전처리
    blob = cv2.dnn.blobFromImage(
        test_image,
        scalefactor=1.0,
        size=(640, 640),
        mean=[104.0, 117.0, 123.0],
        swapRB=True,
        crop=False
    )
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    
    print(f"출력 개수: {len(outputs)}")
    
    # 현재 코드에서 오류가 발생하는 부분 시뮬레이션
    try:
        if len(outputs) >= 3:
            boxes = outputs[0][0]      # 여기서 오류 발생!
            scores = outputs[1][0]     # 여기서 오류 발생!
            landmarks = outputs[2][0]  # 여기서 오류 발생!
            
            print("✅ 인덱싱 성공")
            
    except Exception as e:
        print(f"❌ 인덱싱 오류: {e}")
        
        # 실제 출력 형태 확인
        for i, output in enumerate(outputs):
            print(f"출력 {i}: {output.shape}, 타입: {type(output)}")
            
            # 스칼라인지 확인
            if np.isscalar(output):
                print(f"  스칼라 값: {output}")
            elif hasattr(output, 'shape'):
                print(f"  배열 형태: {output.shape}")
                if len(output.shape) == 1:
                    print(f"  1차원 배열: {output[:5]}...")  # 처음 5개 값만
                elif len(output.shape) == 2:
                    print(f"  2차원 배열: {output.shape}")
                    print(f"  첫 번째 행: {output[0][:5]}...")
                elif len(output.shape) == 3:
                    print(f"  3차원 배열: {output.shape}")
                    print(f"  첫 번째 배치: {output[0].shape}")
            else:
                print(f"  기타 타입: {type(output)}")

def fix_retinaface_processing():
    """RetinaFace 처리 로직 수정"""
    
    model_path = "models/weights/face_detection_retinaface_resnet50.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # 테스트 이미지
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 전처리
    blob = cv2.dnn.blobFromImage(
        test_image,
        scalefactor=1.0,
        size=(640, 640),
        mean=[104.0, 117.0, 123.0],
        swapRB=True,
        crop=False
    )
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    
    print(f"\n🔧 RetinaFace 처리 로직 수정:")
    
    # 안전한 출력 처리
    if len(outputs) >= 3:
        try:
            # 출력 형태에 따른 안전한 처리
            if hasattr(outputs[0], 'shape') and len(outputs[0].shape) >= 2:
                boxes = outputs[0][0]  # [n_anchors, 4]
            else:
                boxes = outputs[0]     # 직접 사용
                
            if hasattr(outputs[1], 'shape') and len(outputs[1].shape) >= 2:
                scores = outputs[1][0]  # [n_anchors, 2]
            else:
                scores = outputs[1]     # 직접 사용
                
            if hasattr(outputs[2], 'shape') and len(outputs[2].shape) >= 2:
                landmarks = outputs[2][0]  # [n_anchors, 10]
            else:
                landmarks = outputs[2]     # 직접 사용
            
            print(f"✅ 안전한 처리 성공")
            print(f"  박스 형태: {boxes.shape if hasattr(boxes, 'shape') else '스칼라'}")
            print(f"  신뢰도 형태: {scores.shape if hasattr(scores, 'shape') else '스칼라'}")
            print(f"  특징점 형태: {landmarks.shape if hasattr(landmarks, 'shape') else '스칼라'}")
            
        except Exception as e:
            print(f"❌ 안전한 처리도 실패: {e}")

if __name__ == "__main__":
    print("🔍 RetinaFace 오류 분석 시작\n")
    debug_retinaface_error()
    fix_retinaface_processing() 