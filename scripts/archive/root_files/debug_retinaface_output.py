#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetinaFace 모델 출력 분석 스크립트
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_retinaface_output():
    """RetinaFace 모델 출력 상세 분석"""
    
    # 모델 로드
    model_path = "models/weights/face_detection_retinaface_resnet50.onnx"
    if not Path(model_path).exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        print(f"✅ RetinaFace 모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 중앙에 얼굴 영역 생성
    center_x, center_y = 320, 240
    face_size = 150
    test_image[center_y-face_size//2:center_y+face_size//2, 
               center_x-face_size//2:center_x+face_size//2] = np.random.randint(200, 255, (face_size, face_size, 3))
    
    # RetinaFace 전처리
    input_size = (640, 640)
    blob = cv2.dnn.blobFromImage(
        test_image,
        scalefactor=1.0,
        size=input_size,
        mean=[104.0, 117.0, 123.0],
        swapRB=True,
        crop=False
    )
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    
    print(f"\n🎯 RetinaFace 출력 분석:")
    print(f"출력 개수: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"\n--- 출력 {i} ---")
        print(f"형태: {output.shape}")
        print(f"데이터 타입: {output.dtype}")
        print(f"값 범위: {output.min():.3f} ~ {output.max():.3f}")
        
        if len(output.shape) == 3:
            print(f"배치 크기: {output.shape[0]}")
            print(f"첫 번째 배치 형태: {output[0].shape}")
    
    # 예상 출력 형태와 비교
    expected_outputs = {
        0: "바운딩 박스 (n_anchors, 4)",
        1: "신뢰도 점수 (n_anchors, 2)", 
        2: "특징점 (n_anchors, 10)"
    }
    
    print(f"\n📋 출력 형태 분석:")
    for i, (output, expected) in enumerate(zip(outputs, expected_outputs.items())):
        print(f"출력 {i}: {output.shape} - {expected[1]}")
    
    # 특징점 출력 확인
    if len(outputs) >= 3:
        landmarks_output = outputs[2]
        print(f"\n🔍 특징점 출력 상세:")
        print(f"특징점 출력 형태: {landmarks_output.shape}")
        
        if len(landmarks_output.shape) == 3:
            landmarks_batch = landmarks_output[0]  # 첫 번째 배치
            print(f"특징점 배치 형태: {landmarks_batch.shape}")
            
            # 특징점 값 확인
            if landmarks_batch.size > 0:
                print(f"특징점 값 범위: {landmarks_batch.min():.3f} ~ {landmarks_batch.max():.3f}")
                
                # 유효한 특징점 개수 확인
                valid_landmarks = 0
                for i in range(landmarks_batch.shape[0]):
                    landmark = landmarks_batch[i]
                    if len(landmark) == 10:  # 5점 x 2좌표
                        # 모든 좌표가 0~1 범위 내인지 확인
                        if np.all((landmark >= 0) & (landmark <= 1)):
                            valid_landmarks += 1
                
                print(f"유효한 특징점 개수: {valid_landmarks}/{landmarks_batch.shape[0]}")

def test_retinaface_detection():
    """RetinaFace 검출 테스트"""
    print(f"\n🧪 RetinaFace 검출 테스트:")
    
    # 모델 로드
    model_path = "models/weights/face_detection_retinaface_resnet50.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # 테스트 이미지
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 전처리
    input_size = (640, 640)
    blob = cv2.dnn.blobFromImage(
        test_image,
        scalefactor=1.0,
        size=input_size,
        mean=[104.0, 117.0, 123.0],
        swapRB=True,
        crop=False
    )
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    
    # 검출 결과 처리
    if len(outputs) >= 3:
        boxes = outputs[0][0]      # [n_anchors, 4]
        scores = outputs[1][0]     # [n_anchors, 2]
        landmarks = outputs[2][0]  # [n_anchors, 10]
        
        print(f"박스 개수: {len(boxes)}")
        print(f"신뢰도 개수: {len(scores)}")
        print(f"특징점 개수: {len(landmarks)}")
        
        # 신뢰도 계산
        face_scores = scores[:, 1]  # 얼굴 클래스 신뢰도
        
        # 높은 신뢰도 검출 확인
        high_conf_indices = face_scores > 0.5
        print(f"높은 신뢰도 검출: {np.sum(high_conf_indices)}개")
        
        if np.any(high_conf_indices):
            print(f"최고 신뢰도: {face_scores.max():.3f}")
            print(f"평균 신뢰도: {face_scores.mean():.3f}")

if __name__ == "__main__":
    print("🔍 RetinaFace 모델 출력 분석 시작\n")
    analyze_retinaface_output()
    test_retinaface_detection() 