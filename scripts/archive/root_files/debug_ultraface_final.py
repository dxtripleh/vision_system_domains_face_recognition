#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UltraFace 모델 최종 분석 및 수정 스크립트
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_ultraface_output():
    """UltraFace 모델 출력 상세 분석"""
    
    # 모델 로드
    model_path = "models/weights/face_detection_ultraface_rfb_320.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # 테스트 이미지 생성 (중앙에 얼굴 영역)
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # 중앙에 얼굴 영역 생성
    center_x, center_y = 160, 120
    face_size = 100
    test_image[center_y-face_size//2:center_y+face_size//2, 
               center_x-face_size//2:center_x+face_size//2] = np.random.randint(200, 255, (face_size, face_size, 3))
    
    # 전처리
    resized = cv2.resize(test_image, (320, 240))
    blob = resized.astype(np.float32)
    blob = (blob - 127.0) / 127.0
    blob = blob.transpose(2, 0, 1)
    blob = blob[np.newaxis, :]
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    
    print(f"🎯 UltraFace 출력 분석:")
    print(f"출력 개수: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"\n--- 출력 {i} ---")
        print(f"형태: {output.shape}")
        print(f"데이터 타입: {output.dtype}")
        print(f"값 범위: {output.min():.3f} ~ {output.max():.3f}")
        
        if output.shape == (4420, 4):
            print(f"\n🔍 UltraFace 박스 분석:")
            
            # 모든 박스의 좌표 범위 확인
            x1_values = output[:, 0]
            y1_values = output[:, 1]
            x2_values = output[:, 2]
            y2_values = output[:, 3]
            
            print(f"x1 범위: {x1_values.min():.3f} ~ {x1_values.max():.3f}")
            print(f"y1 범위: {y1_values.min():.3f} ~ {y1_values.max():.3f}")
            print(f"x2 범위: {x2_values.min():.3f} ~ {x2_values.max():.3f}")
            print(f"y2 범위: {y2_values.min():.3f} ~ {y2_values.max():.3f}")
            
            # 유효한 박스 개수 확인
            valid_boxes = 0
            for box in output:
                x1, y1, x2, y2 = box
                if (0 <= x1 <= 1 and 0 <= y1 <= 1 and 
                    0 <= x2 <= 1 and 0 <= y2 <= 1 and
                    x2 > x1 and y2 > y1):
                    valid_boxes += 1
            
            print(f"유효한 박스 개수: {valid_boxes}/{len(output)} ({valid_boxes/len(output)*100:.1f}%)")
            
            # 박스 크기 분포 확인
            box_sizes = []
            for box in output:
                x1, y1, x2, y2 = box
                if (0 <= x1 <= 1 and 0 <= y1 <= 1 and 
                    0 <= x2 <= 1 and 0 <= y2 <= 1 and
                    x2 > x1 and y2 > y1):
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    box_sizes.append(area)
            
            if box_sizes:
                box_sizes = np.array(box_sizes)
                print(f"박스 크기 분포:")
                print(f"  최소: {box_sizes.min():.4f}")
                print(f"  최대: {box_sizes.max():.4f}")
                print(f"  평균: {box_sizes.mean():.4f}")
                print(f"  중앙값: {np.median(box_sizes):.4f}")
                
                # 크기별 분포
                size_ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]
                for min_size, max_size in size_ranges:
                    count = np.sum((box_sizes >= min_size) & (box_sizes < max_size))
                    print(f"  {min_size:.2f}-{max_size:.2f}: {count}개 ({count/len(box_sizes)*100:.1f}%)")

def test_ultraface_filtering():
    """UltraFace 필터링 테스트"""
    print(f"\n🧪 UltraFace 필터링 테스트:")
    
    # 모델 로드
    model_path = "models/weights/face_detection_ultraface_rfb_320.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # 테스트 이미지
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # 전처리
    resized = cv2.resize(test_image, (320, 240))
    blob = resized.astype(np.float32)
    blob = (blob - 127.0) / 127.0
    blob = blob.transpose(2, 0, 1)
    blob = blob[np.newaxis, :]
    
    # 추론
    net.setInput(blob)
    outputs = net.forward()
    boxes = outputs[0]
    
    # 다양한 필터링 방법 테스트
    print(f"원본 박스 개수: {len(boxes)}")
    
    # 방법 1: 기본 필터링
    filtered1 = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        if (0 <= x1 <= 1 and 0 <= y1 <= 1 and 
            0 <= x2 <= 1 and 0 <= y2 <= 1 and
            x2 > x1 and y2 > y1):
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if 0.01 <= area <= 0.5:  # 적당한 크기
                confidence = min(0.9, area * 5)
                if confidence > 0.3:
                    filtered1.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
    
    print(f"방법 1 (기본): {len(filtered1)}개")
    
    # 방법 2: 매우 엄격한 필터링
    filtered2 = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        if (0 <= x1 <= 1 and 0 <= y1 <= 1 and 
            0 <= x2 <= 1 and 0 <= y2 <= 1 and
            x2 > x1 and y2 > y1):
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if 0.05 <= area <= 0.3:  # 더 엄격한 크기
                confidence = min(0.9, area * 8)
                if confidence > 0.5:  # 더 높은 임계값
                    filtered2.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
    
    print(f"방법 2 (엄격): {len(filtered2)}개")
    
    # 방법 3: 상위 N개만 선택
    all_candidates = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        if (0 <= x1 <= 1 and 0 <= y1 <= 1 and 
            0 <= x2 <= 1 and 0 <= y2 <= 1 and
            x2 > x1 and y2 > y1):
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if 0.01 <= area <= 0.5:
                confidence = min(0.9, area * 5)
                all_candidates.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
    
    # 신뢰도 순으로 정렬하고 상위 5개만
    all_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    filtered3 = all_candidates[:5]
    
    print(f"방법 3 (상위 5개): {len(filtered3)}개")
    
    return filtered3

if __name__ == "__main__":
    print("🔍 UltraFace 모델 최종 분석 시작\n")
    analyze_ultraface_output()
    test_ultraface_filtering() 