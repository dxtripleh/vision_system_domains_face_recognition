#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 모델 검출 테스트
"""

import cv2
import numpy as np
from pathlib import Path

def test_simple_detection():
    """간단한 검출 테스트"""
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    models = [
        ('YuNet', 'face_detection_yunet_2023mar.onnx', 'yunet'),
        ('UltraFace', 'ultraface_rfb_320_robust.onnx', 'ultraface'),
        ('RetinaFace', 'face_detection_retinaface_resnet50.onnx', 'retinaface'),
        ('SCRFD', 'face_detection_scrfd_10g_20250628.onnx', 'scrfd')
    ]
    
    for name, model_file, model_type in models:
        print(f"\n{'='*30}")
        print(f"테스트: {name}")
        
        model_path = Path('models/weights') / model_file
        
        if not model_path.exists():
            print(f"❌ 파일 없음: {model_file}")
            continue
        
        try:
            # 모델 로드
            net = cv2.dnn.readNetFromONNX(str(model_path))
            print(f"✅ 모델 로드 성공")
            
            # 전처리
            if model_type == 'yunet':
                blob = cv2.dnn.blobFromImage(test_image, 1.0, (320, 320), (0, 0, 0), True, False)
            elif model_type == 'ultraface':
                resized = cv2.resize(test_image, (320, 240))
                blob = resized.astype(np.float32)
                blob = (blob - 127.0) / 127.0
                blob = blob.transpose(2, 0, 1)[np.newaxis, :]
            else:  # retinaface, scrfd
                blob = cv2.dnn.blobFromImage(test_image, 1.0, (640, 640), (0, 0, 0), True, False)
            
            # 추론
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"출력 shape: {[out.shape for out in outputs]}")
            
            # 검출 결과 확인
            detection_count = 0
            
            if model_type == 'yunet' and len(outputs) > 0:
                output = outputs[0]
                for detection in output:
                    if len(detection) >= 15:
                        confidence = detection[4]
                        if confidence > 0.1:  # 매우 낮은 임계값
                            detection_count += 1
            
            elif model_type == 'ultraface' and len(outputs) > 0:
                boxes = outputs[0]
                for box in boxes:
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:
                            detection_count += 1
            
            else:  # retinaface, scrfd
                for output in outputs:
                    if len(output.shape) == 2 and output.shape[1] >= 5:
                        for detection in output:
                            if len(detection) >= 5:
                                confidence = detection[4]
                                if confidence > 0.1:  # 매우 낮은 임계값
                                    detection_count += 1
            
            print(f"🔍 검출된 얼굴: {detection_count}개")
            
            if detection_count == 0:
                print("⚠️ 검출된 얼굴이 없습니다. 임계값을 더 낮춰보세요.")
            
        except Exception as e:
            print(f"❌ 오류: {str(e)}")

if __name__ == "__main__":
    test_simple_detection() 