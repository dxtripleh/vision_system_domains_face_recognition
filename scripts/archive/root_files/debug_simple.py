#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 모델 출력 분석
"""

import cv2
import numpy as np
from pathlib import Path

def test_models():
    """모든 모델의 출력 형태를 간단히 테스트"""
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    models = [
        ('YuNet', 'face_detection_yunet_2023mar.onnx', (320, 320), (0, 0, 0)),
        ('UltraFace', 'ultraface_rfb_320_robust.onnx', (320, 240), (127, 127, 127)),
        ('RetinaFace', 'face_detection_retinaface_resnet50.onnx', (640, 640), (104, 117, 123)),
        ('SCRFD', 'face_detection_scrfd_10g_20250628.onnx', (640, 640), (0, 0, 0))
    ]
    
    for name, model_file, input_size, mean in models:
        print(f"\n{'='*30}")
        print(f"테스트: {name}")
        print(f"파일: {model_file}")
        
        model_path = Path('models/weights') / model_file
        
        if not model_path.exists():
            print(f"❌ 파일 없음")
            continue
        
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            
            # 전처리
            if name == 'UltraFace':
                resized = cv2.resize(test_image, input_size)
                blob = resized.astype(np.float32)
                blob = (blob - np.array(mean)) / 127.0
                blob = blob.transpose(2, 0, 1)[np.newaxis, :]
            else:
                blob = cv2.dnn.blobFromImage(test_image, 1.0, input_size, mean, True, False)
            
            # 추론
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"✅ 출력 개수: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   출력 {i}: {output.shape}")
                if len(output.shape) == 2 and output.shape[0] > 0:
                    print(f"   첫 행: {output[0]}")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    test_models() 