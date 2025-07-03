#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 출력 형태 디버깅 스크립트
"""

import cv2
import numpy as np
import os
from pathlib import Path

def test_model_outputs():
    """각 모델의 출력 형태를 테스트"""
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    models_dir = Path('models/weights')
    
    # 테스트할 모델들
    test_models = [
        {
            'name': 'YuNet',
            'path': models_dir / 'face_detection_yunet_2023mar.onnx',
            'type': 'yunet',
            'input_size': (320, 320),
            'mean': (0, 0, 0),
            'scale': 1.0
        },
        {
            'name': 'UltraFace',
            'path': models_dir / 'ultraface_rfb_320_robust.onnx',
            'type': 'ultraface',
            'input_size': (320, 240),
            'mean': (127, 127, 127),
            'scale': 127.0
        },
        {
            'name': 'RetinaFace',
            'path': models_dir / 'face_detection_retinaface_resnet50.onnx',
            'type': 'retinaface',
            'input_size': (640, 640),
            'mean': (104, 117, 123),
            'scale': 1.0
        },
        {
            'name': 'SCRFD',
            'path': models_dir / 'face_detection_scrfd_10g_20250628.onnx',
            'type': 'scrfd',
            'input_size': (640, 640),
            'mean': (0, 0, 0),
            'scale': 1.0
        }
    ]
    
    for model_config in test_models:
        print(f"\n{'='*50}")
        print(f"테스트 모델: {model_config['name']}")
        print(f"모델 경로: {model_config['path']}")
        
        if not model_config['path'].exists():
            print(f"❌ 모델 파일이 존재하지 않음: {model_config['path']}")
            continue
        
        try:
            # 모델 로드
            net = cv2.dnn.readNetFromONNX(str(model_config['path']))
            print(f"✅ 모델 로드 성공")
            
            # 전처리
            if model_config['type'] == 'ultraface':
                # UltraFace 특별 처리
                resized = cv2.resize(test_image, model_config['input_size'])
                blob = resized.astype(np.float32)
                blob = (blob - model_config['mean']) / model_config['scale']
                blob = blob.transpose(2, 0, 1)[np.newaxis, :]
            else:
                # 일반적인 전처리
                blob = cv2.dnn.blobFromImage(
                    test_image, 
                    model_config['scale'], 
                    model_config['input_size'], 
                    model_config['mean'], 
                    True, 
                    False
                )
            
            print(f"📊 입력 blob shape: {blob.shape}")
            
            # 추론
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"📊 출력 개수: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   출력 {i}: shape={output.shape}, dtype={output.dtype}")
                if len(output.shape) <= 2:
                    print(f"   출력 {i} 샘플값: {output[:3] if len(output) > 3 else output}")
            
        except Exception as e:
            print(f"❌ 모델 테스트 실패: {e}")
    
    print(f"\n{'='*50}")
    print("디버깅 완료")

if __name__ == "__main__":
    test_model_outputs() 