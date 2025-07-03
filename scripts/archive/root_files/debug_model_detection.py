#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 검출 디버깅 스크립트
"""

import cv2
import numpy as np
import os
from pathlib import Path

def test_model_detection():
    """각 모델의 검출 결과를 테스트"""
    
    # 테스트 이미지 생성 (얼굴이 있는 것처럼 보이는 이미지)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 중앙에 얼굴 모양의 사각형 그리기 (테스트용)
    cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
    
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
        print(f"테스트: {model_config['name']}")
        print(f"파일: {model_config['path']}")
        
        if not model_config['path'].exists():
            print(f"❌ 모델 파일이 존재하지 않음")
            continue
        
        try:
            # 모델 로드
            net = cv2.dnn.readNetFromONNX(str(model_config['path']))
            print(f"✅ 모델 로드 성공")
            
            # 전처리
            if model_config['type'] == 'yunet':
                blob = cv2.dnn.blobFromImage(test_image, 1.0, (320, 320), (0, 0, 0), True, False)
            elif model_config['type'] == 'ultraface':
                resized = cv2.resize(test_image, (320, 240))
                blob = resized.astype(np.float32)
                blob = (blob - 127.0) / 127.0
                blob = blob.transpose(2, 0, 1)[np.newaxis, :]
            elif model_config['type'] in ['retinaface', 'scrfd']:
                blob = cv2.dnn.blobFromImage(test_image, 1.0, (640, 640), model_config['mean'], True, False)
            
            print(f"📊 입력 blob shape: {blob.shape}")
            
            # 추론
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"📊 출력 shape: {[out.shape for out in outputs]}")
            
            # 검출 결과 분석
            detections = []
            
            if model_config['type'] == 'yunet':
                if len(outputs) > 0:
                    output = outputs[0]
                    print(f"YuNet 출력 샘플: {output[0] if len(output) > 0 else 'None'}")
                    
                    for detection in output:
                        if len(detection) >= 15:
                            x, y, w, h = detection[:4]
                            confidence = detection[4]
                            print(f"  바운딩 박스: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), 신뢰도: {confidence:.2f}")
                            
                            if confidence > 0.5:  # 임계값 낮춤
                                detections.append({
                                    'bbox': [int(x * 640 / 320), int(y * 480 / 320), 
                                            int(w * 640 / 320), int(h * 480 / 320)],
                                    'confidence': float(confidence),
                                    'landmarks': [],
                                    'model_type': 'yunet'
                                })
            
            elif model_config['type'] == 'ultraface':
                if len(outputs) > 0:
                    boxes = outputs[0]
                    print(f"UltraFace 출력 샘플: {boxes[0] if len(boxes) > 0 else 'None'}")
                    
                    for box in boxes:
                        if len(box) >= 4:
                            x1, y1, x2, y2 = box[:4]
                            print(f"  바운딩 박스: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                            
                            # 좌표 변환
                            x = int(x1 * 640 / 320)
                            y = int(y1 * 480 / 240)
                            w = int((x2 - x1) * 640 / 320)
                            h = int((y2 - y1) * 480 / 240)
                            
                            if w > 0 and h > 0:
                                detections.append({
                                    'bbox': [x, y, w, h],
                                    'confidence': 0.8,
                                    'landmarks': [],
                                    'model_type': 'ultraface'
                                })
            
            elif model_config['type'] in ['retinaface', 'scrfd']:
                for output in outputs:
                    if len(output.shape) == 2 and output.shape[1] >= 5:
                        print(f"{model_config['name']} 출력 샘플: {output[0] if len(output) > 0 else 'None'}")
                        
                        for detection in output:
                            if len(detection) >= 5:
                                x, y, w, h = detection[:4]
                                confidence = detection[4]
                                print(f"  바운딩 박스: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), 신뢰도: {confidence:.2f}")
                                
                                if confidence > 0.3:  # 임계값 낮춤
                                    detections.append({
                                        'bbox': [int(x * 640 / 640), int(y * 480 / 640), 
                                                int(w * 640 / 640), int(h * 480 / 640)],
                                        'confidence': float(confidence),
                                        'landmarks': [],
                                        'model_type': model_config['type']
                                    })
            
            print(f"🔍 검출된 얼굴 수: {len(detections)}")
            for i, det in enumerate(detections):
                bbox = det['bbox']
                print(f"  얼굴 {i+1}: {bbox}, 신뢰도: {det['confidence']:.2f}, 모델: {det['model_type']}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_model_detection() 