#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 모델 출력 형태 디버깅 스크립트
"""

import cv2
import numpy as np
import os
from pathlib import Path

def debug_onnx_model(model_path: str, model_name: str):
    """ONNX 모델의 출력 형태 디버깅"""
    print(f"\n🔍 {model_name} 디버깅 시작...")
    
    try:
        # 모델 로드
        net = cv2.dnn.readNetFromONNX(model_path)
        print(f"✅ 모델 로드 성공: {model_path}")
        
        # 더미 이미지 생성 (640x480)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"📸 더미 이미지 생성: {dummy_image.shape}")
        
        # 전처리 (모델별로 다를 수 있음)
        if 'retinaface' in model_name.lower():
            blob = cv2.dnn.blobFromImage(dummy_image, 1.0, (640, 640), (104, 117, 123), True, False)
        else:
            blob = cv2.dnn.blobFromImage(dummy_image, 1.0, (640, 640), (0, 0, 0), True, False)
        
        print(f"🔄 전처리 완료: blob shape = {blob.shape}")
        
        # 추론
        net.setInput(blob)
        outputs = net.forward()
        
        print(f"🎯 추론 완료!")
        print(f"   출력 개수: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"   출력 {i}: shape = {output.shape}, dtype = {output.dtype}")
                print(f"   출력 {i}: 값 범위 = [{output.min():.3f}, {output.max():.3f}]")
                
                # 첫 몇 개 값 출력
                if output.size > 0:
                    flat_output = output.flatten()
                    print(f"   출력 {i}: 첫 10개 값 = {flat_output[:10]}")
            else:
                print(f"   출력 {i}: type = {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return False

def main():
    """메인 함수"""
    models_dir = Path('models/weights')
    
    # 테스트할 모델들
    test_models = [
        ('face_detection_retinaface_resnet50.onnx', 'RetinaFace ResNet50'),
        ('face_detection_scrfd_10g_20250628.onnx', 'SCRFD'),
        ('face_detection_yunet_2023mar.onnx', 'YuNet'),
        ('ultraface_rfb_320_robust.onnx', 'UltraFace')
    ]
    
    print("🚀 ONNX 모델 출력 형태 디버깅 시작")
    print("="*60)
    
    success_count = 0
    total_count = len(test_models)
    
    for model_file, model_name in test_models:
        model_path = models_dir / model_file
        
        if model_path.exists():
            if debug_onnx_model(str(model_path), model_name):
                success_count += 1
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
    
    print("\n" + "="*60)
    print(f"📊 결과: {success_count}/{total_count} 모델 성공")
    print("="*60)

if __name__ == "__main__":
    main() 