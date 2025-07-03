#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
현재 RetinaFace 모델들 테스트
"""

import cv2
import numpy as np
from pathlib import Path

def test_retinaface_model(model_path: str) -> bool:
    """RetinaFace 모델 테스트"""
    try:
        print(f"\n🔍 테스트 중: {model_path}")
        
        # 모델 로드
        net = cv2.dnn.readNetFromONNX(model_path)
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 전처리 (640x640)
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
        
        print(f"  출력 개수: {len(outputs)}")
        
        # 출력 형태 분석
        for i, output in enumerate(outputs):
            print(f"  출력 {i}: {output.shape}")
        
        # 올바른 RetinaFace 출력 형태 확인
        # 예상: 3개 출력 (박스, 신뢰도, 특징점)
        if len(outputs) == 3:
            print(f"  ✅ 올바른 RetinaFace 출력 형태!")
            
            # 특징점 출력 확인
            landmarks_output = outputs[2]
            if len(landmarks_output.shape) == 3 and landmarks_output.shape[2] == 10:
                print(f"  ✅ 특징점 출력 확인: {landmarks_output.shape}")
                return True
            else:
                print(f"  ⚠️  특징점 출력 형태 이상: {landmarks_output.shape}")
        else:
            print(f"  ❌ 잘못된 출력 개수: {len(outputs)} (예상: 3)")
        
        return False
        
    except Exception as e:
        print(f"  ❌ 모델 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🎯 RetinaFace 모델 테스트 시작\n")
    
    # 현재 RetinaFace 모델들 찾기
    retinaface_models = list(Path("models/weights").glob("*retinaface*.onnx"))
    
    if not retinaface_models:
        print("❌ RetinaFace 모델을 찾을 수 없습니다.")
        return
    
    print(f"📋 발견된 RetinaFace 모델들:")
    for model in retinaface_models:
        size = model.stat().st_size / (1024 * 1024)
        print(f"  - {model.name} ({size:.1f} MB)")
    
    # 각 모델 테스트
    valid_models = []
    for model_path in retinaface_models:
        if test_retinaface_model(str(model_path)):
            valid_models.append(model_path)
    
    print(f"\n📊 테스트 결과:")
    print(f"총 모델 수: {len(retinaface_models)}")
    print(f"유효한 모델 수: {len(valid_models)}")
    
    if valid_models:
        print(f"\n✅ 사용 가능한 모델들:")
        for model in valid_models:
            print(f"  - {model.name}")
        
        # 가장 큰 모델을 기본으로 추천 (ResNet50)
        largest_model = max(valid_models, key=lambda x: x.stat().st_size)
        print(f"\n🎯 추천 모델: {largest_model.name}")
        
    else:
        print(f"\n❌ 사용 가능한 RetinaFace 모델이 없습니다.")
        print(f"UltraFace 모델을 계속 사용하세요.")

if __name__ == "__main__":
    main() 