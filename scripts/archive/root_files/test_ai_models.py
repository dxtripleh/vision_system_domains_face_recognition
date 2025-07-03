#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 모델 로딩 테스트 스크립트
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_onnx_model_loading():
    """ONNX 모델 로딩 테스트"""
    print("🔍 ONNX 모델 로딩 테스트")
    print("=" * 50)
    
    models = {
        "RetinaFace": "models/weights/face_detection_retinaface_mnet025_20250628.onnx",
        "SCRFD": "models/weights/face_detection_scrfd_10g_20250628.onnx",
        "ArcFace R50": "models/weights/face_recognition_arcface_r50_20250628.onnx",
        "ArcFace R100": "models/weights/face_recognition_arcface_r100_20250628.onnx",
        "MobileFaceNet": "models/weights/face_recognition_mobilefacenet_20250628.onnx"
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n📥 {model_name} 테스트 중...")
        
        if not os.path.exists(model_path):
            print(f"❌ 파일 없음: {model_path}")
            results[model_name] = False
            continue
        
        try:
            # ONNX 모델 로드
            net = cv2.dnn.readNetFromONNX(model_path)
            
            # 모델 정보 확인
            layer_names = net.getLayerNames()
            # input_shape는 일부 OpenCV 버전에서 지원하지 않음
            try:
                input_shape = net.getInputShape()
            except:
                input_shape = "Unknown"
            
            print(f"✅ 로드 성공")
            print(f"   레이어 수: {len(layer_names)}")
            print(f"   입력 형태: {input_shape}")
            print(f"   첫 5개 레이어: {layer_names[:5]}")
            
            results[model_name] = True
            
        except Exception as e:
            print(f"❌ 로드 실패: {str(e)}")
            results[model_name] = False
    
    return results

def test_model_inference():
    """모델 추론 테스트"""
    print("\n🔍 모델 추론 테스트")
    print("=" * 50)
    
    # 테스트 이미지 생성 (더미 데이터)
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # SCRFD 모델로 추론 테스트
    try:
        print("📥 SCRFD 모델 추론 테스트...")
        model_path = "models/weights/face_detection_scrfd_10g_20250628.onnx"
        
        if os.path.exists(model_path):
            net = cv2.dnn.readNetFromONNX(model_path)
            
            # 이미지 전처리
            blob = cv2.dnn.blobFromImage(test_image, 1.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
            
            # 추론 실행
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"✅ 추론 성공")
            print(f"   출력 형태: {[out.shape for out in outputs]}")
            print(f"   출력 개수: {len(outputs)}")
            
            return True
        else:
            print("❌ 모델 파일 없음")
            return False
            
    except Exception as e:
        print(f"❌ 추론 실패: {str(e)}")
        return False

def test_face_recognition_models():
    """얼굴 인식 모델 테스트"""
    print("\n🔍 얼굴 인식 모델 테스트")
    print("=" * 50)
    
    # ArcFace R50 모델 테스트
    try:
        print("📥 ArcFace R50 모델 테스트...")
        model_path = "models/weights/face_recognition_arcface_r50_20250628.onnx"
        
        if os.path.exists(model_path):
            net = cv2.dnn.readNetFromONNX(model_path)
            
            # 테스트 이미지 (112x112 얼굴 이미지)
            test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # 이미지 전처리
            blob = cv2.dnn.blobFromImage(test_face, 1.0/255.0, (112, 112), (0, 0, 0), swapRB=True)
            
            # 추론 실행
            net.setInput(blob)
            embedding = net.forward()
            
            print(f"✅ 임베딩 추출 성공")
            print(f"   임베딩 형태: {embedding.shape}")
            print(f"   임베딩 차원: {embedding.size}")
            
            return True
        else:
            print("❌ 모델 파일 없음")
            return False
            
    except Exception as e:
        print(f"❌ 임베딩 추출 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    print("🚀 AI 모델 테스트 시작")
    print("=" * 60)
    
    # 1. ONNX 모델 로딩 테스트
    loading_results = test_onnx_model_loading()
    
    # 2. 모델 추론 테스트
    inference_success = test_model_inference()
    
    # 3. 얼굴 인식 모델 테스트
    recognition_success = test_face_recognition_models()
    
    # 결과 요약
    print("\n📊 테스트 결과 요약")
    print("=" * 50)
    
    print("ONNX 모델 로딩:")
    for model_name, success in loading_results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {model_name}: {status}")
    
    print(f"\n모델 추론: {'✅ 성공' if inference_success else '❌ 실패'}")
    print(f"얼굴 인식: {'✅ 성공' if recognition_success else '❌ 실패'}")
    
    # 성공률 계산
    total_models = len(loading_results)
    successful_models = sum(loading_results.values())
    success_rate = (successful_models / total_models) * 100
    
    print(f"\n📈 전체 성공률: {success_rate:.1f}% ({successful_models}/{total_models})")
    
    if success_rate >= 80:
        print("🎉 대부분의 AI 모델이 정상적으로 작동합니다!")
    elif success_rate >= 50:
        print("⚠️ 일부 AI 모델에 문제가 있습니다.")
    else:
        print("❌ 대부분의 AI 모델에 문제가 있습니다.")

if __name__ == "__main__":
    main() 