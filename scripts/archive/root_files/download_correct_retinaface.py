#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
올바른 RetinaFace ResNet50 ONNX 모델 다운로드
"""

import os
import urllib.request
from pathlib import Path

def download_retinaface_model():
    """RetinaFace ResNet50 ONNX 모델 다운로드"""
    
    # 모델 정보
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_retinaface/face_detection_retinaface.onnx"
    model_path = "models/weights/face_detection_retinaface_resnet50_correct.onnx"
    
    # 모델 디렉토리 생성
    Path("models/weights").mkdir(parents=True, exist_ok=True)
    
    print(f"🔽 RetinaFace ResNet50 모델 다운로드 중...")
    print(f"URL: {model_url}")
    print(f"저장 경로: {model_path}")
    
    try:
        # 모델 다운로드
        urllib.request.urlretrieve(model_url, model_path)
        
        # 파일 크기 확인
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ 다운로드 완료! 파일 크기: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def download_alternative_retinaface():
    """대안 RetinaFace 모델 다운로드"""
    
    # 대안 모델 URL들
    alternative_urls = [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_retinaface/face_detection_retinaface.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_retinaface/face_detection_retinaface_resnet50.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_retinaface/face_detection_retinaface_mobilenet.onnx"
    ]
    
    for i, url in enumerate(alternative_urls):
        model_path = f"models/weights/face_detection_retinaface_alt_{i+1}.onnx"
        
        print(f"\n🔽 대안 모델 {i+1} 다운로드 중...")
        print(f"URL: {url}")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ 다운로드 완료! 파일 크기: {file_size:.1f} MB")
            return model_path
            
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            continue
    
    return None

if __name__ == "__main__":
    print("🎯 올바른 RetinaFace 모델 다운로드 시작\n")
    
    # 메인 모델 다운로드 시도
    if not download_retinaface_model():
        print("\n🔄 대안 모델 다운로드 시도...")
        alt_model = download_alternative_retinaface()
        if alt_model:
            print(f"\n✅ 대안 모델 다운로드 성공: {alt_model}")
        else:
            print("\n❌ 모든 다운로드 시도 실패")
    
    print("\n📋 다운로드된 모델 목록:")
    for model_file in Path("models/weights").glob("*retinaface*.onnx"):
        size = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size:.1f} MB)") 