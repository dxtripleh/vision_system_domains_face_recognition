#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 모델 다운로드 스크립트.

얼굴인식에 필요한 AI 모델들을 다운로드하고 설정합니다.
"""

import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "models/weights",
        "models/metadata", 
        "models/configs",
        "data/temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 생성: {directory}")

def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """파일 해시 계산"""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def download_file(url: str, output_path: str, expected_hash: Optional[str] = None) -> bool:
    """파일 다운로드"""
    try:
        print(f"⬇️ 다운로드 시작: {os.path.basename(output_path)}")
        print(f"   URL: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   진행률: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print()  # 새 줄
        
        # 해시 검증
        if expected_hash:
            actual_hash = calculate_file_hash(output_path)
            if actual_hash != expected_hash:
                print(f"❌ 해시 검증 실패:")
                print(f"   예상: {expected_hash}")
                print(f"   실제: {actual_hash}")
                os.remove(output_path)
                return False
            else:
                print(f"✅ 해시 검증 성공")
        
        print(f"✅ 다운로드 완료: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def download_opencv_models():
    """OpenCV 모델 다운로드"""
    print("\n🔍 OpenCV 모델 다운로드")
    print("=" * 50)
    
    models = {
        "haarcascade_frontalface_default.xml": {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "description": "OpenCV Haar Cascade 얼굴 검출 모델"
        },
        "haarcascade_eye.xml": {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml", 
            "description": "OpenCV Haar Cascade 눈 검출 모델"
        }
    }
    
    success_count = 0
    
    for filename, info in models.items():
        output_path = f"models/weights/{filename}"
        
        if os.path.exists(output_path):
            print(f"⏭️ 이미 존재함: {filename}")
            success_count += 1
            continue
        
        print(f"\n📥 {info['description']}")
        if download_file(info['url'], output_path):
            success_count += 1
    
    print(f"\n📊 OpenCV 모델 다운로드 완료: {success_count}/{len(models)}")
    return success_count == len(models)

def download_onnx_models():
    """ONNX 모델 다운로드 (실제 URL은 사용자가 제공해야 함)"""
    print("\n🤖 ONNX AI 모델 다운로드")
    print("=" * 50)
    
    # 실제 모델 URL들 (예시 - 실제로는 유효한 URL로 교체 필요)
    models = {
        "retinaface_r50.onnx": {
            "url": "https://example.com/models/retinaface_r50.onnx",  # 실제 URL로 교체 필요
            "description": "RetinaFace 얼굴 검출 모델 (ResNet-50)",
            "size_mb": 100,
            "hash": None  # 실제 해시값으로 교체 필요
        },
        "arcface_r100.onnx": {
            "url": "https://example.com/models/arcface_r100.onnx",  # 실제 URL로 교체 필요
            "description": "ArcFace 얼굴 인식 모델 (ResNet-100)",
            "size_mb": 200,
            "hash": None  # 실제 해시값으로 교체 필요
        }
    }
    
    print("⚠️ 주의: ONNX 모델은 실제 URL이 필요합니다.")
    print("현재는 예시 URL이므로 다운로드가 실패할 수 있습니다.")
    print("실제 모델 파일을 직접 다운로드하여 models/weights/ 폴더에 저장하세요.")
    
    success_count = 0
    
    for filename, info in models.items():
        output_path = f"models/weights/{filename}"
        
        if os.path.exists(output_path):
            print(f"⏭️ 이미 존재함: {filename}")
            success_count += 1
            continue
        
        print(f"\n📥 {info['description']} (~{info['size_mb']}MB)")
        
        # 실제 URL이 아니므로 건너뛰기
        print(f"⏭️ 건너뛰기: 실제 URL 필요 ({filename})")
        
        # 실제 다운로드를 원한다면 아래 주석 해제
        # if download_file(info['url'], output_path, info['hash']):
        #     success_count += 1
    
    print(f"\n📊 ONNX 모델 다운로드 완료: {success_count}/{len(models)}")
    return True  # 실제로는 success_count == len(models)

def create_model_configs():
    """모델 설정 파일 생성"""
    print("\n⚙️ 모델 설정 파일 생성")
    print("=" * 50)
    
    # OpenCV 모델 설정
    opencv_config = {
        "haarcascade_frontalface": {
            "model_path": "models/weights/haarcascade_frontalface_default.xml",
            "model_type": "opencv_cascade",
            "input_format": "grayscale",
            "parameters": {
                "scaleFactor": 1.1,
                "minNeighbors": 5,
                "minSize": [30, 30],
                "maxSize": [300, 300]
            }
        }
    }
    
    # ONNX 모델 설정
    onnx_config = {
        "retinaface": {
            "model_path": "models/weights/retinaface_r50.onnx",
            "model_type": "onnx",
            "input_size": [640, 640],
            "input_format": "rgb",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4
        },
        "arcface": {
            "model_path": "models/weights/arcface_r100.onnx", 
            "model_type": "onnx",
            "input_size": [112, 112],
            "input_format": "rgb",
            "embedding_size": 512,
            "normalize": True
        }
    }
    
    # 통합 설정
    model_config = {
        "default_detection_model": "haarcascade_frontalface",
        "default_recognition_model": "arcface",
        "models": {
            **opencv_config,
            **onnx_config
        }
    }
    
    # 설정 파일 저장
    import json
    config_path = "models/configs/model_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 모델 설정 파일 생성: {config_path}")
    
    # 메타데이터 파일 생성
    metadata = {
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "1.0",
        "models": {
            "opencv": {
                "haarcascade_frontalface_default.xml": {
                    "type": "face_detection",
                    "framework": "opencv",
                    "accuracy": "medium",
                    "speed": "fast"
                }
            },
            "onnx": {
                "retinaface_r50.onnx": {
                    "type": "face_detection", 
                    "framework": "onnx",
                    "accuracy": "high",
                    "speed": "medium"
                },
                "arcface_r100.onnx": {
                    "type": "face_recognition",
                    "framework": "onnx", 
                    "accuracy": "high",
                    "speed": "medium"
                }
            }
        }
    }
    
    metadata_path = "models/metadata/models_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 메타데이터 파일 생성: {metadata_path}")

def check_dependencies():
    """의존성 확인"""
    print("\n🔍 의존성 확인")
    print("=" * 50)
    
    dependencies = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "requests": "requests"
    }
    
    optional_dependencies = {
        "onnxruntime": "onnxruntime",
        "onnxruntime-gpu": "onnxruntime"  # GPU 버전
    }
    
    missing_deps = []
    
    # 필수 의존성 확인
    for package, module in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"❌ {package}: 누락")
            missing_deps.append(package)
    
    # 선택적 의존성 확인
    print("\n선택적 의존성:")
    for package, module in optional_dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"⚠️ {package}: 누락 (AI 모델 사용을 위해 권장)")
    
    if missing_deps:
        print(f"\n❌ 누락된 필수 의존성: {', '.join(missing_deps)}")
        print("다음 명령으로 설치하세요:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("\n✅ 모든 필수 의존성이 설치되어 있습니다!")
    return True

def main():
    """메인 함수"""
    print("🤖 AI 모델 다운로드 시스템")
    print("=" * 60)
    print("얼굴인식에 필요한 AI 모델들을 다운로드하고 설정합니다.")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 문제로 인해 중단됩니다.")
        return 1
    
    # 디렉토리 생성
    create_directories()
    
    # OpenCV 모델 다운로드
    opencv_success = download_opencv_models()
    
    # ONNX 모델 다운로드
    onnx_success = download_onnx_models()
    
    # 설정 파일 생성
    create_model_configs()
    
    # 결과 요약
    print("\n📊 다운로드 결과 요약")
    print("=" * 50)
    print(f"OpenCV 모델: {'✅ 성공' if opencv_success else '❌ 실패'}")
    print(f"ONNX 모델: {'⚠️ 수동 설치 필요' if onnx_success else '❌ 실패'}")
    print(f"설정 파일: ✅ 생성 완료")
    
    print("\n🎯 다음 단계:")
    print("1. ONNX 모델을 직접 다운로드하여 models/weights/ 폴더에 저장")
    print("2. python run_simple_demo.py 로 OpenCV 모델 테스트")
    print("3. ONNX 모델 설치 후 python run_face_recognition_demo.py 테스트")
    
    print("\n✅ 모델 다운로드 스크립트 완료!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 