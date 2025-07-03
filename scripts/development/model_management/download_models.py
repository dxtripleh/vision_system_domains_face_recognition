#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
비전 시스템 모델 다운로드 스크립트

RetinaFace MobileNet0.25 ONNX와 MobileFaceNet ONNX 모델을 다운로드합니다.
"""

import os
import sys
import requests
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 모델 다운로드 URL 및 정보
MODELS_TO_DOWNLOAD = {
    # 1차 선택: RetinaFace MobileNet0.25 ONNX
    "face_detection_retinaface_mobilenet025.onnx": {
        "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/face_detection/retinaface/model/retinaface_mobilenet0.25_final.onnx",
        "description": "RetinaFace MobileNet0.25 얼굴 검출 모델",
        "size_mb": 1.7,
        "priority": 1
    },
    
    # 대체 RetinaFace 모델
    "face_detection_retinaface_resnet50.onnx": {
        "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/face_detection/retinaface/model/retinaface_resnet50_final.onnx",
        "description": "RetinaFace ResNet50 얼굴 검출 모델 (고성능)",
        "size_mb": 105,
        "priority": 2
    },
    
    # 2차 선택: MobileFaceNet ONNX
    "face_recognition_mobilefacenet.onnx": {
        "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/face_recognition/mobilefacenet/model/mobilefacenet.onnx",
        "description": "MobileFaceNet 얼굴 인식 모델",
        "size_mb": 5.2,
        "priority": 3
    },
    
    # UltraFace (MobileNet 기반)
    "face_detection_ultraface_rfb_320.onnx": {
        "url": "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
        "description": "UltraFace RFB-320 초경량 얼굴 검출 모델",
        "size_mb": 1.2,
        "priority": 4
    },
    
    # SCRFD (RetinaFace 계열)
    "face_detection_scrfd_10g.onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx",
        "description": "SCRFD 10G 얼굴 검출 모델",
        "size_mb": 16.9,
        "priority": 5
    }
}

def download_file(url: str, filepath: Path, description: str = "") -> bool:
    """파일 다운로드"""
    try:
        print(f"다운로드 시작: {description}")
        logger.info(f"다운로드 시작: {url} -> {filepath}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 파일 크기 확인
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 진행률 표시
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r진행률: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n다운로드 완료: {filepath}")
        logger.info(f"다운로드 완료: {filepath}")
        return True
        
    except Exception as e:
        print(f"\n다운로드 실패: {e}")
        logger.error(f"다운로드 실패: {url} - {e}")
        return False

def main():
    """메인 함수"""
    print("=== 비전 시스템 모델 다운로드 ===")
    
    # 모델 저장 디렉토리 생성
    models_dir = project_root / "models" / "weights"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 우선순위 순으로 정렬
    sorted_models = sorted(MODELS_TO_DOWNLOAD.items(), key=lambda x: x[1]['priority'])
    
    success_count = 0
    total_count = len(sorted_models)
    
    for filename, info in sorted_models:
        filepath = models_dir / filename
        
        print(f"\n[{info['priority']}/{total_count}] {info['description']}")
        print(f"파일명: {filename}")
        print(f"크기: {info['size_mb']}MB")
        
        # 이미 파일이 존재하는지 확인
        if filepath.exists():
            print(f"이미 존재함: {filepath}")
            logger.info(f"이미 존재함: {filepath}")
            success_count += 1
            continue
        
        # 다운로드 실행
        if download_file(info['url'], filepath, info['description']):
            success_count += 1
        else:
            print(f"다운로드 실패: {filename}")
    
    print(f"\n=== 다운로드 완료 ===")
    print(f"성공: {success_count}/{total_count}")
    
    if success_count > 0:
        print("\n다운로드된 모델:")
        for filename, info in sorted_models:
            filepath = models_dir / filename
            if filepath.exists():
                print(f"  - {filename} ({info['description']})")
    
    # 우선순위 모델 확인
    priority_models = [
        "face_detection_retinaface_mobilenet025.onnx",
        "face_recognition_mobilefacenet.onnx"
    ]
    
    available_priority = []
    for model in priority_models:
        if (models_dir / model).exists():
            available_priority.append(model)
    
    if available_priority:
        print(f"\n우선순위 모델 사용 가능: {len(available_priority)}/2")
        for model in available_priority:
            print(f"  ✓ {model}")
    else:
        print("\n⚠️  우선순위 모델을 사용할 수 없습니다.")
        print("   OpenCV Haar Cascade를 백업으로 사용합니다.")

if __name__ == "__main__":
    main() 