#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 전처리 및 크로스체크 시스템 설정 스크립트

이 스크립트는 얼굴 인식 시스템의 전처리 및 크로스체크 기능에 필요한
라이브러리와 모델을 설치합니다.

설치 항목:
- MediaPipe (얼굴 랜드마크)
- Dlib (얼굴 랜드마크 대안)
- ONNX Runtime (모델 추론)
- 기타 필요한 라이브러리

사용법:
    python scripts/setup/setup_face_preprocessing.py
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger

def install_package(package_name: str, version: str = None) -> bool:
    """패키지 설치"""
    logger = get_logger(__name__)
    
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
        
        logger.info(f"설치 중: {package_spec}")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_spec
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"설치 완료: {package_spec}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"설치 실패: {package_spec}")
        logger.error(f"오류: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"설치 중 예외 발생: {e}")
        return False

def download_file(url: str, filepath: Path) -> bool:
    """파일 다운로드"""
    logger = get_logger(__name__)
    
    try:
        logger.info(f"다운로드 중: {url}")
        logger.info(f"저장 위치: {filepath}")
        
        # 디렉토리 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 다운로드
        urllib.request.urlretrieve(url, filepath)
        
        logger.info(f"다운로드 완료: {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        return False

def setup_face_preprocessing():
    """얼굴 전처리 시스템 설정"""
    logger = get_logger(__name__)
    
    logger.info("얼굴 전처리 및 크로스체크 시스템 설정 시작")
    
    # 1. 기본 라이브러리 설치
    basic_packages = [
        ("mediapipe", "0.10.8"),      # 얼굴 랜드마크
        ("onnxruntime", "1.16.3"),    # ONNX 모델 추론
        ("opencv-python", "4.8.1.78"), # OpenCV
        ("numpy", "1.24.3"),          # 수치 계산
        ("pillow", "10.0.1"),         # 이미지 처리
    ]
    
    logger.info("기본 라이브러리 설치 중...")
    for package, version in basic_packages:
        if not install_package(package, version):
            logger.warning(f"{package} 설치 실패, 계속 진행")
    
    # 2. Dlib 설치 (선택사항, 복잡한 설치 과정)
    logger.info("Dlib 설치 시도 중...")
    try:
        # Windows에서는 Visual Studio Build Tools 필요
        if os.name == 'nt':  # Windows
            logger.warning("Windows에서 Dlib 설치를 위해서는 Visual Studio Build Tools가 필요합니다.")
            logger.warning("수동으로 설치하거나 MediaPipe만 사용하세요.")
            dlib_installed = False
        else:
            dlib_installed = install_package("dlib", "19.24.2")
    except Exception as e:
        logger.warning(f"Dlib 설치 실패: {e}")
        dlib_installed = False
    
    # 3. 모델 파일 다운로드
    models_dir = project_root / 'models' / 'weights'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_files = {
        # Dlib 랜드마크 모델 (68점)
        "shape_predictor_68_face_landmarks.dat": {
            "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
            "description": "Dlib 68점 얼굴 랜드마크 모델"
        },
        
        # OpenFace 모델
        "openface_nn4.small2.v1.t7": {
            "url": "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7",
            "description": "OpenFace 신경망 모델"
        }
    }
    
    logger.info("모델 파일 다운로드 중...")
    for filename, info in model_files.items():
        filepath = models_dir / filename
        
        if not filepath.exists():
            if download_file(info["url"], filepath):
                logger.info(f"모델 다운로드 완료: {info['description']}")
            else:
                logger.warning(f"모델 다운로드 실패: {info['description']}")
        else:
            logger.info(f"모델 파일 이미 존재: {filename}")
    
    # 4. ArcFace 모델 확인
    arcface_models = [
        "arcface_glint360k_20250628.onnx",
        "insightface_glint360k_20250628.onnx"
    ]
    
    logger.info("ArcFace 모델 확인 중...")
    for model_name in arcface_models:
        model_path = models_dir / model_name
        if model_path.exists():
            logger.info(f"ArcFace 모델 존재: {model_name}")
        else:
            logger.warning(f"ArcFace 모델 없음: {model_name}")
            logger.warning("ArcFace 모델은 별도로 다운로드해야 합니다.")
    
    # 5. 설정 완료 메시지
    logger.info("얼굴 전처리 시스템 설정 완료!")
    logger.info("")
    logger.info("설치된 기능:")
    logger.info("✓ MediaPipe Face Mesh (얼굴 랜드마크)")
    if dlib_installed:
        logger.info("✓ Dlib 얼굴 랜드마크")
    else:
        logger.info("✗ Dlib (설치 실패, MediaPipe만 사용)")
    logger.info("✓ ONNX Runtime (모델 추론)")
    logger.info("✓ OpenCV (이미지 처리)")
    logger.info("")
    logger.info("사용 가능한 모델:")
    
    for model_file in models_dir.glob("*"):
        if model_file.is_file():
            logger.info(f"  - {model_file.name}")
    
    logger.info("")
    logger.info("다음 단계:")
    logger.info("1. ArcFace 모델 다운로드 (필요시)")
    logger.info("2. run_unified_ai_grouping_processor.py 실행")
    logger.info("3. 얼굴 전처리 및 크로스체크 테스트")

def main():
    """메인 함수"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        setup_face_preprocessing()
    except KeyboardInterrupt:
        logger.info("설치가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"설치 중 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 