#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
입력 폴더에서 얼굴 검출 및 저장 (OpenCV 전용 버전)

이 스크립트는 captured와 uploads 폴더의 모든 이미지에서 얼굴을 검출하고,
detected_faces 폴더에 저장합니다. OpenCV Haar Cascade만 사용합니다.
"""

import os
import sys
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Any
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine
from shared.vision_core.naming import UniversalNamingSystem, UniversalMetadata

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 경로 설정
CAPTURED_DIR = Path("data/domains/face_recognition/raw_input/captured")
UPLOADS_DIR = Path("data/domains/face_recognition/raw_input/uploads")
DETECTED_DIR = Path("data/domains/face_recognition/detected_faces")

def get_timestamp() -> str:
    """현재 타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def load_face_detector():
    """얼굴 검출기 로딩 (OpenCV 전용)"""
    try:
        config = {
            'cascade_path': 'models/weights/haarcascade_frontalface_default.xml',
            'confidence_threshold': 0.3,
            'scale_factor': 1.05,
            'min_neighbors': 2,
            'min_size': (20, 20)
        }
        detector = OpenCVDetectionEngine(config)
        print("[OpenCV 전용] OpenCV Haar Cascade 사용")
        logger.info("[OpenCV 전용] OpenCV Haar Cascade 사용")
        return detector
    except Exception as e:
        print(f"[OpenCV 전용] OpenCV 로딩 실패: {e}")
        logger.error(f"[OpenCV 전용] OpenCV 로딩 실패: {e}")
        return None

def get_all_input_files() -> List[Path]:
    """모든 입력 파일 수집"""
    files = []
    
    # captured 폴더에서 모든 이미지 파일 수집 (하위 폴더 포함)
    if CAPTURED_DIR.exists():
        for file_path in CAPTURED_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                files.append(file_path)
    
    # uploads 폴더에서 모든 이미지 파일 수집
    if UPLOADS_DIR.exists():
        for file_path in UPLOADS_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                files.append(file_path)
    
    return files

def get_existing_detections() -> Dict[str, List[Path]]:
    """기존 검출 결과 수집"""
    existing_detections = {}
    
    if DETECTED_DIR.exists():
        for date_dir in DETECTED_DIR.iterdir():
            if date_dir.is_dir():
                date_str = date_dir.name
                existing_detections[date_str] = []
                
                for file_path in date_dir.glob("*.jpg"):
                    existing_detections[date_str].append(file_path)
    
    return existing_detections

def should_process_file(file_path: Path, existing_detections: Dict[str, List[Path]]) -> bool:
    """파일 처리 여부 결정"""
    # 모든 파일을 재처리 (기존 결과 무시)
    return True

def save_detected_face(face_img, bbox, confidence, source_path, sequence):
    """검출된 얼굴 저장"""
    try:
        # 네이밍 시스템 사용
        naming_system = UniversalNamingSystem()
        
        # 메타데이터 생성
        metadata = UniversalMetadata.create_detection_metadata(
            domain="face_recognition",
            object_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sequence:02d}",
            confidence=confidence,
            bbox=bbox,
            original_capture=source_path.name
        )
        
        # 파일명 생성
        filename = naming_system.generate_filename(
            domain="face_recognition",
            object_type="face",
            sequence=sequence,
            source_name=source_path.stem,
            confidence=int(confidence * 100)
        )
        
        # 저장 경로 생성
        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = DETECTED_DIR / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 저장
        image_path = save_dir / f"{filename}.jpg"
        cv2.imwrite(str(image_path), face_img)
        
        # 메타데이터 저장
        metadata_path = save_dir / f"{filename}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"검출 얼굴 저장: {image_path}")
        return True
        
    except Exception as e:
        logger.error(f"얼굴 저장 실패: {e}")
        return False

def process_image(detector, img_path: Path) -> int:
    """이미지 처리"""
    try:
        print(f"처리 중: {img_path.name}")
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"이미지 로드 실패: {img_path}")
            return 0
        
        # 얼굴 검출
        detections = detector.detect(img)
        
        if not detections:
            logger.info(f"{img_path.name}: 얼굴 없음")
            return 0
        
        # 검출된 얼굴 저장
        saved_count = 0
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # 얼굴 영역 추출
            face_img = img[y:y+h, x:x+w]
            
            # 저장
            if save_detected_face(face_img, detection['bbox'], confidence, img_path, i+1):
                saved_count += 1
        
        logger.info(f"{img_path.name}: {len(detections)}개 얼굴 검출")
        return saved_count
        
    except Exception as e:
        logger.error(f"이미지 처리 실패 {img_path}: {e}")
        return 0

def main():
    """메인 함수"""
    print("=== 얼굴 검출 시작 ===")
    logger.info("=== 얼굴 검출 시작 ===")
    
    # 얼굴 검출기 로딩
    detector = load_face_detector()
    if detector is None:
        print("얼굴 검출기 로딩 실패")
        logger.error("얼굴 검출기 로딩 실패")
        return
    
    # 입력 파일 수집
    input_files = get_all_input_files()
    print(f"총 입력 파일: {len(input_files)}개")
    logger.info(f"총 입력 파일: {len(input_files)}개")
    
    # 기존 검출 결과 확인
    existing_detections = get_existing_detections()
    print(f"기존 검출 결과: {len(existing_detections)}개 파일")
    logger.info(f"기존 검출 결과: {len(existing_detections)}개 파일")
    
    # 처리할 파일 결정
    files_to_process = []
    for file_path in input_files:
        if should_process_file(file_path, existing_detections):
            files_to_process.append(file_path)
            logger.info(f"처리 대상: {file_path.name}")
    
    # 이미지 처리
    total_detections = 0
    for file_path in files_to_process:
        detections = process_image(detector, file_path)
        total_detections += detections
    
    print("=== 얼굴 검출 완료 ===")
    print(f"처리된 파일: {len(files_to_process)}개")
    print(f"검출된 얼굴: {total_detections}개")
    logger.info("=== 얼굴 검출 완료 ===")
    logger.info(f"처리된 파일: {len(files_to_process)}개")
    logger.info(f"검출된 얼굴: {total_detections}개")

if __name__ == "__main__":
    main() 