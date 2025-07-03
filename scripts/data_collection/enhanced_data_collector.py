#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Data Collector for Continuous Learning.

지속적 학습을 위한 향상된 데이터 수집 시스템입니다.
원본 이미지, 임베딩, 메타데이터를 모두 체계적으로 저장합니다.
"""

import os
import sys
import time
import cv2
import json
import uuid
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService

logger = logging.getLogger(__name__)


@dataclass
class FaceMetadata:
    """얼굴 메타데이터"""
    face_id: str
    person_name: str
    person_id: str
    
    # 이미지 정보
    original_image_path: str
    face_image_path: str
    image_width: int
    image_height: int
    face_width: int
    face_height: int
    
    # 품질 정보
    detection_confidence: float
    face_quality_score: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    
    # 위치 정보
    bbox: List[int]  # [x, y, w, h]
    landmarks: Optional[List[List[float]]]
    head_pose: Optional[Dict[str, float]]  # yaw, pitch, roll
    
    # 수집 정보
    collection_method: str  # "camera", "upload", "video", "batch"
    collection_timestamp: str
    camera_id: Optional[int]
    video_source: Optional[str]
    frame_number: Optional[int]
    
    # 환경 정보
    lighting_condition: str  # "good", "poor", "backlight"
    image_quality: str  # "excellent", "good", "fair", "poor"
    occlusion_level: str  # "none", "partial", "heavy"
    
    # 임베딩 정보
    embedding_model: str
    embedding_version: str
    embedding_vector: List[float]


class EnhancedDataCollector:
    """향상된 데이터 수집기"""
    
    def __init__(self):
        """초기화"""
        self.detection_service = FaceDetectionService()
        self.recognition_service = FaceRecognitionService()
        
        # 저장 디렉토리 구조
        self.base_dir = Path("datasets/face_recognition")
        self.setup_directory_structure()
        
        # 데이터 품질 기준
        self.quality_thresholds = {
            'min_face_size': 80,
            'max_blur_threshold': 0.3,
            'min_brightness': 50,
            'max_brightness': 200,
            'min_contrast': 0.3
        }
        
        # 수집 통계
        self.collection_stats = {
            'total_collected': 0,
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'rejected': 0
        }
    
    def setup_directory_structure(self):
        """디렉토리 구조 설정"""
        directories = [
            self.base_dir / "raw" / "original_images",      # 원본 이미지
            self.base_dir / "raw" / "face_crops",           # 얼굴 크롭 이미지
            self.base_dir / "raw" / "metadata",             # 메타데이터
            self.base_dir / "processed" / "aligned",        # 정렬된 얼굴
            self.base_dir / "processed" / "normalized",     # 정규화된 얼굴
            self.base_dir / "augmented" / "rotated",        # 회전 증강
            self.base_dir / "augmented" / "brightness",     # 밝기 증강
            self.base_dir / "augmented" / "contrast",       # 대비 증강
            self.base_dir / "splits" / "train",             # 훈련용
            self.base_dir / "splits" / "validation",        # 검증용
            self.base_dir / "splits" / "test",              # 테스트용
            self.base_dir / "annotations" / "bounding_boxes", # 바운딩 박스
            self.base_dir / "annotations" / "landmarks",    # 랜드마크
            self.base_dir / "annotations" / "labels",       # 라벨
            self.base_dir / "quality_analysis",             # 품질 분석
            self.base_dir / "failed_cases",                 # 실패 케이스
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"디렉토리 구조 설정 완료: {self.base_dir}")
    
    def collect_from_camera(self, camera_id: int, person_name: str, 
                           target_count: int = 50) -> List[FaceMetadata]:
        """카메라에서 체계적 데이터 수집"""
        logger.info(f"카메라 {camera_id}에서 {person_name}의 얼굴 {target_count}개 수집 시작")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"카메라 {camera_id} 열기 실패")
        
        collected_faces = []
        frame_count = 0
        
        print(f"\n📷 {person_name}의 얼굴 수집을 시작합니다")
        print("   - 다양한 각도로 얼굴을 수집합니다")
        print("   - 's' 키: 현재 프레임 저장")
        print("   - 'q' 키: 종료")
        
        while len(collected_faces) < target_count:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # 간단한 얼굴 검출 (OpenCV 기본)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # 얼굴 표시
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 진행률 표시
            progress_text = f"Collected: {len(collected_faces)}/{target_count}"
            cv2.putText(frame, progress_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(faces) > 0:
                # 가장 큰 얼굴 저장
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # 메타데이터 생성
                metadata = self._create_simple_metadata(
                    frame, largest_face, person_name,
                    collection_method="camera",
                    camera_id=camera_id,
                    frame_number=frame_count
                )
                
                # 이미지 저장
                if self._save_face_data(frame, largest_face, metadata):
                    collected_faces.append(metadata)
                    print(f"   ✅ 수집: {len(collected_faces)}/{target_count}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"카메라 수집 완료: {len(collected_faces)}개 얼굴")
        self._update_collection_stats(collected_faces)
        
        return collected_faces
    
    def collect_from_video(self, video_path: str, person_name: str,
                          frame_skip: int = 30) -> List[FaceMetadata]:
        """동영상에서 체계적 데이터 수집"""
        logger.info(f"동영상 {video_path}에서 {person_name}의 얼굴 수집 시작")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"동영상 {video_path} 열기 실패")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        collected_faces = []
        frame_number = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # 프레임 건너뛰기로 다양성 확보
            if frame_number % frame_skip != 0:
                continue
            
            processed_frames += 1
            
            # 얼굴 검출
            detection_result = self.detection_service.detect_faces(frame)
            
            for face in detection_result.faces:
                # 품질 검사
                quality_info = self._assess_face_quality(frame, face)
                
                if quality_info['overall_quality'] in ['excellent', 'good', 'fair']:
                    # 메타데이터 생성
                    metadata = self._create_face_metadata(
                        frame, face, person_name,
                        collection_method="video",
                        video_source=video_path,
                        frame_number=frame_number,
                        quality_info=quality_info
                    )
                    
                    # 이미지 저장
                    if self._save_face_data(frame, face, metadata):
                        collected_faces.append(metadata)
            
            # 진행률 표시
            progress = processed_frames / (total_frames // frame_skip) * 100
            if processed_frames % 10 == 0:
                print(f"진행률: {progress:.1f}% - 수집된 얼굴: {len(collected_faces)}")
        
        cap.release()
        
        logger.info(f"동영상 수집 완료: {len(collected_faces)}개 얼굴")
        self._update_collection_stats(collected_faces)
        
        return collected_faces
    
    def _assess_face_quality(self, image: np.ndarray, face) -> Dict[str, Any]:
        """얼굴 품질 평가"""
        x, y, w, h = face.bbox
        face_crop = image[y:y+h, x:x+w]
        
        # 크기 검사
        size_score = min(w, h) / self.quality_thresholds['min_face_size']
        
        # 블러 검사 (Laplacian variance)
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        blur_normalized = min(blur_score / 1000, 1.0)
        
        # 밝기 검사
        brightness = np.mean(gray_face)
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        
        # 대비 검사
        contrast = np.std(gray_face) / 127.5
        contrast_score = min(contrast, 1.0)
        
        # 전체 품질 점수 계산
        quality_score = (
            size_score * 0.3 +
            blur_normalized * 0.3 +
            brightness_score * 0.2 +
            contrast_score * 0.2
        )
        
        # 품질 등급 결정
        if quality_score >= 0.8:
            overall_quality = "excellent"
        elif quality_score >= 0.6:
            overall_quality = "good"
        elif quality_score >= 0.4:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            'size_score': size_score,
            'blur_score': blur_normalized,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'quality_score': quality_score,
            'overall_quality': overall_quality,
            'brightness': brightness,
            'contrast': contrast * 127.5
        }
    
    def _create_face_metadata(self, image: np.ndarray, face, person_name: str,
                             collection_method: str, **kwargs) -> FaceMetadata:
        """얼굴 메타데이터 생성"""
        face_id = str(uuid.uuid4())
        person_id = str(uuid.uuid4())  # 실제로는 기존 person_id 사용
        
        x, y, w, h = face.bbox
        quality_info = kwargs.get('quality_info', {})
        
        # 임베딩 추출
        face_crop = image[y:y+h, x:x+w]
        embedding = self.recognition_service.extract_embedding(face_crop)
        
        metadata = FaceMetadata(
            face_id=face_id,
            person_name=person_name,
            person_id=person_id,
            
            # 이미지 정보
            original_image_path="",  # 저장 후 설정
            face_image_path="",      # 저장 후 설정
            image_width=image.shape[1],
            image_height=image.shape[0],
            face_width=w,
            face_height=h,
            
            # 품질 정보
            detection_confidence=face.confidence,
            face_quality_score=quality_info.get('quality_score', 0.0),
            blur_score=quality_info.get('blur_score', 0.0),
            brightness_score=quality_info.get('brightness_score', 0.0),
            contrast_score=quality_info.get('contrast_score', 0.0),
            
            # 위치 정보
            bbox=[x, y, w, h],
            landmarks=None,
            head_pose=None,
            
            # 수집 정보
            collection_method=collection_method,
            collection_timestamp=datetime.now().isoformat(),
            camera_id=kwargs.get('camera_id'),
            video_source=kwargs.get('video_source'),
            frame_number=kwargs.get('frame_number'),
            
            # 환경 정보
            lighting_condition=self._assess_lighting(quality_info),
            image_quality=quality_info.get('overall_quality', 'unknown'),
            occlusion_level="none",  # 추후 구현
            
            # 임베딩 정보
            embedding_model="arcface",
            embedding_version="1.0",
            embedding_vector=embedding.vector.tolist()
        )
        
        return metadata
    
    def _save_face_data(self, image: np.ndarray, face, metadata: FaceMetadata) -> bool:
        """얼굴 데이터 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 원본 이미지 저장
            original_filename = f"{metadata.person_name}_{timestamp}_original_{metadata.face_id[:8]}.jpg"
            original_path = self.base_dir / "raw" / "original_images" / original_filename
            cv2.imwrite(str(original_path), image)
            
            # 얼굴 크롭 저장
            x, y, w, h = metadata.bbox
            face_crop = image[y:y+h, x:x+w]
            face_filename = f"{metadata.person_name}_{timestamp}_face_{metadata.face_id[:8]}.jpg"
            face_path = self.base_dir / "raw" / "face_crops" / face_filename
            cv2.imwrite(str(face_path), face_crop)
            
            # 메타데이터 업데이트
            metadata.original_image_path = str(original_path)
            metadata.face_image_path = str(face_path)
            
            # 메타데이터 JSON 저장
            metadata_filename = f"{metadata.person_name}_{timestamp}_metadata_{metadata.face_id[:8]}.json"
            metadata_path = self.base_dir / "raw" / "metadata" / metadata_filename
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"얼굴 데이터 저장 완료: {face_filename}")
            return True
            
        except Exception as e:
            logger.error(f"얼굴 데이터 저장 실패: {str(e)}")
            return False
    
    def _assess_lighting(self, quality_info: Dict) -> str:
        """조명 상태 평가"""
        brightness = quality_info.get('brightness', 127.5)
        
        if brightness < 80:
            return "poor"
        elif brightness > 180:
            return "backlight"
        else:
            return "good"
    
    def _draw_collection_preview(self, frame: np.ndarray, faces: List, 
                               collected: int, target: int):
        """수집 미리보기 표시"""
        for face in faces:
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 진행률 표시
        progress_text = f"Collected: {collected}/{target}"
        cv2.putText(frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _update_collection_stats(self, collected_faces: List[FaceMetadata]):
        """수집 통계 업데이트"""
        for face in collected_faces:
            self.collection_stats['total_collected'] += 1
            
            if face.image_quality == 'excellent':
                self.collection_stats['high_quality'] += 1
            elif face.image_quality == 'good':
                self.collection_stats['medium_quality'] += 1
            elif face.image_quality == 'fair':
                self.collection_stats['low_quality'] += 1
            else:
                self.collection_stats['rejected'] += 1
    
    def export_training_dataset(self, train_ratio: float = 0.7, 
                               val_ratio: float = 0.2) -> Dict[str, str]:
        """훈련용 데이터셋 export"""
        metadata_dir = self.base_dir / "raw" / "metadata"
        
        # 모든 메타데이터 로드
        all_metadata = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
        
        # 인물별 그룹화
        person_groups = {}
        for metadata in all_metadata:
            person_name = metadata['person_name']
            if person_name not in person_groups:
                person_groups[person_name] = []
            person_groups[person_name].append(metadata)
        
        # 분할 수행
        splits = {'train': [], 'validation': [], 'test': []}
        
        for person_name, faces in person_groups.items():
            n_faces = len(faces)
            n_train = int(n_faces * train_ratio)
            n_val = int(n_faces * val_ratio)
            
            # 랜덤 셔플 후 분할
            import random
            random.shuffle(faces)
            
            splits['train'].extend(faces[:n_train])
            splits['validation'].extend(faces[n_train:n_train + n_val])
            splits['test'].extend(faces[n_train + n_val:])
        
        # 분할 결과 저장
        split_info = {}
        for split_name, split_data in splits.items():
            split_file = self.base_dir / "splits" / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            split_info[split_name] = str(split_file)
            logger.info(f"{split_name.capitalize()} split: {len(split_data)} samples")
        
        return split_info
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """수집 요약 정보"""
        # 실제 저장된 파일 수 계산
        metadata_dir = self.base_dir / "raw" / "metadata"
        stored_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0
        
        return {
            'stored_files': stored_count,
            'data_locations': {
                'original_images': str(self.base_dir / "raw" / "original_images"),
                'face_crops': str(self.base_dir / "raw" / "face_crops"),
                'metadata': str(self.base_dir / "raw" / "metadata")
            },
            'directory_structure': str(self.base_dir)
        }


def main():
    """메인 함수"""
    setup_logging()
    
    collector = EnhancedDataCollector()
    
    print("🎯 향상된 데이터 수집 시스템")
    print("=" * 50)
    print("1. 카메라에서 수집")
    print("2. 동영상에서 수집")  
    print("3. 수집 요약 보기")
    
    choice = input("선택하세요 (1-3): ")
    
    if choice == '1':
        person_name = input("인물 이름: ")
        target_count = int(input("목표 수집 개수 (기본 50): ") or "50")
        
        faces = collector.collect_from_camera(0, person_name, target_count)
        print(f"✅ {len(faces)}개 얼굴 수집 완료")
        
    elif choice == '2':
        video_path = input("동영상 경로: ")
        person_name = input("인물 이름: ")
        
        faces = collector.collect_from_video(video_path, person_name)
        print(f"✅ {len(faces)}개 얼굴 수집 완료")
        
    elif choice == '3':
        summary = collector.get_collection_summary()
        print(f"저장된 파일 수: {summary['stored_files']}")
        print(f"데이터 저장소: {summary['directory_structure']}")


if __name__ == "__main__":
    main() 