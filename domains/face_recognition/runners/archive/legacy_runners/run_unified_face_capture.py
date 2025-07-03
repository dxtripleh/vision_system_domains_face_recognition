#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 얼굴 캡처 시스템 (사용자 제안 구조)

📋 통합 데이터 흐름:
   카메라/동영상 → 얼굴 검출 → 사용자 선택 → data/temp/face_staging/
                                                        ↓
                                                    🎯 분기 선택:
                                                    1️⃣ 즉시 등록 (임베딩 → storage)
                                                    2️⃣ 훈련용 수집 (품질평가 → datasets)
"""

import cv2
import numpy as np
import time
import sys
import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.services.face_grouping_service import FaceGroupingService
from domains.face_recognition.core.entities.person import Person
from domains.face_recognition.core.entities.face import Face
from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
from domains.face_recognition.infrastructure.storage.file_storage import FileStorage
from shared.vision_core.quality.face_quality_assessor import CustomFaceQualityAssessor
from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine

logger = logging.getLogger(__name__)

class UnifiedFaceCaptureSystem:
    """통합 얼굴 캡처 시스템"""
    
    def __init__(self, camera_id: int = 0):
        """초기화"""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 직접 OpenCV 검출 엔진 초기화 (간단한 설정)
        detection_config = {
            'confidence_threshold': 0.5,
            'scale_factor': 1.1,
            'min_neighbors': 5,
            'min_size': (30, 30)
        }
        self.detector = OpenCVDetectionEngine(detection_config)
        
        # 카메라 설정
        self.camera_id = camera_id
        self.cap = None
        
        # 경로 설정
        self.face_staging_dir = project_root / 'data' / 'temp' / 'face_staging'
        self.captured_frames_dir = project_root / 'data' / 'temp' / 'captured_frames'
        self.output_dir = project_root / 'data' / 'output'
        
        # 폴더 생성
        for directory in [self.face_staging_dir, self.captured_frames_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 모드 설정
        self.is_auto_mode = False  # 기본값: 수동 모드
        
        # UI 상태
        self.show_info = True
        self.confidence_threshold = detection_config.get('confidence_threshold', 0.5)
        self.min_confidence = 0.1
        self.max_confidence = 0.9
        self.confidence_step = 0.05
        
        # 녹화 설정
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # 일시정지 설정
        self.is_paused = False
        self.paused_frame = None
        
        # 캡처된 얼굴 정보
        self.captured_faces = []
        
        # 자동 모드 설정
        self.auto_capture_buffer = []
        self.auto_capture_interval = 2.0  # 2초마다 자동 캡처
        self.last_auto_capture = 0
        self.similarity_threshold = 0.6
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # 환경 정보 저장 (간단한 기본값)
        self.detection_model = "opencv_cascade"
        self.performance_tier = "balanced"
        
        self.logger.info("통합 얼굴 캡처 시스템 초기화 완료")
        
    def start_capture(self):
        """캡처 시작"""
        print("🚀 통합 얼굴 캡처 시스템 시작")
        print("="*60)
        
        # 환경 분석 결과 출력
        self._print_environment_info()
        
        # 카메라 초기화
        if not self._initialize_camera():
            return False
        
        print("\n📹 카메라 시작됨. 키보드 명령어:")
        self._print_help()
        
        try:
            # 메인 캡처 루프
            self._capture_loop()
            
        except KeyboardInterrupt:
            print("\n⚠️  사용자가 중단했습니다.")
        except Exception as e:
            self.logger.error(f"캡처 중 오류: {str(e)}")
            print(f"❌ 오류 발생: {str(e)}")
        finally:
            self._cleanup()
        
        return True
    
    def _print_environment_info(self):
        """환경 정보 출력"""
        tier_names = {
            'high_performance': '고성능 🚀',
            'balanced': '균형형 ⚖️',
            'lightweight': '경량형 🪶'
        }
        
        print(f"🔍 환경 분석 결과:")
        print(f"   성능 등급: {tier_names.get(self.performance_tier, self.performance_tier)}")
        print(f"   선택된 검출 모델: {self.detection_model}")
        print(f"   신뢰도 임계값: {self.confidence_threshold:.2f}")
        print(f"   모드: {'🤖 자동' if self.is_auto_mode else '👤 수동'}")
    
    def _initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ 카메라 {self.camera_id} 연결 실패")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 테스트 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 카메라에서 프레임을 읽을 수 없습니다")
                return False
            
            print(f"✅ 카메라 {self.camera_id} 연결 성공 ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            self.logger.error(f"카메라 초기화 실패: {str(e)}")
            print(f"❌ 카메라 초기화 실패: {str(e)}")
            return False
    
    def _cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        print(f"\n📊 세션 요약:")
        print(f"   모드: {'자동' if self.is_auto_mode else '수동'}")
        print(f"   캡처된 얼굴: {len(getattr(self, 'captured_faces', []))}개")
        print("✅ 시스템 종료")
    
    def _print_help(self):
        """도움말 출력"""
        print("📋 사용법:")
        print("  'c' → 캡처 모드 진입")
        print("  '1' → 즉시 등록 모드")  
        print("  '2' → 훈련용 수집 모드")
        print("  's' → 현재 프레임 저장")
        print("  'q' → 종료")
        print("=" * 50)
    
    def _capture_loop(self):
        """캡처 루프"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("프레임 읽기 실패")
                continue
            
            # 🔍 얼굴 검출 (공통)
            detections = self.detector.detect(frame)
            
            # 🎨 화면 표시
            display_frame = self._draw_detections(frame.copy(), detections)
            self._draw_ui_info(display_frame)
            
            cv2.imshow('통합 얼굴 캡처 시스템', display_frame)
            
            # ⌨️ 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self._enter_capture_mode(detections, frame)
            elif key == ord('1'):
                self._set_immediate_mode()
            elif key == ord('2'):
                self._set_training_mode()
            elif key == ord('s'):
                self._save_current_frame(frame, detections)
            elif key >= ord('1') and key <= ord('9') and self.is_auto_mode:
                self._select_face_by_number(key - ord('1'), detections, frame)
            
    def _enter_capture_mode(self, detections: List, frame: np.ndarray):
        """🎮 캡처 모드 진입"""
        if len(detections) == 0:
            print("❌ 검출된 얼굴이 없습니다!")
            return
            
        self.is_auto_mode = True
        print(f"\n🎯 캡처 모드 진입! {len(detections)}개 얼굴 검출됨")
        
        # 인물 이름 입력
        self.selected_person_name = input("👤 인물 이름 입력: ").strip()
        if not self.selected_person_name:
            print("❌ 이름이 입력되지 않았습니다.")
            self.is_auto_mode = False
            return
        
        print(f"📋 '{self.selected_person_name}'으로 설정됨")
        print("🔢 1-9번 키로 원하는 얼굴을 선택하세요")
    
    def _set_immediate_mode(self):
        """1️⃣ 즉시 등록 모드 설정"""
        self.is_auto_mode = False
        print("✅ 즉시 등록 모드 활성화 (임베딩 → data/storage)")
    
    def _set_training_mode(self):
        """2️⃣ 훈련용 수집 모드 설정"""
        self.is_auto_mode = False
        print("✅ 훈련용 수집 모드 활성화 (품질평가 → datasets/raw)")
    
    def _save_current_frame(self, frame: np.ndarray, detections: List):
        """📸 현재 프레임 임시 저장"""
        timestamp = int(time.time())
        temp_path = self.face_staging_dir / f"temp_frame_{timestamp}.jpg"
        cv2.imwrite(str(temp_path), frame)
        
        # 검출된 얼굴 정보도 저장
        faces_info = []
        for i, detection in enumerate(detections):
            faces_info.append({
                'face_index': i,
                'bbox': detection.bbox.to_list(),
                'confidence': detection.confidence.value
            })
        
        info_path = self.face_staging_dir / f"temp_frame_{timestamp}.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'frame_path': str(temp_path),
                'faces': faces_info
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 프레임 임시 저장: {temp_path.name}")
    
    def _select_face_by_number(self, face_index: int, detections: List, frame: np.ndarray):
        """🔢 번호로 얼굴 선택 및 처리"""
        if face_index >= len(detections):
            print(f"❌ 얼굴 번호 {face_index + 1}이 범위를 벗어남")
            return
        
        selected_face = detections[face_index]
        print(f"✅ 얼굴 #{face_index + 1} 선택됨")
        
        # 📁 1단계: 공통 임시 저장
        temp_path = self._save_to_temp(frame, selected_face, face_index)
        
        if not self.is_auto_mode:
            print("⚠️ 처리 모드가 설정되지 않았습니다. '1' 또는 '2' 키로 모드를 선택하세요.")
            return
        
        # 🎯 2단계: 선택된 모드로 분기 처리
        if self.is_auto_mode == 'immediate':
            self._process_for_immediate_registration(temp_path, selected_face, frame)
        elif self.is_auto_mode == 'training':
            self._process_for_training_collection(temp_path, selected_face, frame)
        
        self.is_auto_mode = False
    
    def _save_to_temp(self, frame: np.ndarray, face_detection, face_index: int) -> Path:
        """📁 공통: data/temp/face_staging 에 저장"""
        timestamp = int(time.time())
        
        # 전체 프레임 저장
        frame_filename = f"{self.selected_person_name}_{timestamp}_frame_{face_index}.jpg"
        frame_path = self.face_staging_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # 얼굴 크롭 저장  
        x, y, w, h = face_detection.bbox.to_list()
        face_crop = frame[y:y+h, x:x+w]
        face_filename = f"{self.selected_person_name}_{timestamp}_face_{face_index}.jpg"
        face_path = self.face_staging_dir / face_filename
        cv2.imwrite(str(face_path), face_crop)
        
        # 메타데이터 저장
        metadata = {
            'person_name': self.selected_person_name,
            'timestamp': timestamp,
            'face_index': face_index,
            'bbox': face_detection.bbox.to_list(),
            'confidence': face_detection.confidence.value,
            'frame_path': str(frame_path),
            'face_path': str(face_path)
        }
        
        metadata_filename = f"{self.selected_person_name}_{timestamp}_meta_{face_index}.json"
        metadata_path = self.face_staging_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 공통 임시 저장 완료: {face_filename}")
        return face_path
    
    def _process_for_immediate_registration(self, face_path: Path, face_detection, frame: np.ndarray):
        """
        1️⃣ 즉시 등록 처리: 임베딩 추출 → data/storage 저장
        """
        print("🚀 1️⃣ 즉시 등록 처리 시작...")
        
        try:
            # 얼굴 이미지 로드
            face_image = cv2.imread(str(face_path))
            if face_image is None:
                self.logger.error("얼굴 이미지 로드 실패")
                return
            
            # 임베딩 추출
            embedding = self.recognition_service.extract_embedding(face_image)
            
            # Person 생성 (기존에 있으면 로드)
            person = self._get_or_create_person(self.selected_person_name)
            
            # Face 엔티티 생성
            face = Face(
                face_id=str(uuid.uuid4()),
                person_id=person.person_id,
                embedding=embedding,
                bounding_box=face_detection.bbox,
                confidence=face_detection.confidence,
                created_at=datetime.now()
            )
            
            # data/storage에 저장
            person_saved = self.storage.save_person(person)
            face_saved = self.storage.save_face(face)
            
            if person_saved and face_saved:
                # 🔗 자동 그룹핑 처리
                group_id = self.grouping_service.process_face(face)
                
                # 그룹에 사용자가 입력한 이름 자동 적용
                if self.selected_person_name:
                    self.grouping_service.set_group_name(group_id, self.selected_person_name)
                    print(f"🔗 그룹핑 완료: '{self.selected_person_name}' 그룹 (ID: {group_id[:8]})")
                
                print(f"✅ 즉시 등록 완료: {self.selected_person_name}")
                print(f"   📁 저장 위치: data/storage/")
                print(f"   🆔 Person ID: {person.person_id}")
                print(f"   🆔 Face ID: {face.face_id}")
                print(f"   🔗 Group ID: {group_id[:8]}")
                print(f"   🎯 즉시 실시간 인식 가능!")
            else:
                print("❌ 저장 실패")
                
        except Exception as e:
            self.logger.error(f"즉시 등록 처리 실패: {str(e)}")
            print(f"❌ 즉시 등록 실패: {str(e)}")
    
    def _process_for_training_collection(self, face_path: Path, face_detection, frame: np.ndarray):
        """
        2️⃣ 훈련용 수집 처리: 품질평가 → datasets/raw 저장
        """
        print("🚀 2️⃣ 훈련용 수집 처리 시작...")
        
        try:
            # 얼굴 품질 평가
            face_bbox = face_detection.bbox.to_list()
            quality_result = self.quality_assessor.assess_face_quality(frame, face_bbox)
            
            print(f"📊 품질 평가 결과:")
            print(f"   전체 점수: {quality_result['quality_score']:.3f}")
            print(f"   품질 등급: {quality_result['overall_quality']}")
            print(f"   크기 점수: {quality_result.get('size_score', 0):.3f}")
            print(f"   선명도 점수: {quality_result.get('blur_score', 0):.3f}")
            
            # 품질 기준 통과 확인
            if quality_result['overall_quality'] == 'poor':
                print("❌ 품질 기준 미달로 훈련용 수집에서 제외됨")
                return
            
            # datasets/raw 에 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 원본 이미지 저장 (full frame)
            original_dir = self.training_dir / "original_images"
            original_dir.mkdir(exist_ok=True)
            original_filename = f"{self.selected_person_name}_{timestamp}_original.jpg"
            original_path = original_dir / original_filename
            cv2.imwrite(str(original_path), frame)
            
            # 얼굴 크롭 저장
            face_crop_dir = self.training_dir / "face_crops"
            face_crop_dir.mkdir(exist_ok=True)
            face_crop_filename = f"{self.selected_person_name}_{timestamp}_face.jpg"
            face_crop_path = face_crop_dir / face_crop_filename
            
            # 임시 저장된 얼굴 이미지를 복사
            import shutil
            shutil.copy2(str(face_path), str(face_crop_path))
            
            # 메타데이터 저장
            metadata_dir = self.training_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            metadata_filename = f"{self.selected_person_name}_{timestamp}_metadata.json"
            metadata_path = metadata_dir / metadata_filename
            
            training_metadata = {
                'face_id': str(uuid.uuid4()),
                'person_name': self.selected_person_name,
                'collection_timestamp': datetime.now().isoformat(),
                'original_image_path': str(original_path),
                'face_image_path': str(face_crop_path),
                'bbox': face_bbox,
                'detection_confidence': face_detection.confidence.value,
                'quality_assessment': quality_result,
                'collection_method': 'unified_capture_system'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(training_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 훈련용 수집 완료: {self.selected_person_name}")
            print(f"   📁 저장 위치: datasets/face_recognition/raw/")
            print(f"   📊 품질 등급: {quality_result['overall_quality']}")
            print(f"   🎯 향후 모델 훈련에 활용 가능!")
            
        except Exception as e:
            self.logger.error(f"훈련용 수집 처리 실패: {str(e)}")
            print(f"❌ 훈련용 수집 실패: {str(e)}")
    
    def _get_or_create_person(self, person_name: str) -> Person:
        """기존 인물 로드 또는 새 인물 생성"""
        try:
            # 기존 인물 검색
            existing_persons = self.storage.load_all_persons()
            for person in existing_persons:
                if person.name == person_name:
                    return person
            
            # 새 인물 생성
            return Person(
                person_id=str(uuid.uuid4()),
                name=person_name,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.warning(f"인물 로드 실패, 새로 생성: {str(e)}")
            return Person(
                person_id=str(uuid.uuid4()),
                name=person_name,
                created_at=datetime.now()
            )
    
    def _draw_detections(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """🎨 검출 결과 표시"""
        for i, detection in enumerate(detections):
            bbox = detection.bbox.to_list()
            x, y, w, h = bbox
            
            # 얼굴 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 번호 표시
            cv2.putText(frame, f'{i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 신뢰도 표시
            conf_text = f'{detection.confidence.value:.2f}'
            cv2.putText(frame, conf_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _draw_ui_info(self, frame: np.ndarray):
        """🎨 UI 정보 표시"""
        info_lines = [
            f"Mode: {'🤖 자동' if self.is_auto_mode else '👤 수동'}",
            f"Capture: {'ON' if self.is_auto_mode else 'OFF'}",
            f"Person: {self.selected_person_name or 'None'}"
        ]
        
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    """메인 함수"""
    setup_logging()
    
    try:
        system = UnifiedFaceCaptureSystem()
        system.start_capture()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        print(f"❌ 오류: {str(e)}")

if __name__ == "__main__":
    main() 