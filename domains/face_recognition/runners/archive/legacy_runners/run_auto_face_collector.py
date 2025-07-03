#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
자동 얼굴 수집기 (사용자 제안 흐름 1단계-1)

카메라에서 자동으로 얼굴을 감지하여 data/temp/auto_collected에 저장하는 기능을 구현합니다.
"""

import cv2
import time
import json
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from shared.vision_core.quality.face_quality_assessor import CustomFaceQualityAssessor
from common.logging import setup_logging

logger = logging.getLogger(__name__)


class AutoFaceCollector:
    """자동 얼굴 수집기"""
    
    def __init__(self):
        """초기화"""
        self.detection_service = FaceDetectionService()
        self.quality_assessor = CustomFaceQualityAssessor()
        
        # 자동 수집 저장 경로
        self.auto_collected_dir = Path("data/temp/auto_collected")
        self.auto_collected_dir.mkdir(parents=True, exist_ok=True)
        
        # 수집 설정
        self.collection_settings = {
            'min_face_size': (80, 80),           # 최소 얼굴 크기
            'collection_interval': 2.0,          # 수집 간격 (초)
            'quality_threshold': 0.6,            # 품질 임계값
            'max_collections_per_person': 20,    # 인물당 최대 수집 수
            'confidence_threshold': 0.7          # 검출 신뢰도 임계값
        }
        
        # 상태 관리
        self.last_collection_time = 0
        self.collected_faces = []
        self.current_session_id = str(uuid.uuid4())
        
    def run_auto_collection(self, camera_id: int = 0):
        """
        🤖 자동 얼굴 수집 실행
        
        카메라에서 자동으로 얼굴을 감지하고 품질 좋은 것들을 수집
        """
        print("🤖 자동 얼굴 수집기 시작")
        print("=" * 50)
        print("📋 자동 수집 설정:")
        print(f"  • 수집 간격: {self.collection_settings['collection_interval']}초")
        print(f"  • 품질 임계값: {self.collection_settings['quality_threshold']}")
        print(f"  • 최소 얼굴 크기: {self.collection_settings['min_face_size']}")
        print("📋 조작법:")
        print("  'p' → 일시정지/재개")
        print("  'r' → 수집 리셋")
        print("  'q' → 종료 및 이름 설정 단계로")
        print("=" * 50)
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"카메라 {camera_id} 열기 실패")
            return
        
        is_paused = False
        
        try:
            while True:
                if not is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("프레임 읽기 실패")
                        continue
                    
                    # 🔍 얼굴 검출
                    detections = self.detection_service.detect_faces(frame)
                    
                    # 🤖 자동 수집 처리
                    current_time = time.time()
                    if (current_time - self.last_collection_time >= self.collection_settings['collection_interval'] 
                        and len(detections) > 0):
                        
                        self._auto_collect_faces(frame, detections)
                        self.last_collection_time = current_time
                    
                    # 🎨 화면 표시
                    display_frame = self._draw_auto_collection_ui(frame.copy(), detections)
                else:
                    # 일시정지 상태에서는 마지막 프레임 유지
                    display_frame = self._draw_pause_message(frame.copy())
                
                cv2.imshow('자동 얼굴 수집기', display_frame)
                
                # ⌨️ 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    is_paused = not is_paused
                    status = "일시정지" if is_paused else "재개"
                    print(f"🔄 자동 수집 {status}")
                elif key == ord('r'):
                    self._reset_collection()
                    print("🔄 수집 데이터 리셋")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 수집된 얼굴이 있으면 이름 설정 단계로
            if self.collected_faces:
                self._proceed_to_naming_stage()
            else:
                print("❌ 수집된 얼굴이 없습니다.")
    
    def _auto_collect_faces(self, frame: np.ndarray, detections: List):
        """🤖 자동으로 얼굴 수집"""
        for detection in detections:
            # 신뢰도 확인
            if detection.confidence.value < self.collection_settings['confidence_threshold']:
                continue
            
            # 얼굴 크기 확인
            bbox = detection.bbox.to_list()
            face_w, face_h = bbox[2], bbox[3]
            min_w, min_h = self.collection_settings['min_face_size']
            
            if face_w < min_w or face_h < min_h:
                continue
            
            # 품질 평가
            quality_result = self.quality_assessor.assess_face_quality(frame, bbox)
            
            if quality_result['quality_score'] < self.collection_settings['quality_threshold']:
                continue
            
            # 얼굴 크롭 및 저장
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w]
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_id = str(uuid.uuid4())[:8]
            filename = f"auto_{timestamp}_{face_id}.jpg"
            
            # 저장
            file_path = self.auto_collected_dir / filename
            cv2.imwrite(str(file_path), face_crop)
            
            # 메타데이터 생성
            metadata = {
                'face_id': face_id,
                'session_id': self.current_session_id,
                'collection_timestamp': datetime.now().isoformat(),
                'file_path': str(file_path),
                'bbox': bbox,
                'detection_confidence': detection.confidence.value,
                'quality_assessment': quality_result,
                'collection_method': 'auto_collection'
            }
            
            # 메타데이터 저장
            metadata_path = self.auto_collected_dir / f"auto_{timestamp}_{face_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.collected_faces.append({
                'face_id': face_id,
                'file_path': file_path,
                'metadata_path': metadata_path,
                'quality_score': quality_result['quality_score']
            })
            
            print(f"✅ 자동 수집: {filename} (품질: {quality_result['quality_score']:.3f})")
            
            # 최대 수집 수 확인
            if len(self.collected_faces) >= self.collection_settings['max_collections_per_person']:
                print(f"📊 최대 수집 수({self.collection_settings['max_collections_per_person']})에 도달했습니다.")
                return
    
    def _draw_auto_collection_ui(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """🎨 자동 수집 UI 표시"""
        # 검출된 얼굴 표시
        for i, detection in enumerate(detections):
            bbox = detection.bbox.to_list()
            x, y, w, h = bbox
            
            # 얼굴 박스 (품질에 따라 색상 변경)
            confidence = detection.confidence.value
            if confidence >= self.collection_settings['confidence_threshold']:
                color = (0, 255, 0)  # 녹색 (수집 가능)
            else:
                color = (0, 255, 255)  # 노란색 (품질 부족)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도 표시
            conf_text = f'{confidence:.2f}'
            cv2.putText(frame, conf_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 상태 정보 표시
        info_lines = [
            f"Auto Collection: ON",
            f"Collected: {len(self.collected_faces)}",
            f"Session: {self.current_session_id[:8]}",
            f"Next collection in: {max(0, self.collection_settings['collection_interval'] - (time.time() - self.last_collection_time)):.1f}s"
        ]
        
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_pause_message(self, frame: np.ndarray) -> np.ndarray:
        """일시정지 메시지 표시"""
        cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 50, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return frame
    
    def _reset_collection(self):
        """수집 데이터 리셋"""
        self.collected_faces = []
        self.current_session_id = str(uuid.uuid4())
        self.last_collection_time = 0
    
    def _proceed_to_naming_stage(self):
        """이름 설정 단계로 진행"""
        print(f"\n🏷️ 이름 설정 단계 - 수집된 얼굴: {len(self.collected_faces)}개")
        print("=" * 50)
        
        # 수집된 얼굴들을 품질 순으로 정렬
        self.collected_faces.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 대표 얼굴들 표시 (상위 5개)
        print("📷 수집된 얼굴들 (품질 순):")
        display_count = min(5, len(self.collected_faces))
        
        for i in range(display_count):
            face_info = self.collected_faces[i]
            print(f"  {i+1}. {face_info['face_id']} (품질: {face_info['quality_score']:.3f})")
        
        if len(self.collected_faces) > 5:
            print(f"  ... 외 {len(self.collected_faces) - 5}개")
        
        # 인물 이름 입력
        while True:
            person_name = input("\n👤 이 인물의 이름을 입력하세요: ").strip()
            
            if person_name:
                # data/temp/face_staging로 이동
                self._move_to_face_staging(person_name)
                break
            else:
                print("❌ 이름을 입력해주세요.")
    
    def _move_to_face_staging(self, person_name: str):
        """data/temp/face_staging로 이동 (사용자 제안 흐름)"""
        face_staging_dir = Path("data/temp/face_staging")
        face_staging_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        
        for face_info in self.collected_faces:
            try:
                # 파일 이동
                old_path = face_info['file_path']
                new_filename = f"{person_name}_{old_path.stem}.jpg"
                new_path = face_staging_dir / new_filename
                
                # 이미지 파일 복사
                import shutil
                shutil.copy2(str(old_path), str(new_path))
                
                # 메타데이터 업데이트 및 이동
                with open(face_info['metadata_path'], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metadata['person_name'] = person_name
                metadata['moved_to_face_staging'] = datetime.now().isoformat()
                metadata['original_auto_collected_path'] = str(old_path)
                
                new_metadata_path = face_staging_dir / f"{person_name}_{old_path.stem}.json"
                with open(new_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                moved_count += 1
                
            except Exception as e:
                logger.error(f"파일 이동 실패 {old_path}: {str(e)}")
        
        print(f"\n✅ 자동 수집 완료!")
        print(f"   📁 이동된 얼굴: {moved_count}개")
        print(f"   📂 저장 위치: data/temp/face_staging/")
        print(f"   👤 인물 이름: {person_name}")
        print("\n🎯 다음 단계:")
        print("   1️⃣ 기존 모델로 즉시 등록하려면 → 통합 캡처 시스템 실행")
        print("   2️⃣ 훈련용 데이터로 사용하려면 → 데이터 수집 도구 실행")


def main():
    """메인 함수"""
    try:
        setup_logging()
        logger.info("Starting Auto Face Collector")
        
        collector = AutoFaceCollector()
        collector.run_auto_collection()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        logger.info("Auto Face Collector finished")


if __name__ == "__main__":
    main() 