#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
향상된 자동 얼굴 수집기

카메라로 얼굴을 자동 검출하여 수집하고, 유사한 얼굴들을 자동 그룹핑하는 시스템입니다.
"""

import os
import sys
import cv2
import time
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine
from domains.face_recognition.infrastructure.models.arcface_recognizer import ArcFaceRecognizer
from shared.vision_core.utils.fps_counter import FPSCounter

class AutoFaceCollector:
    """향상된 자동 얼굴 수집기"""
    
    def __init__(self):
        self.logger = setup_logging()
        
        # AI 모델 초기화
        self.detector = OpenCVDetectionEngine()
        self.recognizer = ArcFaceRecognizer()
        self.fps_counter = FPSCounter()
        
        # 저장 경로 설정
        self.auto_collected_dir = project_root / 'data' / 'temp' / 'auto_collected'
        self.face_staging_dir = project_root / 'data' / 'temp' / 'face_staging'
        self.auto_collected_dir.mkdir(parents=True, exist_ok=True)
        self.face_staging_dir.mkdir(parents=True, exist_ok=True)
        
        # 수집 설정
        self.collection_interval = 2.0  # 2초마다 수집
        self.last_collection_time = 0
        self.collected_faces = []
        self.face_groups = defaultdict(list)
        self.group_counter = 0
        
        # 그룹핑 설정
        self.similarity_threshold = 0.6  # 유사도 임계값
        self.min_group_size = 3  # 최소 그룹 크기
        
        self.logger.info("자동 얼굴 수집기 초기화 완료")
    
    def start_collection(self, camera_id: int = 0, duration_minutes: int = 5):
        """자동 수집 시작"""
        # 카메라 초기화
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"카메라 {camera_id}를 열 수 없습니다")
            return False
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        self.logger.info(f"자동 수집 시작 ({duration_minutes}분간)")
        self._show_instructions(duration_minutes)
        
        try:
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("프레임을 읽을 수 없습니다")
                    break
                
                current_time = time.time()
                
                # 주기적 얼굴 수집
                if current_time - self.last_collection_time > self.collection_interval:
                    self._collect_faces_from_frame(frame)
                    self.last_collection_time = current_time
                
                # 화면 표시
                display_frame = self._create_display_frame(frame, end_time - current_time)
                cv2.imshow('Auto Face Collector', display_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    self._start_grouping_process()
                    
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 수집 완료 후 자동 그룹핑
            if self.collected_faces:
                self._start_grouping_process()
            
            self.logger.info("자동 수집 종료")
    
    def _show_instructions(self, duration_minutes):
        """사용법 안내"""
        print("\n" + "="*60)
        print("🤖 자동 얼굴 수집기")
        print("="*60)
        print(f"⏱️  수집 시간: {duration_minutes}분")
        print(f"📸 수집 간격: {self.collection_interval}초")
        print("📋 사용법:")
        print("  [G] - 즉시 그룹핑 시작")
        print("  [Q] - 수집 종료")
        print("="*60)
        print("🎯 자동으로 얼굴을 검출하고 수집합니다...")
        print()
    
    def _collect_faces_from_frame(self, frame):
        """프레임에서 얼굴 수집"""
        detections = self.detector.detect_faces(frame)
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 신뢰도가 높은 얼굴만 수집
            if confidence < 0.8:
                continue
            
            # 얼굴 영역 추출
            face_crop = self._extract_face_crop(frame, bbox)
            if face_crop is None:
                continue
            
            # 임베딩 추출
            try:
                embedding = self.recognizer.extract_embedding(face_crop)
                if embedding is None:
                    continue
                
                # 수집된 얼굴 정보 저장
                face_info = {
                    'timestamp': time.time(),
                    'bbox': bbox,
                    'confidence': confidence,
                    'embedding': embedding.tolist(),
                    'face_crop': face_crop,
                    'id': str(uuid.uuid4())
                }
                
                self.collected_faces.append(face_info)
                
                # 파일로 저장
                self._save_collected_face(face_info)
                
                print(f"📸 얼굴 수집: {len(self.collected_faces)}개 (신뢰도: {confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"임베딩 추출 실패: {str(e)}")
    
    def _extract_face_crop(self, frame, bbox):
        """얼굴 영역 추출"""
        x, y, w, h = bbox
        
        # 여유를 두고 자르기
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
            return None
        
        return face_crop
    
    def _save_collected_face(self, face_info):
        """수집된 얼굴 저장"""
        timestamp = datetime.fromtimestamp(face_info['timestamp']).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"auto_face_{timestamp}_{face_info['id'][:8]}.jpg"
        
        file_path = self.auto_collected_dir / filename
        success = cv2.imwrite(str(file_path), face_info['face_crop'])
        
        if success:
            # 메타데이터도 저장
            metadata_file = file_path.with_suffix('.json')
            metadata = {
                'id': face_info['id'],
                'timestamp': face_info['timestamp'],
                'bbox': face_info['bbox'],
                'confidence': face_info['confidence'],
                'embedding': face_info['embedding']
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        
        return success
    
    def _create_display_frame(self, frame, remaining_time):
        """화면 표시용 프레임 생성"""
        display_frame = frame.copy()
        
        # 얼굴 검출 표시
        detections = self.detector.detect_faces(frame)
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            color = (0, 255, 0) if confidence >= 0.8 else (0, 255, 255)
            cv2.rectangle(display_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                         color, 2)
            
            cv2.putText(display_frame, f"{confidence:.2f}", 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 상태 정보 표시
        self._draw_status_info(display_frame, remaining_time)
        
        # FPS 표시
        current_fps = self.fps_counter.tick()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def _draw_status_info(self, frame, remaining_time):
        """상태 정보 그리기"""
        height, width = frame.shape[:2]
        
        # 배경 박스
        cv2.rectangle(frame, (10, height - 100), (400, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 100), (400, height - 10), (255, 255, 255), 2)
        
        # 상태 텍스트
        y_offset = height - 80
        cv2.putText(frame, f"Collected: {len(self.collected_faces)} faces", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        cv2.putText(frame, f"Time left: {minutes:02d}:{seconds:02d}", 
                   (15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'G' to group now", 
                   (15, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _start_grouping_process(self):
        """그룹핑 프로세스 시작"""
        if len(self.collected_faces) < 2:
            print("❌ 그룹핑하기에 충분한 얼굴이 없습니다 (최소 2개 필요)")
            return
        
        print(f"\n🧠 {len(self.collected_faces)}개 얼굴 그룹핑 시작...")
        
        # 유사도 기반 그룹핑
        groups = self._group_faces_by_similarity()
        
        if not groups:
            print("❌ 그룹핑 결과가 없습니다")
            return
        
        print(f"✅ {len(groups)}개 그룹 생성됨")
        
        # 그룹별 처리
        for group_id, faces in groups.items():
            self._process_face_group(group_id, faces)
    
    def _group_faces_by_similarity(self):
        """유사도 기반 얼굴 그룹핑"""
        if not self.collected_faces:
            return {}
        
        # 임베딩 배열 생성
        embeddings = []
        for face in self.collected_faces:
            embeddings.append(np.array(face['embedding']))
        
        embeddings = np.array(embeddings)
        
        # 유사도 매트릭스 계산
        similarities = np.dot(embeddings, embeddings.T)
        
        # 그룹핑
        groups = {}
        used_indices = set()
        group_id = 0
        
        for i in range(len(self.collected_faces)):
            if i in used_indices:
                continue
            
            # 새 그룹 시작
            current_group = [i]
            used_indices.add(i)
            
            # 유사한 얼굴들 찾기
            for j in range(i + 1, len(self.collected_faces)):
                if j in used_indices:
                    continue
                
                if similarities[i][j] >= self.similarity_threshold:
                    current_group.append(j)
                    used_indices.add(j)
            
            # 최소 그룹 크기 확인
            if len(current_group) >= self.min_group_size:
                groups[f"group_{group_id:03d}"] = [self.collected_faces[idx] for idx in current_group]
                group_id += 1
        
        return groups
    
    def _process_face_group(self, group_id, faces):
        """그룹 처리 및 이름 입력"""
        print(f"\n📊 그룹 {group_id}: {len(faces)}개 얼굴")
        
        # 대표 얼굴 표시
        representative_face = faces[0]['face_crop']
        cv2.imshow(f'Group {group_id} - Representative Face', representative_face)
        cv2.waitKey(1)
        
        # 이름 입력 받기
        person_name = input(f"👤 그룹 {group_id}의 인물 이름을 입력하세요 (Enter=건너뛰기): ").strip()
        
        cv2.destroyWindow(f'Group {group_id} - Representative Face')
        
        if not person_name:
            print(f"⏭️  그룹 {group_id} 건너뛰기")
            return
        
        # face_staging으로 이동
        self._move_group_to_staging(group_id, person_name, faces)
    
    def _move_group_to_staging(self, group_id, person_name, faces):
        """그룹을 face_staging으로 이동"""
        # 안전한 파일명 생성
        safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_name}_{timestamp}_auto"
        staging_dir = self.face_staging_dir / folder_name
        staging_dir.mkdir(exist_ok=True)
        
        # 메타데이터 생성
        metadata = {
            'person_name': person_name,
            'safe_name': safe_name,
            'created_at': timestamp,
            'source': 'auto_collector',
            'group_id': group_id,
            'face_count': len(faces),
            'collection_session_id': str(uuid.uuid4())
        }
        
        metadata_file = staging_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 얼굴들 복사
        copied_count = 0
        for i, face in enumerate(faces):
            timestamp_str = datetime.fromtimestamp(face['timestamp']).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"face_{timestamp_str}_{i:02d}_conf{face['confidence']:.2f}.jpg"
            
            dest_path = staging_dir / filename
            success = cv2.imwrite(str(dest_path), face['face_crop'])
            
            if success:
                copied_count += 1
        
        print(f"✅ {person_name}: {copied_count}개 얼굴을 face_staging으로 이동")
        print(f"📁 저장 위치: {staging_dir}")

def main():
    """메인 함수"""
    print("🚀 자동 얼굴 수집기 시작")
    
    # 하드웨어 연결 확인
    try:
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            print("❌ 카메라가 연결되지 않았습니다. 하드웨어를 확인하세요.")
            return
        test_cap.release()
        print("✅ 카메라 연결 확인 완료")
        
    except Exception as e:
        print(f"❌ 하드웨어 확인 중 오류: {str(e)}")
        return
    
    # 수집 시간 설정
    try:
        duration = int(input("📅 수집 시간을 분 단위로 입력하세요 (기본값: 5): ") or "5")
        if duration <= 0:
            duration = 5
    except ValueError:
        duration = 5
    
    # 시스템 시작
    collector = AutoFaceCollector()
    collector.start_collection(duration_minutes=duration)

if __name__ == "__main__":
    main() 