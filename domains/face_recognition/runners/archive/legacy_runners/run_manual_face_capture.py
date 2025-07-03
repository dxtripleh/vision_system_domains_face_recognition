#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
수동 얼굴 캡처 시스템

카메라로 얼굴을 캡처하고 이름을 지정하여 face_staging에 저장합니다.
키보드 인터페이스:
- i: 정보 표시 토글
- +/-: 신뢰도 임계값 조절
- s: 현재 프레임 저장
- c: 얼굴 캡처 + 이름 지정
- r: 녹화 시작/중지
- p: 일시정지/재생
- h: 도움말 표시
- q: 종료
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
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.core.services.environment_analyzer import EnvironmentAnalyzer
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine

class ManualFaceCaptureSystem:
    """수동 얼굴 캡처 시스템"""
    
    def __init__(self, camera_id: int = 0):
        import logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 환경 분석 및 최적 모델 선택
        self.env_analyzer = EnvironmentAnalyzer()
        self.analysis_result = self.env_analyzer.analyze_environment()
        
        # 최적 검출 설정
        detection_config = self.analysis_result['optimal_models']['detection']['config']
        self.detector = OpenCVDetectionEngine(detection_config)
        
        # 카메라 설정
        self.camera_id = camera_id
        self.cap = None
        
        # 경로 설정
        self.face_staging_dir = project_root / 'data' / 'temp' / 'face_staging'
        self.output_dir = project_root / 'data' / 'output'
        
        # 폴더 생성
        for directory in [self.face_staging_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
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
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        self.logger.info("수동 얼굴 캡처 시스템 초기화 완료")
    
    def start_capture(self):
        """캡처 시작"""
        print("🚀 수동 얼굴 캡처 시스템 시작")
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
        
        tier = self.analysis_result['performance_tier']
        detection_model = self.analysis_result['optimal_models']['detection']['model']
        
        print(f"🔍 환경 분석 결과:")
        print(f"   성능 등급: {tier_names.get(tier, tier)}")
        print(f"   선택된 검출 모델: {detection_model}")
        print(f"   신뢰도 임계값: {self.confidence_threshold:.2f}")
    
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
    
    def _capture_loop(self):
        """메인 캡처 루프"""
        while True:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    break
                
                # 현재 프레임 저장 (일시정지용)
                self.paused_frame = frame.copy()
            else:
                # 일시정지 상태에서는 저장된 프레임 사용
                frame = self.paused_frame.copy()
            
            # 얼굴 검출
            detections = self.detector.detect(frame)
            
            # 검출 결과 시각화
            display_frame = self._visualize_detections(frame, detections)
            
            # 정보 오버레이
            if self.show_info:
                display_frame = self._draw_info_overlay(display_frame, detections)
            
            # 녹화 중이면 프레임 저장
            if self.is_recording and self.video_writer:
                self.video_writer.write(display_frame)
            
            # 화면 표시
            cv2.imshow('Manual Face Capture - Press h for help', display_frame)
            
            # 키보드 입력 처리
            action = self._handle_keyboard_input()
            if action == 'quit':
                break
            elif action == 'capture_face':
                self._capture_face_with_name(frame, detections)
            elif action == 'save_frame':
                self._save_current_frame(display_frame)
            elif action == 'toggle_record':
                self._toggle_recording(display_frame)
            elif action == 'toggle_pause':
                self._toggle_pause()
            elif action == 'show_help':
                self._print_help()
            elif action == 'toggle_info':
                self.show_info = not self.show_info
            elif action == 'increase_threshold':
                self._adjust_confidence_threshold(0.05)
            elif action == 'decrease_threshold':
                self._adjust_confidence_threshold(-0.05)
            
            # FPS 계산
            self._update_fps()
    
    def _visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """검출 결과 시각화"""
        display_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 신뢰도 필터링
            if confidence < self.confidence_threshold:
                continue
            
            x, y, w, h = bbox
            
            # 바운딩 박스 그리기
            color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255)
            thickness = 2 if confidence >= 0.7 else 1
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # 신뢰도 텍스트
            text = f"{confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(display_frame, (x, y - text_size[1] - 5), 
                         (x + text_size[0], y), color, -1)
            cv2.putText(display_frame, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return display_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """정보 오버레이 그리기"""
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 정보 텍스트
        y_offset = 30
        line_height = 20
        
        # 기본 정보
        texts = [
            f"FPS: {self.current_fps:.1f}",
            f"Faces: {len([d for d in detections if d['confidence'] >= self.confidence_threshold])}",
            f"Confidence: {self.confidence_threshold:.2f}",
            f"Captured: {len(self.captured_faces)}",
        ]
        
        # 상태 정보
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            texts.append(f"🔴 REC {elapsed:.0f}s")
        
        if self.is_paused:
            texts.append("⏸️ PAUSED")
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _handle_keyboard_input(self) -> Optional[str]:
        """키보드 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('i'):
            return 'toggle_info'
        elif key == ord('s'):
            return 'save_frame'
        elif key == ord('c'):
            return 'capture_face'
        elif key == ord('r'):
            return 'toggle_record'
        elif key == ord('p'):
            return 'toggle_pause'
        elif key == ord('h'):
            return 'show_help'
        elif key == ord('+') or key == ord('='):
            return 'increase_threshold'
        elif key == ord('-'):
            return 'decrease_threshold'
        
        return None
    
    def _capture_face_with_name(self, frame: np.ndarray, detections: List[Dict]):
        """얼굴 캡처 및 이름 지정"""
        # 신뢰도 기준으로 필터링
        valid_detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        if not valid_detections:
            print("⚠️  캡처할 얼굴이 없습니다. (신뢰도 기준 미달)")
            return
        
        print(f"\n👤 {len(valid_detections)}개 얼굴 발견됨")
        
        # 이름 입력 창 표시
        person_name = self._get_person_name_input(frame, valid_detections)
        
        if not person_name:
            print("⏭️  캡처 취소됨")
            return
        
        # 얼굴들 저장
        saved_count = self._save_faces_to_staging(frame, valid_detections, person_name)
        
        if saved_count > 0:
            print(f"✅ {person_name}: {saved_count}개 얼굴 캡처 완료")
            self.captured_faces.extend(valid_detections)
        else:
            print("❌ 얼굴 저장 실패")
    
    def _get_person_name_input(self, frame: np.ndarray, detections: List[Dict]) -> Optional[str]:
        """이름 입력 받기"""
        # 얼굴 영역들을 하이라이트한 프레임 생성
        highlight_frame = frame.copy()
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            cv2.rectangle(highlight_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(highlight_frame, f"Conf: {detection['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 입력 안내 텍스트 추가
        cv2.putText(highlight_frame, "Enter person name in console", 
                   (20, highlight_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Face Capture - Enter name in console', highlight_frame)
        cv2.waitKey(1)
        
        # 콘솔에서 이름 입력 받기
        try:
            person_name = input(f"👤 인물 이름을 입력하세요 (Enter=취소): ").strip()
            
            # 입력 창 닫기
            cv2.destroyWindow('Face Capture - Enter name in console')
            
            return person_name if person_name else None
            
        except (KeyboardInterrupt, EOFError):
            cv2.destroyWindow('Face Capture - Enter name in console')
            return None
    
    def _save_faces_to_staging(self, frame: np.ndarray, detections: List[Dict], person_name: str) -> int:
        """얼굴들을 face_staging으로 저장"""
        # 안전한 파일명 생성
        safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_name}_{timestamp}_manual"
        staging_dir = self.face_staging_dir / folder_name
        staging_dir.mkdir(exist_ok=True)
        
        # 메타데이터 생성
        metadata = {
            'person_name': person_name,
            'safe_name': safe_name,
            'created_at': timestamp,
            'source': 'manual_capture',
            'camera_id': self.camera_id,
            'face_count': len(detections),
            'confidence_threshold': self.confidence_threshold,
            'capture_session_id': str(uuid.uuid4())
        }
        
        metadata_file = staging_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 얼굴들 저장
        saved_count = 0
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 얼굴 영역 추출
            face_crop = self._extract_face_crop(frame, bbox)
            if face_crop is None:
                continue
            
            # 파일명 생성
            filename = f"face_{safe_name}_{timestamp}_{i:02d}_conf{confidence:.2f}.jpg"
            dest_path = staging_dir / filename
            
            # 이미지 저장
            success = cv2.imwrite(str(dest_path), face_crop)
            if success:
                saved_count += 1
        
        print(f"📁 저장 위치: {staging_dir}")
        return saved_count
    
    def _extract_face_crop(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """얼굴 영역 자르기"""
        x, y, w, h = bbox
        
        # 여유를 두고 자르기
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
            return None
        
        return face_crop
    
    def _save_current_frame(self, frame: np.ndarray):
        """현재 프레임 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{timestamp}.jpg"
        dest_path = self.output_dir / filename
        
        success = cv2.imwrite(str(dest_path), frame)
        if success:
            print(f"📸 프레임 저장: {dest_path}")
        else:
            print("❌ 프레임 저장 실패")
    
    def _toggle_recording(self, frame: np.ndarray):
        """녹화 토글"""
        if not self.is_recording:
            # 녹화 시작
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            dest_path = self.output_dir / filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (frame.shape[1], frame.shape[0])
            
            self.video_writer = cv2.VideoWriter(str(dest_path), fourcc, fps, frame_size)
            
            if self.video_writer.isOpened():
                self.is_recording = True
                self.recording_start_time = time.time()
                print(f"🔴 녹화 시작: {dest_path}")
            else:
                print("❌ 녹화 시작 실패")
                self.video_writer = None
        else:
            # 녹화 중지
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            elapsed = time.time() - self.recording_start_time
            self.is_recording = False
            print(f"⏹️  녹화 중지 (총 {elapsed:.1f}초)")
    
    def _toggle_pause(self):
        """일시정지 토글"""
        self.is_paused = not self.is_paused
        status = "일시정지" if self.is_paused else "재생"
        print(f"⏸️  {status}")
    
    def _adjust_confidence_threshold(self, delta: float):
        """신뢰도 임계값 조절"""
        new_threshold = self.confidence_threshold + delta
        self.confidence_threshold = max(self.min_confidence, 
                                      min(self.max_confidence, new_threshold))
        print(f"🎚️  신뢰도 임계값: {self.confidence_threshold:.2f}")
    
    def _update_fps(self):
        """FPS 업데이트"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def _print_help(self):
        """도움말 출력"""
        help_text = """
📋 키보드 명령어:
   i  - 정보 표시 토글
   +  - 신뢰도 임계값 증가
   -  - 신뢰도 임계값 감소
   s  - 현재 프레임 저장
   c  - 얼굴 캡처 + 이름 지정
   r  - 녹화 시작/중지
   p  - 일시정지/재생
   h  - 도움말 표시
   q  - 종료
        """
        print(help_text)
    
    def _cleanup(self):
        """리소스 정리"""
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        print(f"\n📊 세션 요약:")
        print(f"   캡처된 얼굴: {len(self.captured_faces)}개")
        print(f"   평균 FPS: {self.current_fps:.1f}")
        print("✅ 시스템 종료")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="수동 얼굴 캡처 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    args = parser.parse_args()
    
    # 하드웨어 연결 검증
    print("🔍 하드웨어 연결 상태 확인 중...")
    
    # 카메라 연결 테스트
    test_cap = cv2.VideoCapture(args.camera)
    if not test_cap.isOpened():
        print(f"❌ 카메라 {args.camera}에 연결할 수 없습니다.")
        print("💡 다른 카메라 ID를 시도해보세요: --camera 1")
        return
    
    ret, frame = test_cap.read()
    test_cap.release()
    
    if not ret:
        print(f"❌ 카메라 {args.camera}에서 영상을 읽을 수 없습니다.")
        return
    
    print(f"✅ 카메라 {args.camera} 연결 확인됨")
    
    # 시스템 시작
    capture_system = ManualFaceCaptureSystem(camera_id=args.camera)
    capture_system.start_capture()

if __name__ == "__main__":
    main() 