#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 얼굴인식 시스템.

웹캠을 통해 실시간으로 얼굴을 검출하고 인식하는 시스템입니다.
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

# 하드웨어 검증 임포트
from scripts.core.validation.validate_hardware_connection import validate_hardware_for_runtime

from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from shared.vision_core.detection.face_detector import FaceDetector
from common.logging import setup_logging, get_logger

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


class RealTimeFaceRecognition:
    """실시간 얼굴인식 시스템"""
    
    def __init__(self, 
                 camera_id: int = 0,
                 detection_confidence: float = 0.5,
                 recognition_threshold: float = 0.6):
        """
        초기화
        
        Args:
            camera_id: 카메라 ID
            detection_confidence: 검출 신뢰도 임계값
            recognition_threshold: 인식 신뢰도 임계값
        """
        self.camera_id = camera_id
        self.detection_confidence = detection_confidence
        self.recognition_threshold = recognition_threshold
        
        # 🔧 하드웨어 검증 (필수)
        if not self._validate_hardware():
            raise RuntimeError("하드웨어 검증 실패. 실제 카메라를 연결하고 다시 시도하세요.")
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService()
        self.recognition_service = FaceRecognitionService()
        
        # 검출기 초기화
        self.detector = FaceDetector(
            detector_type="opencv",
            confidence_threshold=detection_confidence
        )
        
        # 성능 추적
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        
        # 상태
        self.is_running = False
        self.camera = None
        self.show_info = True
        self.recording = False
        
        logger.info("실시간 얼굴인식 시스템 초기화 완료")
    
    def _validate_hardware(self) -> bool:
        """하드웨어 검증"""
        logger.info("🔍 하드웨어 연결 상태 검증 중...")
        
        # 시뮬레이션 방지 체크
        if os.environ.get("USE_SIMULATION", "False").lower() in ["true", "1", "yes"]:
            logger.error("❌ 시뮬레이션 모드는 금지되어 있습니다.")
            return False
        
        if os.environ.get("USE_MOCK", "False").lower() in ["true", "1", "yes"]:
            logger.error("❌ Mock 모드는 실행 시 금지되어 있습니다.")
            return False
        
        # 하드웨어 검증 실행
        validation_result = validate_hardware_for_runtime()
        
        if not validation_result:
            logger.error("❌ 하드웨어 검증 실패")
            return False
        
        logger.info("✅ 하드웨어 검증 완료")
        return True
    
    def initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            logger.info(f"📹 카메라 {self.camera_id} 초기화 중...")
            
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                logger.error(f"❌ 카메라 {self.camera_id} 열기 실패")
                return False
            
            # 카메라 설정
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # 카메라 테스트
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                logger.error("❌ 카메라에서 프레임을 읽을 수 없습니다")
                return False
            
            if test_frame.shape[0] < 100 or test_frame.shape[1] < 100:
                logger.error(f"❌ 카메라 해상도가 너무 낮습니다: {test_frame.shape}")
                return False
            
            logger.info(f"✅ 카메라 초기화 완료 - 해상도: {test_frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 카메라 초기화 실패: {str(e)}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        프레임 처리
        
        Args:
            frame: 입력 프레임
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: (처리된 프레임, 인식 결과)
        """
        results = []
        
        try:
            # 얼굴 검출
            detection_result = self.detection_service.detect_faces(frame)
            
            for face in detection_result.faces:
                # 얼굴 영역 추출
                face_region = self._extract_face_region(frame, face.bbox)
                
                # 얼굴 인식
                if face_region is not None and face_region.size > 0:
                    # 임베딩 추출
                    embedding = self.recognition_service.extract_embedding(face_region)
                    face.embedding = embedding
                    
                    # 인물 식별
                    identified_person = self.recognition_service.identify_face(face)
                    
                    # 결과 저장
                    result = {
                        'face': face,
                        'person': identified_person,
                        'face_region': face_region
                    }
                    results.append(result)
            
            # 프레임에 결과 그리기
            annotated_frame = self._draw_results(frame, results)
            
            return annotated_frame, results
            
        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {str(e)}")
            return frame, []
    
    def _extract_face_region(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """얼굴 영역 추출"""
        try:
            x, y, w, h = bbox
            
            # 마진 추가 (20%)
            margin = 0.2
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # 확장된 영역 계산
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            # 얼굴 영역 추출
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            return face_region
            
        except Exception as e:
            logger.warning(f"얼굴 영역 추출 실패: {str(e)}")
            return None
    
    def _draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """결과를 프레임에 그리기"""
        annotated_frame = frame.copy()
        
        for result in results:
            face = result['face']
            person = result['person']
            
            x, y, w, h = face.bbox
            
            # 바운딩 박스 색상 결정
            if person is not None:
                color = (0, 255, 0)  # 초록색 - 인식됨
                label = f"{person.name} ({face.confidence:.2f})"
            else:
                color = (0, 0, 255)  # 빨간색 - 미인식
                label = f"Unknown ({face.confidence:.2f})"
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # 라벨 배경 그리기
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 라벨 텍스트 그리기
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 정보 오버레이
        if self.show_info:
            annotated_frame = self._draw_info_overlay(annotated_frame, len(results))
        
        return annotated_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, face_count: int) -> np.ndarray:
        """정보 오버레이 그리기"""
        height, width = frame.shape[:2]
        
        # FPS 표시
        current_fps = self.fps_counter.get_fps()
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 얼굴 수 표시
        face_text = f"Faces: {face_count}"
        cv2.putText(frame, face_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 프레임 카운트 표시
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 도움말 표시
        help_text = "Press 'h' for help, 'q' to quit"
        help_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, help_text, (width - help_size[0] - 10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def handle_key_input(self, key: int) -> str:
        """키보드 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('i'):
            self.show_info = not self.show_info
            logger.info(f"정보 표시: {'ON' if self.show_info else 'OFF'}")
            return 'toggle_info'
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"data/output/capture_{timestamp}.jpg"
            return 'save_frame'
        elif key == ord('r'):
            self.recording = not self.recording
            logger.info(f"녹화: {'시작' if self.recording else '중지'}")
            return 'toggle_record'
        elif key == ord('h'):
            return 'show_help'
        return None
    
    def show_help(self):
        """도움말 표시"""
        print("\n" + "="*50)
        print("📋 실시간 얼굴인식 시스템 단축키")
        print("="*50)
        print("q: 프로그램 종료")
        print("i: 정보 표시 토글")
        print("s: 현재 프레임 저장")
        print("r: 녹화 시작/중지")
        print("h: 이 도움말 표시")
        print("="*50 + "\n")
    
    def run(self):
        """메인 실행 루프"""
        if not self.initialize_camera():
            logger.error("카메라 초기화 실패")
            return
        
        logger.info("🚀 실시간 얼굴인식 시작")
        logger.info("Press 'h' for help, 'q' to quit")
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.warning("프레임 읽기 실패")
                    break
                
                # 프레임 처리
                processed_frame, results = self.process_frame(frame)
                
                # 프레임 표시
                cv2.imshow('Real-time Face Recognition', processed_frame)
                
                # FPS 카운터 업데이트
                self.fps_counter.tick()
                self.frame_count += 1
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 키가 눌렸을 때
                    action = self.handle_key_input(key)
                    
                    if action == 'quit':
                        break
                    elif action == 'show_help':
                        self.show_help()
                    elif action == 'save_frame':
                        timestamp = int(time.time())
                        filename = f"data/output/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        logger.info(f"프레임 저장: {filename}")
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"실행 중 오류: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 리소스 정리 중...")
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        logger.info("✅ 정리 완료")


class FPSCounter:
    """FPS 카운터"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def tick(self) -> float:
        """FPS 카운트 업데이트"""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # 윈도우 크기 유지
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """현재 FPS 반환"""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / time_span


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="실시간 얼굴인식 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--detection-conf", type=float, default=0.5, help="검출 신뢰도 (기본값: 0.5)")
    parser.add_argument("--recognition-threshold", type=float, default=0.6, help="인식 임계값 (기본값: 0.6)")
    parser.add_argument("--no-hardware-check", action="store_true", help="하드웨어 검증 건너뛰기 (개발용)")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("🎥 실시간 얼굴인식 시스템")
    print("=" * 60)
    
    # 개발용 옵션이 아닌 경우 하드웨어 검증 필수
    if not args.no_hardware_check:
        logger.info("🔍 하드웨어 검증 수행 중...")
        if not validate_hardware_for_runtime():
            logger.error("❌ 하드웨어 검증 실패. 실제 카메라를 연결하고 다시 시도하세요.")
            return 1
    
    try:
        # 실시간 얼굴인식 시스템 시작
        system = RealTimeFaceRecognition(
            camera_id=args.camera,
            detection_confidence=args.detection_conf,
            recognition_threshold=args.recognition_threshold
        )
        
        system.run()
        
        logger.info("✅ 프로그램 정상 종료")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 프로그램 실행 실패: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 