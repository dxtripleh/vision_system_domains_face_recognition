#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 얼굴 캡처 및 등록 시스템.

카메라에서 실시간으로 얼굴을 감지하고, 사용자가 선택한 얼굴을 바로 등록할 수 있습니다.
"""

import os
import sys
import time
import cv2
import uuid
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.entities.person import Person
from domains.face_recognition.core.entities.face import Face

logger = get_logger(__name__)


class SimpleFPSCounter:
    """간단한 FPS 카운터"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
    
    def tick(self) -> float:
        """프레임 카운트 증가 및 FPS 계산"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:  # 1초마다 FPS 업데이트
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps
    
    def get_fps(self) -> float:
        """현재 FPS 반환"""
        return self.fps


class RealTimeFaceCaptureAndRegister:
    """실시간 얼굴 캡처 및 등록 시스템"""
    
    def __init__(self, camera_id: int = 0):
        """초기화"""
        self.camera_id = camera_id
        self.camera = None
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService(
            config={
                'min_confidence': 0.5,
                'min_face_size': (80, 80),
                'max_faces': 5
            }
        )
        self.recognition_service = FaceRecognitionService()
        
        # 캡처 관련 설정
        self.face_staging = []  # 캡처된 얼굴들
        self.current_frame = None
        self.current_detections = []
        
        # UI 상태
        self.fps_counter = SimpleFPSCounter()
        self.show_help = True
        self.capture_mode = False  # True이면 캡처 모드
        self.selected_person_name = None
        
        # 출력 디렉토리
        self.output_dir = Path("data/temp/face_staging")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            logger.info("✅ 카메라 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 카메라 초기화 실패: {str(e)}")
            return False
    
    def get_person_name(self) -> Optional[str]:
        """사용자로부터 인물 이름 입력받기"""
        print("\n" + "="*50)
        print("🧑‍💼 새로운 인물 등록")
        print("="*50)
        
        while True:
            name = input("등록할 인물의 이름을 입력하세요 (취소: 'cancel'): ").strip()
            
            if name.lower() == 'cancel':
                return None
            
            if not name:
                print("❌ 이름을 입력해주세요.")
                continue
            
            if len(name) < 2:
                print("❌ 이름은 최소 2글자 이상이어야 합니다.")
                continue
            
            # 기존 인물과 중복 확인
            try:
                existing_persons = self.recognition_service.get_all_persons()
                existing_names = [p.name for p in existing_persons]
                
                if name in existing_names:
                    print(f"❌ '{name}'은 이미 등록된 인물입니다.")
                    print(f"기존 등록된 인물들: {', '.join(existing_names)}")
                    continue
            except Exception as e:
                logger.warning(f"기존 인물 목록 조회 실패: {str(e)}")
            
            confirm = input(f"'{name}'으로 등록하시겠습니까? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return name
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 처리 및 UI 그리기"""
        self.current_frame = frame.copy()
        
        # 얼굴 검출
        try:
            detection_result = self.detection_service.detect_faces(frame)
            self.current_detections = detection_result.faces
        except Exception as e:
            logger.error(f"얼굴 검출 오류: {str(e)}")
            self.current_detections = []
        
        # 화면에 그리기
        display_frame = frame.copy()
        
        # 검출된 얼굴들 표시
        for i, face in enumerate(self.current_detections):
            x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
            
            # 색상 결정
            if self.capture_mode:
                color = (0, 255, 255)  # 노란색 - 캡처 모드
                thickness = 3
            else:
                color = (0, 255, 0)    # 초록색 - 일반 모드
                thickness = 2
            
            # 바운딩 박스
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # 라벨
            label = f"Face {i+1}: {face.confidence.value:.2f}"
            if self.capture_mode:
                label = f"[CAPTURE] {label}"
            
            # 라벨 배경
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 라벨 텍스트
            cv2.putText(display_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 번호 표시 (캡처 모드에서)
            if self.capture_mode:
                cv2.putText(display_frame, str(i+1), (x + w//2 - 10, y + h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # 정보 오버레이
        self._draw_info_overlay(display_frame)
        
        return display_frame
    
    def _draw_info_overlay(self, frame: np.ndarray):
        """정보 오버레이 그리기"""
        height, width = frame.shape[:2]
        
        # FPS 표시
        fps = self.fps_counter.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 얼굴 수 표시
        face_count = len(self.current_detections)
        cv2.putText(frame, f"Faces: {face_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 모드 표시
        if self.capture_mode:
            mode_text = f"CAPTURE MODE: {self.selected_person_name}"
            cv2.putText(frame, mode_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 캡처 안내
            cv2.putText(frame, "Press 1-9 to capture face, ESC to exit capture mode", 
                       (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Normal Mode", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 도움말 표시
        if self.show_help:
            help_lines = [
                "Controls:",
                "c: Enter capture mode",
                "h: Toggle help",
                "q: Quit",
                "Space: Pause"
            ]
            
            for i, line in enumerate(help_lines):
                y_pos = height - 150 + (i * 25)
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def enter_capture_mode(self):
        """캡처 모드 진입"""
        if not self.current_detections:
            print("❌ 검출된 얼굴이 없습니다.")
            return
        
        # 인물 이름 입력
        person_name = self.get_person_name()
        if not person_name:
            print("❌ 캡처 모드 취소됨")
            return
        
        self.selected_person_name = person_name
        self.capture_mode = True
        self.show_help = False
        
        print(f"\n✅ 캡처 모드 활성화 - 대상: {person_name}")
        print("📷 1-9번 키를 눌러 해당 번호의 얼굴을 캡처하세요")
        print("🚪 ESC 키로 캡처 모드 종료")
    
    def capture_face(self, face_index: int) -> bool:
        """특정 얼굴 캡처 및 등록"""
        if not self.capture_mode or not self.current_detections:
            return False
        
        if face_index >= len(self.current_detections):
            print(f"❌ 얼굴 번호가 잘못되었습니다. (1-{len(self.current_detections)} 사용)")
            return False
        
        try:
            face = self.current_detections[face_index]
            x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
            
            # 얼굴 영역 추출
            face_image = self.current_frame[y:y+h, x:x+w]
            
            # 품질 검사
            if face_image.size == 0 or min(face_image.shape[:2]) < 80:
                print("❌ 얼굴 이미지가 너무 작거나 품질이 낮습니다.")
                return False
            
            # 임시 파일로 저장
            timestamp = int(time.time())
            temp_filename = f"captured_{self.selected_person_name}_{timestamp}.jpg"
            temp_path = self.output_dir / temp_filename
            
            cv2.imwrite(str(temp_path), face_image)
            
            # 인물 등록 또는 얼굴 추가
            person = self.register_face_to_person(str(temp_path), self.selected_person_name)
            
            if person:
                print(f"✅ 얼굴 등록 성공: {self.selected_person_name}")
                print(f"   저장 위치: {temp_path}")
                
                # 등록된 얼굴 수 표시
                person_faces = self.recognition_service.face_repository.find_faces_by_person(person.person_id)
                print(f"   총 등록된 얼굴 수: {len(person_faces)}")
                
                return True
            else:
                print("❌ 얼굴 등록 실패")
                return False
                
        except Exception as e:
            logger.error(f"얼굴 캡처 중 오류: {str(e)}")
            print(f"❌ 얼굴 캡처 중 오류: {str(e)}")
            return False
    
    def register_face_to_person(self, image_path: str, person_name: str) -> Optional[Person]:
        """얼굴을 인물에 등록"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                return None
            
            # 임베딩 추출
            embedding = self.recognition_service.extract_embedding(image)
            
            # 기존 인물 확인 또는 새 인물 생성
            existing_persons = self.recognition_service.person_repository.list_all_persons()
            person = None
            
            for p in existing_persons:
                if p.name == person_name:
                    person = p
                    break
            
            if not person:
                # 새 인물 생성
                person = Person(name=person_name)
                saved_person = self.recognition_service.person_repository.save(person)
                person = saved_person
            
            # 얼굴 엔티티 생성
            face = Face(
                face_id=str(uuid.uuid4()),
                person_id=person.person_id,
                embedding=embedding,
                image_path=image_path,
                quality_score=0.8  # 임시 품질 점수
            )
            
            # 얼굴 저장
            saved_face = self.recognition_service.face_repository.save(face)
            
            return person
            
        except Exception as e:
            logger.error(f"얼굴 등록 중 오류: {str(e)}")
            return None
    
    def exit_capture_mode(self):
        """캡처 모드 종료"""
        self.capture_mode = False
        self.selected_person_name = None
        self.show_help = True
        print("\n🚪 캡처 모드 종료")
    
    def handle_key_input(self, key: int) -> str:
        """키보드 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('h'):
            self.show_help = not self.show_help
            return 'toggle_help'
        elif key == ord('c') and not self.capture_mode:
            self.enter_capture_mode()
            return 'enter_capture'
        elif key == 27:  # ESC
            if self.capture_mode:
                self.exit_capture_mode()
            return 'exit_capture'
        elif self.capture_mode and ord('1') <= key <= ord('9'):
            face_index = key - ord('1')  # 0-based index
            success = self.capture_face(face_index)
            return 'capture_face' if success else 'capture_failed'
        elif key == ord(' '):
            return 'pause'
        
        return None
    
    def run(self):
        """메인 실행 루프"""
        if not self.initialize_camera():
            return False
        
        print("\n" + "="*60)
        print("🎥 실시간 얼굴 캡처 및 등록 시스템")
        print("="*60)
        print("📋 사용법:")
        print("  c: 캡처 모드 진입 (인물 이름 입력 후 얼굴 선택)")
        print("  h: 도움말 표시/숨김")
        print("  q: 프로그램 종료")
        print("="*60)
        
        try:
            paused = False
            
            while True:
                if not paused:
                    ret, frame = self.camera.read()
                    if not ret:
                        logger.warning("프레임 읽기 실패")
                        continue
                    
                    # 프레임 처리
                    display_frame = self.process_frame(frame)
                    
                    # FPS 업데이트
                    self.fps_counter.tick()
                else:
                    # 일시정지 상태에서는 마지막 프레임 유지
                    display_frame = self.current_frame if self.current_frame is not None else frame
                
                # 화면 표시
                cv2.imshow('Face Capture & Register', display_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    action = self.handle_key_input(key)
                    
                    if action == 'quit':
                        break
                    elif action == 'pause':
                        paused = not paused
                        print(f"⏸️ {'일시정지' if paused else '재생'}")
        
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"실행 중 오류: {str(e)}")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 리소스 정리 중...")
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        print("\n✅ 프로그램 종료")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="실시간 얼굴 캡처 및 등록 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 로깅 설정
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # 출력 디렉토리 생성
    os.makedirs("data/temp/face_staging", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    try:
        # 캡처 시스템 실행
        capture_system = RealTimeFaceCaptureAndRegister(camera_id=args.camera)
        success = capture_system.run()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 