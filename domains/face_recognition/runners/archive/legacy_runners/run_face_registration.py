#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 등록 스크립트.

카메라를 통해 얼굴을 캡처하고 등록합니다.
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService

logger = get_logger(__name__)

class FaceRegistration:
    """얼굴 등록 클래스"""
    
    def __init__(self, camera_id: int = 0):
        """초기화"""
        self.camera_id = camera_id
        self.camera = None
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService(
            config={
                'min_confidence': 0.7,  # 등록시에는 높은 신뢰도 요구
                'min_face_size': (100, 100),
                'max_faces': 1  # 한 번에 한 명만
            }
        )
        self.recognition_service = FaceRecognitionService(use_mock=True)
        
        # 캡처 설정
        self.captured_faces = []
        self.target_captures = 5  # 5장 캡처
        self.capture_interval = 1.0  # 1초 간격
        self.last_capture_time = 0
        
        # 디렉토리 생성
        self.create_directories()
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            "data/output", "data/logs", "data/temp",
            "data/storage/faces", "data/storage/persons"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                logger.error(f"카메라 {self.camera_id} 열기 실패")
                return False
            
            # 카메라 설정
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("카메라 초기화 성공")
            return True
            
        except Exception as e:
            logger.error(f"카메라 초기화 실패: {str(e)}")
            return False
    
    def get_person_name(self) -> Optional[str]:
        """등록할 인물 이름 입력받기"""
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
    
    def capture_faces(self, person_name: str) -> bool:
        """얼굴 캡처"""
        print(f"\n📸 {person_name}님의 얼굴을 캡처합니다")
        print(f"총 {self.target_captures}장의 사진을 {self.capture_interval}초 간격으로 촬영합니다")
        print("카메라를 정면으로 바라보고 다양한 각도로 얼굴을 움직여주세요")
        print("키보드 조작:")
        print("  'c' - 수동 캡처")
        print("  'q' - 취소")
        print("  스페이스바 - 자동 캡처 시작")
        print("="*50)
        
        auto_capture = False
        
        try:
            while len(self.captured_faces) < self.target_captures:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("프레임 읽기 실패")
                    break
                
                # 얼굴 검출
                detection_result = self.detection_service.detect_faces(frame)
                faces = detection_result.faces
                
                # 화면에 그리기
                display_frame = frame.copy()
                
                if faces:
                    face = faces[0]  # 첫 번째 얼굴만 사용
                    x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
                    
                    # 바운딩 박스
                    color = (0, 255, 0)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 신뢰도 표시
                    confidence = face.confidence.value
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 자동 캡처
                    current_time = time.time()
                    if (auto_capture and 
                        current_time - self.last_capture_time >= self.capture_interval and
                        confidence >= 0.7):
                        
                        # 얼굴 영역 추출
                        face_region = frame[y:y+h, x:x+w]
                        self.captured_faces.append(face_region.copy())
                        self.last_capture_time = current_time
                        
                        print(f"📸 자동 캡처 {len(self.captured_faces)}/{self.target_captures}")
                        
                        # 캡처 효과
                        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                                     (255, 255, 255), 10)
                
                # 진행 상황 표시
                progress_text = f"Captured: {len(self.captured_faces)}/{self.target_captures}"
                cv2.putText(display_frame, progress_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                mode_text = "Auto Capture ON" if auto_capture else "Manual Mode"
                cv2.putText(display_frame, mode_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if auto_capture else (255, 255, 255), 2)
                
                # 안내 메시지
                if not faces:
                    cv2.putText(display_frame, "No face detected", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif faces[0].confidence.value < 0.7:
                    cv2.putText(display_frame, "Face confidence too low", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow(f'Face Registration - {person_name}', display_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("❌ 캡처 취소")
                    return False
                elif key == ord('c') and faces and faces[0].confidence.value >= 0.7:
                    # 수동 캡처
                    face = faces[0]
                    x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
                    face_region = frame[y:y+h, x:x+w]
                    self.captured_faces.append(face_region.copy())
                    print(f"📸 수동 캡처 {len(self.captured_faces)}/{self.target_captures}")
                elif key == ord(' '):
                    # 자동 캡처 토글
                    auto_capture = not auto_capture
                    print(f"🔄 자동 캡처: {'ON' if auto_capture else 'OFF'}")
                    self.last_capture_time = time.time()
            
            print(f"✅ 총 {len(self.captured_faces)}장의 얼굴 이미지 캡처 완료")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 중단했습니다")
            return False
        
        except Exception as e:
            print(f"❌ 캡처 중 오류 발생: {str(e)}")
            logger.error(f"얼굴 캡처 오류: {str(e)}")
            return False
    
    def register_person(self, person_name: str) -> bool:
        """인물 등록"""
        if not self.captured_faces:
            print("❌ 캡처된 얼굴이 없습니다")
            return False
        
        print(f"\n🔄 {person_name}님을 등록 중...")
        
        try:
            # 인물 등록
            person_id = self.recognition_service.register_person(
                name=person_name,
                face_images=self.captured_faces,
                metadata={
                    'registration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'capture_count': len(self.captured_faces)
                }
            )
            
            print(f"✅ {person_name}님이 성공적으로 등록되었습니다!")
            print(f"   Person ID: {person_id}")
            print(f"   등록된 얼굴 수: {len(self.captured_faces)}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 등록 중 오류 발생: {str(e)}")
            logger.error(f"인물 등록 오류: {str(e)}")
            return False
    
    def run(self) -> bool:
        """등록 프로세스 실행"""
        print("🎯 얼굴 등록 시스템")
        print("=" * 50)
        
        if not self.initialize_camera():
            return False
        
        try:
            # 인물 이름 입력
            person_name = self.get_person_name()
            if not person_name:
                print("❌ 등록 취소")
                return False
            
            # 얼굴 캡처
            if not self.capture_faces(person_name):
                return False
            
            # 인물 등록
            if not self.register_person(person_name):
                return False
            
            print("\n🎉 등록이 완료되었습니다!")
            return True
            
        finally:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="얼굴 등록 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # 등록 시스템 실행
    registration = FaceRegistration(camera_id=args.camera)
    success = registration.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 