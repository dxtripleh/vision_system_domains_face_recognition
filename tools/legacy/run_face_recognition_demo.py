#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 인식 데모.

얼굴 검출과 인식 기능을 포함한 완전한 데모입니다.
"""

import cv2
import numpy as np
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService

logger = get_logger(__name__)

class FaceRecognitionDemo:
    """얼굴 인식 데모 클래스"""
    
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
        self.recognition_service = FaceRecognitionService(use_mock=True)
        
        # UI 상태
        self.show_info = True
        self.frame_count = 0
        self.start_time = time.time()
        self.registered_persons = {}
        
        # 디렉토리 생성
        self.create_directories()
        
        # 등록된 인물 로드
        self.load_registered_persons()
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            "data/output", "data/logs", "data/temp",
            "data/storage/faces", "data/storage/persons"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_registered_persons(self):
        """등록된 인물 정보 로드"""
        try:
            persons = self.recognition_service.get_all_persons()
            self.registered_persons = {p.person_id: p.name for p in persons}
            logger.info(f"등록된 인물 {len(self.registered_persons)}명 로드")
        except Exception as e:
            logger.warning(f"등록된 인물 로드 실패: {str(e)}")
            self.registered_persons = {}
    
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 처리"""
        try:
            # 얼굴 검출
            detection_result = self.detection_service.detect_faces(frame)
            faces = detection_result.faces
            
            # 각 얼굴에 대해 인식 수행
            for i, face in enumerate(faces):
                x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
                
                # 얼굴 영역 추출
                face_region = frame[y:y+h, x:x+w]
                
                # 얼굴 인식 시도
                person_name = "Unknown"
                confidence = face.confidence.value
                
                try:
                    if face_region.size > 0:
                        # 임베딩 추출 (Mock 모드에서는 더미 임베딩)
                        embedding = self.recognition_service.extract_embedding(face_region)
                        face.embedding = embedding
                        
                        # 인물 식별
                        identified_person = self.recognition_service.identify_face(face)
                        if identified_person:
                            person_name = identified_person.name
                except Exception as e:
                    logger.debug(f"얼굴 인식 오류: {str(e)}")
                
                # 바운딩 박스 색상 결정
                if person_name != "Unknown":
                    color = (0, 255, 0)  # 녹색 - 알려진 인물
                else:
                    color = (0, 255, 255)  # 노란색 - 미지의 인물
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 라벨 텍스트
                label = f"{person_name} ({confidence:.2f})"
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                
                # 라벨 텍스트
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 얼굴 중심점
                center_x, center_y = x + w//2, y + h//2
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"프레임 처리 오류: {str(e)}")
            return frame
    
    def draw_info_overlay(self, frame: np.ndarray):
        """정보 오버레이 그리기"""
        if not self.show_info:
            return
        
        # FPS 계산
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # 배경 박스
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # 정보 텍스트
        y_offset = 35
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Registered: {len(self.registered_persons)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, "Controls:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 20
        cv2.putText(frame, "q: Quit, s: Screenshot", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(frame, "i: Toggle info, r: Reload", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def handle_key_input(self, key: int) -> str:
        """키 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"data/output/face_recognition_demo_{timestamp}.jpg"
            return f'screenshot:{filename}'
        elif key == ord('i'):
            self.show_info = not self.show_info
            return f'toggle_info:{self.show_info}'
        elif key == ord('r'):
            self.load_registered_persons()
            return 'reload'
        return 'continue'
    
    def run(self) -> bool:
        """데모 실행"""
        print("🎯 얼굴 인식 데모 시작")
        print("=" * 50)
        print("기능:")
        print("  - 실시간 얼굴 검출")
        print("  - 등록된 인물 인식")
        print("  - FPS 및 성능 모니터링")
        print("=" * 50)
        print("키보드 조작:")
        print("  'q' - 종료")
        print("  's' - 스크린샷 저장")
        print("  'i' - 정보 표시 토글")
        print("  'r' - 등록된 인물 다시 로드")
        print("=" * 50)
        
        if not self.initialize_camera():
            return False
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("프레임 읽기 실패")
                    break
                
                # 프레임 처리
                processed_frame = self.process_frame(frame)
                
                # 정보 오버레이
                self.draw_info_overlay(processed_frame)
                
                # 프레임 표시
                cv2.imshow('Face Recognition Demo', processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                action = self.handle_key_input(key)
                
                if action == 'quit':
                    print("🛑 종료 요청")
                    break
                elif action.startswith('screenshot:'):
                    filename = action.split(':', 1)[1]
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 스크린샷 저장: {filename}")
                elif action.startswith('toggle_info:'):
                    status = action.split(':', 1)[1]
                    print(f"ℹ️ 정보 표시: {'ON' if status == 'True' else 'OFF'}")
                elif action == 'reload':
                    print("🔄 등록된 인물 다시 로드")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 중단했습니다")
        
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            logger.error(f"데모 실행 오류: {str(e)}")
            return False
        
        finally:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            total_time = time.time() - self.start_time
            print("✅ 데모 종료")
            print(f"📊 총 실행 시간: {total_time:.1f}초")
            print(f"📊 총 프레임 수: {self.frame_count}")
            print(f"📊 평균 FPS: {self.frame_count/total_time:.1f}")
        
        return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="얼굴 인식 데모")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # 데모 실행
    demo = FaceRecognitionDemo(camera_id=args.camera)
    success = demo.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 