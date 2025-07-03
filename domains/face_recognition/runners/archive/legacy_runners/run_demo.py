#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 얼굴인식 데모 시스템.

웹캠을 통해 실시간으로 얼굴을 검출하고 인식하는 데모 시스템입니다.
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

# 프로젝트 모듈
from common.logging import setup_logging, get_logger
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService

# 기본 모니터링 (새로 추가)
from shared.security.privacy import FaceDataProtection

logger = get_logger(__name__)

class RealtimeDemo:
    """실시간 얼굴인식 데모"""
    
    def __init__(self, camera_id: int = 0):
        """초기화"""
        self.camera_id = camera_id
        
        # 얼굴 검출 서비스 초기화 (OpenCV Haar Cascade 사용)
        self.detection_service = FaceDetectionService(
            config={
                'min_confidence': 0.3,
                'min_face_size': (30, 30),
                'max_faces': 10
            }
        )
        
        # 보안 모듈 초기화 (새로 추가)
        self.data_protection = FaceDataProtection()
        
        # 성능 추적
        self.frame_count = 0
        self.start_time = time.time()
        
    def run(self):
        """데모 실행"""
        # 카메라 초기화
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {self.camera_id}를 열 수 없습니다.")
            logger.error(f"Cannot open camera {self.camera_id}")
            return False
        
        # 카메라 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🎥 실시간 얼굴검출 데모 시작!")
        print("키보드 조작:")
        print("  'q' - 종료")
        print("  's' - 스크린샷 저장")
        print("  'i' - 정보 표시 토글")
        print("=" * 50)
        
        show_info = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # 얼굴 검출
                try:
                    detection_result = self.detection_service.detect_faces(frame)
                    faces = detection_result.faces
                    processing_time = detection_result.processing_time_ms
                    
                    # 검출 결과 그리기
                    for i, face in enumerate(faces):
                        bbox = face.bbox
                        confidence = face.confidence.value
                        
                        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
                        
                        # 바운딩 박스 그리기
                        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # 얼굴 번호 및 신뢰도 표시
                        label = f"Face {i+1}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 얼굴 중심점 표시
                        center_x, center_y = bbox.center
                        cv2.circle(frame, (int(center_x), int(center_y)), 3, color, -1)
                    
                except Exception as e:
                    logger.error(f"Face detection error: {str(e)}")
                    faces = []
                    processing_time = 0
                
                # FPS 계산
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # 정보 표시
                if show_info:
                    # 배경 박스
                    cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
                    cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
                    
                    # 텍스트 정보
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Faces: {len(faces)}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Process: {processing_time:.1f}ms", (20, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to quit", (20, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 프레임 표시
                cv2.imshow('Face Detection Demo', frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 종료 요청")
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    os.makedirs("data/output", exist_ok=True)
                    filename = f"data/output/demo_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 스크린샷 저장: {filename}")
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"ℹ️ 정보 표시: {'ON' if show_info else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 중단했습니다.")
            logger.info("Demo interrupted by user")
        
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            logger.error(f"Demo error: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ 데모 종료")
            logger.info("Demo finished")
        
        return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="실시간 얼굴검출 데모")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    print("🎯 얼굴검출 데모 시스템")
    print("=" * 50)
    print(f"카메라 ID: {args.camera}")
    print(f"로그 레벨: {'DEBUG' if args.verbose else 'INFO'}")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    
    # 데모 실행
    demo = RealtimeDemo(camera_id=args.camera)
    success = demo.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 