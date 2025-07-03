#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
비전 시스템 메인 진입점.

이 스크립트는 비전 시스템의 통합 진입점으로 다양한 모드로 실행할 수 있습니다.
- API 서버 모드
- 실시간 웹캠 모드  
- 이미지/비디오 검출 모드

Example:
    API 서버 실행:
    $ python main.py --mode api --port 8000
    
    실시간 웹캠:
    $ python main.py --mode realtime --camera 0
    
    이미지 검출:
    $ python main.py --mode detection --input image.jpg
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from common.logging import setup_logging

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """명령줄 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="비전 시스템 통합 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["api", "realtime", "detection"],
        required=True,
        help="실행 모드 선택"
    )
    
    # API 모드 옵션
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API 서버 호스트")
    
    # 카메라 관련 옵션
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID")
    parser.add_argument("--width", type=int, default=640, help="카메라 해상도 너비")
    parser.add_argument("--height", type=int, default=480, help="카메라 해상도 높이")
    
    # 검출 관련 옵션
    parser.add_argument("--input", type=str, help="입력 이미지/비디오 경로")
    parser.add_argument("--output", type=str, help="출력 저장 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 임계값")
    
    # 공통 옵션
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--no-hardware-check", action="store_true", help="하드웨어 검증 생략")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    return parser.parse_args()

def run_api_server(args: argparse.Namespace):
    """API 서버 실행."""
    logger.info(f"API 서버 시작: {args.host}:{args.port}")
    
    try:
        import uvicorn
        from domains.face_recognition.interfaces.face_recognition_api import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info" if not args.debug else "debug"
        )
        
    except ImportError as e:
        logger.error(f"API 서버 모듈 로드 실패: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류: {e}")
        sys.exit(1)

def run_realtime_recognition(args: argparse.Namespace):
    """실시간 웹캠 얼굴인식 실행."""
    logger.info(f"실시간 웹캠 시작: 카메라 {args.camera}")
    
    # 하드웨어 검증
    if not args.no_hardware_check:
        try:
            from scripts.core.validation.validate_hardware_connection import verify_hardware_connection
            verify_hardware_connection(
                camera_id=args.camera,
                required_fps=15,
                required_resolution=(args.width, args.height)
            )
        except Exception as e:
            logger.error(f"하드웨어 검증 실패: {e}")
            sys.exit(1)
    
    try:
        # 실시간 처리 실행
        import cv2
        import numpy as np
        from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
        from domains.face_recognition.infrastructure.detection.retinaface_detection_engine import RetinaFaceDetectionEngine
        
        # 서비스 초기화
        detection_engine = RetinaFaceDetectionEngine()
        face_service = FaceRecognitionService(detection_engine=detection_engine)
        
        # 카메라 초기화
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        
        if not cap.isOpened():
            logger.error("카메라를 열 수 없습니다")
            sys.exit(1)
        
        logger.info("실시간 처리 시작 (q를 눌러 종료)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임을 읽을 수 없습니다")
                continue
            
            # 얼굴 검출
            faces = face_service.detect_faces(frame)
            
            # 검출 결과 표시
            for face in faces:
                bbox = face.bounding_box
                cv2.rectangle(frame, 
                            (int(bbox.x1), int(bbox.y1)), 
                            (int(bbox.x2), int(bbox.y2)), 
                            (0, 255, 0), 2)
                
                # 신뢰도 표시
                cv2.putText(frame, f"{face.confidence.value:.2f}", 
                          (int(bbox.x1), int(bbox.y1-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('실시간 얼굴 인식', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except ImportError as e:
        logger.error(f"실시간 처리 모듈 로드 실패: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실시간 처리 실행 중 오류: {e}")
        sys.exit(1)

def run_detection(args: argparse.Namespace):
    """이미지/비디오 검출 실행."""
    if not args.input:
        logger.error("검출 모드에서는 --input 옵션이 필요합니다")
        sys.exit(1)
    
    if not Path(args.input).exists():
        logger.error(f"입력 파일이 존재하지 않습니다: {args.input}")
        sys.exit(1)
    
    logger.info(f"검출 시작: {args.input}")
    
    try:
        import cv2
        from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
        from domains.face_recognition.infrastructure.detection.retinaface_detection_engine import RetinaFaceDetectionEngine
        
        # 서비스 초기화
        detection_engine = RetinaFaceDetectionEngine()
        face_service = FaceRecognitionService(detection_engine=detection_engine)
        
        # 이미지 로드
        image = cv2.imread(args.input)
        if image is None:
            logger.error("이미지를 로드할 수 없습니다")
            sys.exit(1)
        
        # 얼굴 검출
        faces = face_service.detect_faces(image)
        logger.info(f"검출된 얼굴 수: {len(faces)}")
        
        # 검출 결과 표시
        for i, face in enumerate(faces):
            bbox = face.bounding_box
            cv2.rectangle(image, 
                        (int(bbox.x1), int(bbox.y1)), 
                        (int(bbox.x2), int(bbox.y2)), 
                        (0, 255, 0), 2)
            
            # 신뢰도 표시
            cv2.putText(image, f"Face {i+1}: {face.confidence.value:.2f}", 
                      (int(bbox.x1), int(bbox.y1-10)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 저장 또는 표시
        if args.output:
            cv2.imwrite(args.output, image)
            logger.info(f"결과 저장됨: {args.output}")
        else:
            cv2.imshow('검출 결과', image)
            logger.info("아무 키나 눌러 종료")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except ImportError as e:
        logger.error(f"검출 모듈 로드 실패: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"검출 실행 중 오류: {e}")
        sys.exit(1)

def main():
    """메인 함수."""
    # 로깅 설정
    setup_logging()
    
    # 인자 파싱
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    logger.info("="*50)
    logger.info("비전 시스템 시작")
    logger.info(f"실행 모드: {args.mode}")
    logger.info("="*50)
    
    try:
        # 모드별 실행
        if args.mode == "api":
            run_api_server(args)
        elif args.mode == "realtime":
            run_realtime_recognition(args)
        elif args.mode == "detection":
            run_detection(args)
        else:
            logger.error(f"지원하지 않는 모드: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("비전 시스템 종료")

if __name__ == "__main__":
    main() 