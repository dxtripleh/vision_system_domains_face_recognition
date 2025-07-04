#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition.

실시간 얼굴인식을 수행하는 메인 실행 스크립트입니다.
USB 카메라 또는 이미지 파일을 입력으로 받아 얼굴을 검출하고 인식합니다.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from common.config import load_config
from .services.service import FaceRecognitionService

logger = get_logger(__name__)

class FaceRecognitionRunner:
    """실시간 얼굴인식 실행기"""
    
    def __init__(self, config_path: Optional[str] = None):
        """초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        if config_path:
            self.config = load_config(config_path)
        else:
            default_config_path = project_root / "config" / "face_recognition.yaml"
            self.config = load_config(str(default_config_path))
        
        # 하드웨어 환경 감지 및 최적화
        self.hardware_config = self._detect_hardware_environment()
        logger.info(f"하드웨어 환경: {self.hardware_config}")
        
        # 얼굴인식 서비스 초기화
        self.face_service = FaceRecognitionService(config=self.config)
        
        # 실행 상태
        self.is_running = False
        self.is_paused = False
        self.save_results = False
        self.show_help = False
        
        logger.info("얼굴인식 실행기 초기화 완료")
    
    def _detect_hardware_environment(self) -> dict:
        """하드웨어 환경 자동 감지"""
        import platform
        import psutil
        
        system = platform.system().lower()
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total // (1024**3)
        
        # GPU 확인
        gpu_available = False
        gpu_memory = 0
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except ImportError:
            pass
        
        # Jetson 플랫폼 확인
        is_jetson = self._is_jetson_platform()
        
        if is_jetson:
            device_type = "jetson"
            batch_size = 1
            precision = "fp16"
        elif gpu_available and gpu_memory >= 4:
            device_type = "gpu"
            batch_size = 4 if gpu_memory >= 8 else 2
            precision = "fp16"
        else:
            device_type = "cpu"
            batch_size = 1
            precision = "fp32"
        
        return {
            'system': system,
            'device_type': device_type,
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory,
            'is_jetson': is_jetson,
            'batch_size': batch_size,
            'precision': precision
        }
    
    def _is_jetson_platform(self) -> bool:
        """Jetson 플랫폼 감지"""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    def _create_camera(self, source: str) -> cv2.VideoCapture:
        """플랫폼별 최적화된 카메라 생성"""
        system = self.hardware_config['system']
        
        # 카메라 ID 또는 파일 경로 처리
        if source.isdigit():
            camera_id = int(source)
            
            if system == "windows":
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            elif system == "linux":
                cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                if self.hardware_config['is_jetson']:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(camera_id)
        else:
            # 파일 또는 RTSP 스트림
            cap = cv2.VideoCapture(source)
        
        # 카메라 설정 최적화
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 버퍼 크기 최소화 (지연 감소)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    
    def _handle_keyboard_input(self) -> Optional[str]:
        """키보드 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('p'):
            return 'toggle_pause'
        elif key == ord('s'):
            return 'save_frame'
        elif key == ord('r'):
            return 'toggle_record'
        elif key == ord('h'):
            return 'toggle_help'
        elif key == ord('i'):
            return 'toggle_info'
        elif key == ord(' '):  # 스페이스바
            return 'toggle_pause'
        
        return None
    
    def _display_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """도움말 오버레이 표시"""
        help_text = [
            "=== 키보드 단축키 ===",
            "q: 종료",
            "p/Space: 일시정지/재생",
            "s: 현재 프레임 저장",
            "r: 녹화 시작/중지",
            "h: 도움말 토글",
            "i: 정보 표시 토글"
        ]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트 표시
        for i, text in enumerate(help_text):
            y_pos = 35 + (i * 25)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _save_frame(self, frame: np.ndarray, result: dict) -> str:
        """프레임 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 원본 프레임 저장
        original_path = output_dir / f"face_recognition_{timestamp}_original.jpg"
        cv2.imwrite(str(original_path), frame)
        
        # 결과 프레임 저장
        result_path = output_dir / f"face_recognition_{timestamp}_result.jpg"
        cv2.imwrite(str(result_path), result['frame_with_results'])
        
        logger.info(f"프레임 저장됨: {original_path}, {result_path}")
        return str(result_path)
    
    def run_realtime(self, source: str = "0") -> bool:
        """실시간 얼굴인식 실행
        
        Args:
            source: 입력 소스 (카메라 ID, 파일 경로, RTSP URL)
            
        Returns:
            실행 성공 여부
        """
        logger.info(f"실시간 얼굴인식 시작 - 입력: {source}")
        
        # 카메라 초기화
        cap = self._create_camera(source)
        
        if not cap.isOpened():
            logger.error(f"카메라 연결 실패: {source}")
            return False
        
        # 실행 상태 초기화
        self.is_running = True
        self.is_paused = False
        frame_count = 0
        
        try:
            while self.is_running:
                if not self.is_paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning("프레임 읽기 실패")
                        break
                    
                    frame_count += 1
                    
                    # 얼굴인식 처리
                    result = self.face_service.process_frame(frame)
                    
                    # 결과 표시
                    display_frame = result['frame_with_results']
                    
                    # 도움말 오버레이
                    if self.show_help:
                        display_frame = self._display_help_overlay(display_frame)
                    
                    # 일시정지 상태 표시
                    if self.is_paused:
                        cv2.putText(display_frame, "PAUSED - Press 'p' to resume", 
                                  (10, display_frame.shape[0] - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('Face Recognition', display_frame)
                
                # 키보드 입력 처리
                action = self._handle_keyboard_input()
                
                if action == 'quit':
                    break
                elif action == 'toggle_pause':
                    self.is_paused = not self.is_paused
                    logger.info(f"일시정지 {'활성화' if self.is_paused else '해제'}")
                elif action == 'save_frame' and not self.is_paused:
                    ret, frame = cap.read()
                    if ret:
                        result = self.face_service.process_frame(frame)
                        saved_path = self._save_frame(frame, result)
                        logger.info(f"프레임 저장: {saved_path}")
                elif action == 'toggle_help':
                    self.show_help = not self.show_help
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        
        except Exception as e:
            logger.error(f"실행 중 오류: {e}")
            return False
        
        finally:
            # 리소스 정리
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            
            logger.info(f"실시간 얼굴인식 종료 - 총 {frame_count}프레임 처리")
            
            # 최종 통계 출력
            service_info = self.face_service.get_service_info()
            stats = service_info['stats']
            logger.info(f"최종 통계: {stats}")
        
        return True
    
    def run_single_image(self, image_path: str) -> bool:
        """단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            처리 성공 여부
        """
        logger.info(f"단일 이미지 처리: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return False
        
        try:
            # 이미지 로드
            frame = cv2.imread(image_path)
            
            if frame is None:
                logger.error(f"이미지 로드 실패: {image_path}")
                return False
            
            # 얼굴인식 처리
            result = self.face_service.process_frame(frame)
            
            # 결과 표시
            cv2.imshow('Face Recognition - Single Image', result['frame_with_results'])
            
            # 결과 저장
            saved_path = self._save_frame(frame, result)
            
            logger.info(f"처리 완료: {len(result['faces'])}개 얼굴 검출")
            logger.info(f"처리 시간: {result['processing_time']*1000:.1f}ms")
            logger.info(f"결과 저장: {saved_path}")
            
            # 키 입력 대기
            print("아무 키나 누르면 종료됩니다...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {e}")
            return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="실시간 얼굴인식")
    parser.add_argument("--source", type=str, default="0", 
                       help="입력 소스 (카메라 ID, 파일 경로, RTSP URL)")
    parser.add_argument("--config", type=str, 
                       help="설정 파일 경로")
    parser.add_argument("--image", type=str, 
                       help="단일 이미지 처리 모드")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 실행기 초기화
        runner = FaceRecognitionRunner(config_path=args.config)
        
        # 실행 모드 선택
        if args.image:
            success = runner.run_single_image(args.image)
        else:
            success = runner.run_realtime(args.source)
        
        if success:
            logger.info("얼굴인식 실행 완료")
            return 0
        else:
            logger.error("얼굴인식 실행 실패")
            return 1
            
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 