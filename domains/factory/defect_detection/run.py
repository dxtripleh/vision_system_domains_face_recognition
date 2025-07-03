#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Factory Defect Detection Runner.

USB 카메라 또는 이미지 파일을 입력으로 받아 실시간 불량 검출을 수행합니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from .model import DefectDetectionModel
from . import DEFECT_TYPES

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "data" / "logs" / "defect_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DefectDetectionRunner:
    """불량 검출 실행기 클래스."""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[dict] = None):
        """
        실행기 초기화.
        
        Args:
            model_path: ONNX 모델 파일 경로
            config: 설정 딕셔너리
        """
        self.model = DefectDetectionModel(model_path, config)
        self.is_running = False
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        # 성능 통계
        self.stats = {
            'total_frames': 0,
            'total_defects': 0,
            'avg_inference_time': 0.0,
            'fps': 0.0
        }
    
    def create_camera(self, source: Union[int, str]) -> cv2.VideoCapture:
        """
        플랫폼별 최적화된 카메라를 생성합니다.
        
        Args:
            source: 카메라 ID 또는 파일 경로
            
        Returns:
            OpenCV VideoCapture 객체
        """
        import platform
        
        system = platform.system().lower()
        
        if isinstance(source, int):
            # USB 카메라
            if system == "windows":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            elif system == "linux":
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                # Jetson 환경에서 버퍼 최소화
                if self._is_jetson_platform():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(source)
        else:
            # 파일 또는 RTSP 스트림
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
        
        logger.info(f"카메라 연결 성공: {source}")
        return cap
    
    def _is_jetson_platform(self) -> bool:
        """Jetson 플랫폼인지 확인합니다."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        검출 결과를 이미지에 그립니다.
        
        Args:
            image: 원본 이미지
            detections: 검출 결과 리스트
            
        Returns:
            시각화된 이미지
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # 바운딩 박스 그리기
            x1, y1, x2, y2 = bbox
            color = DEFECT_TYPES.get(list(DEFECT_TYPES.keys())[class_id], {}).get('color', [0, 255, 0])
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 그리기
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 라벨 배경
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 라벨 텍스트
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def draw_stats(self, image: np.ndarray) -> np.ndarray:
        """
        성능 통계를 이미지에 그립니다.
        
        Args:
            image: 원본 이미지
            
        Returns:
            통계가 추가된 이미지
        """
        result_image = image.copy()
        
        # 통계 정보
        stats_text = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Total Frames: {self.stats['total_frames']}",
            f"Total Defects: {self.stats['total_defects']}",
            f"Avg Inference: {self.stats['avg_inference_time']*1000:.1f}ms"
        ]
        
        # 배경 박스
        text_height = 25
        box_height = len(stats_text) * text_height + 10
        cv2.rectangle(result_image, (10, 10), (300, box_height), (0, 0, 0), -1)
        cv2.rectangle(result_image, (10, 10), (300, box_height), (255, 255, 255), 2)
        
        # 텍스트 그리기
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * text_height
            cv2.putText(result_image, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def update_stats(self, inference_time: float, detections: list):
        """성능 통계를 업데이트합니다."""
        self.stats['total_frames'] += 1
        self.stats['total_defects'] += len(detections)
        self.total_inference_time += inference_time
        self.stats['avg_inference_time'] = self.total_inference_time / self.stats['total_frames']
        
        # FPS 계산 (최근 30프레임 기준)
        if self.stats['total_frames'] % 30 == 0:
            self.stats['fps'] = 30.0 / (time.time() - getattr(self, '_last_fps_time', time.time()))
            self._last_fps_time = time.time()
    
    def log_detection_results(self, detections: list, inference_time: float):
        """검출 결과를 로그에 기록합니다."""
        if detections:
            defect_counts = {}
            for detection in detections:
                class_name = detection['class_name']
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
            
            logger.info(f"검출 결과: {defect_counts}, 추론시간: {inference_time*1000:.1f}ms")
        else:
            logger.debug(f"불량 미검출, 추론시간: {inference_time*1000:.1f}ms")
    
    def run_camera(self, source: Union[int, str], show_display: bool = True, save_output: bool = False):
        """
        카메라를 사용한 실시간 불량 검출을 실행합니다.
        
        Args:
            source: 카메라 ID 또는 파일 경로
            show_display: 화면 표시 여부
            save_output: 결과 저장 여부
        """
        cap = self.create_camera(source)
        
        # 비디오 라이터 설정
        video_writer = None
        if save_output:
            output_path = project_root / "data" / "output" / f"defect_detection_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            self.is_running = True
            logger.info("실시간 불량 검출 시작")
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("프레임을 읽을 수 없습니다.")
                    break
                
                # 불량 검출
                detections, inference_time = self.model.predict(frame)
                
                # 통계 업데이트
                self.update_stats(inference_time, detections)
                
                # 결과 시각화
                result_frame = self.draw_detections(frame, detections)
                result_frame = self.draw_stats(result_frame)
                
                # 로그 기록
                self.log_detection_results(detections, inference_time)
                
                # 화면 표시
                if show_display:
                    cv2.imshow('Factory Defect Detection', result_frame)
                    
                    # 키보드 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # 현재 프레임 저장
                        timestamp = int(time.time())
                        save_path = project_root / "data" / "output" / f"defect_frame_{timestamp}.jpg"
                        cv2.imwrite(str(save_path), result_frame)
                        logger.info(f"프레임 저장: {save_path}")
                
                # 비디오 저장
                if save_output and video_writer:
                    video_writer.write(result_frame)
        
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"실행 중 오류 발생: {e}")
            raise
        finally:
            self.is_running = False
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info("실시간 불량 검출 종료")
    
    def run_image(self, image_path: str, show_display: bool = True, save_output: bool = False):
        """
        이미지 파일에서 불량 검출을 실행합니다.
        
        Args:
            image_path: 이미지 파일 경로
            show_display: 화면 표시 여부
            save_output: 결과 저장 여부
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        logger.info(f"이미지 로드: {image_path}")
        
        # 불량 검출
        detections, inference_time = self.model.predict(image)
        
        # 결과 시각화
        result_image = self.draw_detections(image, detections)
        
        # 통계 정보 추가
        self.stats['total_frames'] = 1
        self.stats['total_defects'] = len(detections)
        self.stats['avg_inference_time'] = inference_time
        result_image = self.draw_stats(result_image)
        
        # 로그 기록
        self.log_detection_results(detections, inference_time)
        
        # 화면 표시
        if show_display:
            cv2.imshow('Factory Defect Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 결과 저장
        if save_output:
            output_path = project_root / "data" / "output" / f"defect_result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"결과 저장: {output_path}")
        
        return detections, inference_time

def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="공장 불량 검출 실행")
    parser.add_argument("--source", type=str, default="0", 
                       help="입력 소스 (카메라 ID, 파일 경로, RTSP URL)")
    parser.add_argument("--model", type=str, 
                       help="ONNX 모델 파일 경로")
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="신뢰도 임계값")
    parser.add_argument("--show", action="store_true", default=True,
                       help="결과 화면 표시")
    parser.add_argument("--save", action="store_true", 
                       help="결과 저장")
    parser.add_argument("--no-display", action="store_true",
                       help="화면 표시 비활성화")
    
    return parser.parse_args()

def main():
    """메인 함수."""
    args = parse_args()
    
    # 설정
    config = {
        'confidence_threshold': args.conf,
        'nms_threshold': 0.4,
        'input_size': (640, 640),
        'max_detections': 100
    }
    
    try:
        # 실행기 초기화
        runner = DefectDetectionRunner(args.model, config)
        
        # 입력 소스 확인
        if args.source.isdigit():
            # 카메라
            source = int(args.source)
            runner.run_camera(source, show_display=not args.no_display, save_output=args.save)
        else:
            # 이미지 파일
            runner.run_image(args.source, show_display=not args.no_display, save_output=args.save)
    
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 