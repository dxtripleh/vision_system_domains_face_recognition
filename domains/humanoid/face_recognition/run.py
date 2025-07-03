#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Humanoid Face Recognition Runner.

USB 카메라 또는 이미지 파일을 입력으로 받아 실시간 얼굴인식을 수행합니다.
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

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from .model import FaceRecognitionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "data" / "logs" / "face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognitionRunner:
    def __init__(self, model_path: Optional[str] = None, config: Optional[dict] = None):
        self.model = FaceRecognitionModel(model_path, config)
        self.is_running = False

    def create_camera(self, source: Union[int, str]) -> cv2.VideoCapture:
        import platform
        system = platform.system().lower()
        if isinstance(source, int):
            if system == "windows":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            elif system == "linux":
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
        logger.info(f"카메라 연결 성공: {source}")
        return cap

    def run_camera(self, source: Union[int, str], show_display: bool = True):
        cap = self.create_camera(source)
        try:
            self.is_running = True
            logger.info("실시간 얼굴인식 시작")
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("프레임을 읽을 수 없습니다.")
                    break
                embedding = self.model.infer(frame)
                logger.info(f"임베딩 벡터 shape: {embedding.shape}")
                if show_display:
                    cv2.imshow('Face Recognition', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            logger.info("실시간 얼굴인식 종료")
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()

    def run_image(self, image_path: str, show_display: bool = True):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        logger.info(f"이미지 로드: {image_path}")
        embedding = self.model.infer(image)
        logger.info(f"임베딩 벡터 shape: {embedding.shape}")
        if show_display:
            cv2.imshow('Face Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return embedding

def parse_args():
    parser = argparse.ArgumentParser(description="얼굴인식 실행")
    parser.add_argument("--source", type=str, default="0", help="입력 소스 (카메라 ID, 파일 경로, RTSP URL)")
    parser.add_argument("--model", type=str, help="ONNX 모델 파일 경로")
    parser.add_argument("--show", action="store_true", default=True, help="결과 화면 표시")
    parser.add_argument("--no-display", action="store_true", help="화면 표시 비활성화")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        runner = FaceRecognitionRunner(args.model)
        if args.source.isdigit():
            source = int(args.source)
            runner.run_camera(source, show_display=not args.no_display)
        else:
            runner.run_image(args.source, show_display=not args.no_display)
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 