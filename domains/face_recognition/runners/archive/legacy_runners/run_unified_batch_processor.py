#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 배치 얼굴 처리기

captured와 uploads 폴더의 모든 이미지/동영상에서 얼굴을 추출하여
detected_faces 폴더에 저장합니다.

사용법:
    python run_unified_batch_processor.py [--source captured|uploads|both] [--auto-save]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
from datetime import datetime
import json

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from common.config import load_config

class EnvironmentAnalyzer:
    """환경 분석을 통한 최적 모델 선택"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def analyze_environment(self) -> Dict:
        """환경 분석 및 최적 모델 추천"""
        env_info = {
            'gpu_available': self._check_gpu(),
            'cpu_cores': os.cpu_count(),
            'available_models': self._scan_available_models(),
            'recommended_model': None
        }
        
        # 최적 모델 선택
        env_info['recommended_model'] = self._select_optimal_model(env_info)
        
        return env_info
    
    def _check_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                # OpenCV DNN GPU 지원 확인
                return cv2.cuda.getCudaEnabledDeviceCount() > 0
            except:
                return False
    
    def _scan_available_models(self) -> List[Dict]:
        """사용 가능한 모델 스캔"""
        models_dir = project_root / 'models' / 'weights'
        available_models = []
        
        # OpenCV Haar Cascade
        haar_path = models_dir / 'face_detection_opencv_haarcascade_20250628.xml'
        if haar_path.exists():
            available_models.append({
                'name': 'OpenCV Haar Cascade',
                'type': 'haar',
                'path': str(haar_path),
                'speed': 'fast',
                'accuracy': 'medium',
                'gpu_required': False
            })
        
        # 기본 Haar Cascade (항상 사용 가능)
        available_models.append({
            'name': 'OpenCV Default Haar',
            'type': 'haar_default',
            'path': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'speed': 'fast',
            'accuracy': 'medium',
            'gpu_required': False
        })
        
        # RetinaFace 모델들 스캔
        for model_file in models_dir.glob('*retinaface*.onnx'):
            available_models.append({
                'name': f'RetinaFace ({model_file.stem})',
                'type': 'retinaface',
                'path': str(model_file),
                'speed': 'medium',
                'accuracy': 'high',
                'gpu_required': False
            })
        
        # YOLO 모델들 스캔
        for model_file in models_dir.glob('*yolo*.pt'):
            available_models.append({
                'name': f'YOLO ({model_file.stem})',
                'type': 'yolo',
                'path': str(model_file),
                'speed': 'medium',
                'accuracy': 'high',
                'gpu_required': True
            })
        
        return available_models
    
    def _select_optimal_model(self, env_info: Dict) -> Dict:
        """최적 모델 선택"""
        available_models = env_info['available_models']
        gpu_available = env_info['gpu_available']
        
        # GPU 사용 가능하면 고성능 모델 우선
        if gpu_available:
            for model in available_models:
                if model['accuracy'] == 'high':
                    return model
        
        # CPU만 사용 가능하면 빠른 모델 우선
        for model in available_models:
            if not model['gpu_required'] and model['speed'] == 'fast':
                return model
        
        # 기본값
        return available_models[0] if available_models else None

class FaceDetector:
    """통합 얼굴 검출기"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model_type = model_config['type']
        self.logger = get_logger(__name__)
        
        # 모델별 설정 (더 민감한 파라미터로 조정)
        self.confidence_threshold = 0.3  # 더 낮은 신뢰도 임계값
        self.scale_factor = 1.05  # 더 세밀한 검출
        self.min_neighbors = 2   # 더 민감한 검출 (5에서 2로 변경)
        self.min_size = (20, 20)  # 더 작은 얼굴도 검출
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        if self.model_type in ['haar', 'haar_default']:
            self.detector = cv2.CascadeClassifier(self.model_config['path'])
            if self.detector.empty():
                self.logger.error(f"Haar Cascade 모델 로드 실패: {self.model_config['path']}")
                self._fallback_to_default_haar()
            else:
                self.logger.info(f"Haar Cascade 모델 로드: {self.model_config['name']}")
        
        elif self.model_type == 'retinaface':
            try:
                # RetinaFace ONNX 모델 로드
                import onnxruntime as ort
                
                # ONNX Runtime 세션 생성
                self.ort_session = ort.InferenceSession(
                    self.model_config['path'],
                    providers=['CPUExecutionProvider']  # CPU 사용
                )
                
                # 입력/출력 정보 확인
                self.input_name = self.ort_session.get_inputs()[0].name
                self.input_shape = self.ort_session.get_inputs()[0].shape
                
                self.logger.info(f"RetinaFace ONNX 모델 로드: {self.model_config['name']}")
                self.logger.info(f"입력 형태: {self.input_shape}")
                
            except ImportError:
                self.logger.error("onnxruntime 패키지가 설치되지 않음. pip install onnxruntime")
                self._fallback_to_default_haar()
            except Exception as e:
                self.logger.error(f"RetinaFace 모델 로드 실패: {e}")
                self._fallback_to_default_haar()
        
        elif self.model_type == 'yolo':
            try:
                # YOLO 모델 초기화 (실제 구현 필요)
                # 현재는 구현되지 않았으므로 Haar로 폴백
                self.logger.warning(f"YOLO 모델은 아직 구현되지 않음: {self.model_config['name']}")
                self._fallback_to_default_haar()
            except Exception as e:
                self.logger.error(f"YOLO 모델 로드 실패: {e}")
                self._fallback_to_default_haar()
        else:
            self.logger.error(f"알 수 없는 모델 타입: {self.model_type}")
            self._fallback_to_default_haar()
    
    def _fallback_to_default_haar(self):
        """기본 Haar Cascade로 폴백"""
        self.model_type = 'haar_default'
        self.model_config = {
            'name': 'OpenCV Default Haar (Fallback)',
            'type': 'haar_default',
            'path': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'speed': 'fast',
            'accuracy': 'medium',
            'gpu_required': False
        }
        self.detector = cv2.CascadeClassifier(self.model_config['path'])
        self.logger.warning("기본 Haar Cascade로 폴백")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """얼굴 검출"""
        if self.detector is None:
            self.logger.error("검출기가 초기화되지 않음")
            return []
        
        if self.model_type in ['haar', 'haar_default']:
            return self._detect_haar(image)
        elif self.model_type == 'retinaface':
            return self._detect_retinaface(image)
        elif self.model_type == 'yolo':
            # 현재는 Haar로 폴백되므로 이 코드는 실행되지 않음
            return self._detect_yolo(image)
        else:
            self.logger.error(f"지원하지 않는 모델 타입: {self.model_type}")
            return []
    
    def _detect_haar(self, image: np.ndarray) -> List[Dict]:
        """Haar Cascade 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 평활화로 조명 개선
        gray = cv2.equalizeHist(gray)
        
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar는 신뢰도를 제공하지 않음
                'quality_score': self._calculate_quality_score(x, y, w, h, image.shape)
            })
        
        return detections
    
    def _detect_retinaface(self, image: np.ndarray) -> List[Dict]:
        """RetinaFace ONNX 모델을 이용한 검출"""
        try:
            # 입력 이미지 전처리
            input_size = (640, 640)  # RetinaFace 표준 입력 크기
            
            # 이미지 리사이즈
            resized = cv2.resize(image, input_size)
            
            # BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 정규화 (0-1)
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # HWC to CHW
            input_tensor = np.transpose(normalized, (2, 0, 1))
            
            # 배치 차원 추가
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # 추론 실행
            outputs = self.ort_session.run(None, {self.input_name: input_tensor})
            
            # 출력 후처리
            detections = self._postprocess_retinaface_outputs(outputs, image.shape, input_size)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"RetinaFace 추론 오류: {e}")
            # 오류 시 Haar Cascade 사용
            return self._detect_haar(image)
    
    def _postprocess_retinaface_outputs(self, outputs, original_shape, input_size) -> List[Dict]:
        """RetinaFace 출력 후처리"""
        try:
            detections = []
            
            # RetinaFace 출력 형태에 따라 조정 필요
            # 일반적으로 [boxes, scores, landmarks] 형태
            if len(outputs) >= 2:
                boxes = outputs[0]  # [N, 4]
                scores = outputs[1]  # [N, 1]
                
                # 신뢰도 임계값 적용
                valid_indices = scores[:, 0] > self.confidence_threshold
                
                if np.any(valid_indices):
                    valid_boxes = boxes[valid_indices]
                    valid_scores = scores[valid_indices]
                    
                    # 좌표 스케일링 (입력 크기 -> 원본 크기)
                    scale_x = original_shape[1] / input_size[0]
                    scale_y = original_shape[0] / input_size[1]
                    
                    for box, score in zip(valid_boxes, valid_scores):
                        x1, y1, x2, y2 = box
                        
                        # 스케일링 적용
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # bbox 형태를 (x, y, w, h)로 변환
                        w = x2 - x1
                        h = y2 - y1
                        
                        detections.append({
                            'bbox': (x1, y1, w, h),
                            'confidence': float(score[0]),
                            'quality_score': self._calculate_quality_score(x1, y1, w, h, original_shape)
                        })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"RetinaFace 후처리 오류: {e}")
            return []
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """YOLO 검출 (구현 필요)"""
        # 실제 YOLO 구현
        return self._detect_haar(image)  # 임시로 Haar 사용
    
    def _calculate_quality_score(self, x: int, y: int, w: int, h: int, image_shape: tuple) -> float:
        """얼굴 품질 점수 계산"""
        # 크기 점수
        face_area = w * h
        image_area = image_shape[0] * image_shape[1]
        size_ratio = face_area / image_area if image_area > 0 else 0
        size_score = min(size_ratio * 10, 1.0)
        
        # 위치 점수 (중앙에 가까울수록 높은 점수)
        center_x, center_y = x + w//2, y + h//2
        img_center_x, img_center_y = image_shape[1]//2, image_shape[0]//2
        distance = ((center_x - img_center_x)**2 + (center_y - img_center_y)**2)**0.5
        max_distance = (img_center_x**2 + img_center_y**2)**0.5
        position_score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return (size_score * 0.7 + position_score * 0.3)

class UnifiedBatchProcessor:
    """통합 배치 얼굴 처리기"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 환경 분석 및 최적 모델 선택
        self.env_analyzer = EnvironmentAnalyzer()
        self.env_info = self.env_analyzer.analyze_environment()
        
        # 최적 모델로 검출기 초기화
        optimal_model = self.env_info['recommended_model']
        if optimal_model:
            self.face_detector = FaceDetector(optimal_model)
            self.logger.info(f"최적 모델 선택: {optimal_model['name']}")
        else:
            self.logger.error("사용 가능한 모델이 없습니다")
            raise RuntimeError("사용 가능한 모델이 없습니다")
        
        # 경로 설정
        self.domain_root = project_root / 'data' / 'domains' / 'face_recognition'
        
        self.paths = {
            # 입력 경로
            'raw_captured': self.domain_root / 'raw_input' / 'captured',
            'raw_uploads': self.domain_root / 'raw_input' / 'uploads',
            
            # 출력 경로
            'detected_captured': self.domain_root / 'detected_faces' / 'from_captured',
            'detected_uploads': self.domain_root / 'detected_faces' / 'from_uploads'
        }
        
        # 폴더 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # 통계
        self.stats = {
            'processed_files': 0,
            'detected_faces': 0,
            'saved_faces': 0,
            'errors': 0
        }
    
    def process_batch(self, source: str = 'both'):
        """배치 처리 실행"""
        self.logger.info(f"배치 처리 시작 - 소스: {source}")
        self.logger.info(f"사용 모델: {self.face_detector.model_config['name']}")
        
        if source in ['captured', 'both']:
            self._process_folder('captured')
        
        if source in ['uploads', 'both']:
            self._process_folder('uploads')
        
        self._print_summary()
    
    def _process_folder(self, folder_type: str):
        """폴더 처리"""
        input_path = self.paths[f'raw_{folder_type}']
        output_path = self.paths[f'detected_{folder_type}']
        
        self.logger.info(f"처리 중: {input_path}")
        
        # 지원되는 이미지 확장자 (JSON 제외)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        
        # 파일 목록 가져오기 (JSON 파일 제외)
        files = []
        for file_path in input_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                # JSON 파일과 README 파일 제외
                if ext in image_extensions or ext in video_extensions:
                    files.append(file_path)
        
        if not files:
            self.logger.warning(f"{input_path}에 처리할 파일이 없습니다")
            return
        
        self.logger.info(f"{len(files)}개 파일 발견")
        
        # 중복 처리 방지: 이미 처리된 파일 확인
        processed_files = self._get_processed_files(output_path)
        new_files = []
        skipped_files = []
        
        for file_path in files:
            if self._is_already_processed(file_path, processed_files):
                skipped_files.append(file_path.name)
            else:
                new_files.append(file_path)
        
        if skipped_files:
            self.logger.info(f"이미 처리된 파일 {len(skipped_files)}개 건너뜀: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")
        
        if not new_files:
            self.logger.info("새로 처리할 파일이 없습니다")
            return
        
        self.logger.info(f"새로 처리할 파일: {len(new_files)}개")
        
        for file_path in new_files:
            try:
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    self._process_image(file_path, output_path)
                elif ext in video_extensions:
                    self._process_video(file_path, output_path)
                    
                self.stats['processed_files'] += 1
                
            except Exception as e:
                self.logger.error(f"파일 처리 실패 {file_path}: {e}")
                self.stats['errors'] += 1
    
    def _get_processed_files(self, output_path: Path) -> Dict[str, Dict]:
        """이미 처리된 파일들의 메타데이터 수집"""
        processed_files = {}
        
        if not output_path.exists():
            return processed_files
        
        # JSON 메타데이터 파일들 수집
        for json_file in output_path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                source_file = metadata.get('source_file', '')
                if source_file:
                    # 절대 경로로 정규화
                    source_path = Path(source_file)
                    if source_path.exists():
                        # 파일 크기와 수정 시간으로 변경 감지
                        stat = source_path.stat()
                        processed_files[str(source_path)] = {
                            'metadata': metadata,
                            'file_size': stat.st_size,
                            'mtime': stat.st_mtime
                        }
            except Exception as e:
                self.logger.debug(f"메타데이터 로드 실패 {json_file}: {e}")
        
        return processed_files
    
    def _is_already_processed(self, file_path: Path, processed_files: Dict[str, Dict]) -> bool:
        """파일이 이미 처리되었는지 확인"""
        file_key = str(file_path)
        
        if file_key not in processed_files:
            return False
        
        # 파일 크기와 수정 시간 확인
        current_stat = file_path.stat()
        processed_info = processed_files[file_key]
        
        # 파일이 변경되었는지 확인 (크기나 수정 시간이 다르면 재처리)
        if (current_stat.st_size != processed_info['file_size'] or 
            current_stat.st_mtime != processed_info['mtime']):
            return False
        
        return True
    
    def _process_image(self, image_path: Path, output_path: Path):
        """이미지 처리"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        # 얼굴 검출
        detections = self.face_detector.detect_faces(image)
        
        if not detections:
            self.logger.debug(f"얼굴 없음: {image_path.name}")
            return
        
        self.stats['detected_faces'] += len(detections)
        
        # 얼굴 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        base_name = image_path.stem
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # 얼굴 영역 추출
            face_crop = image[y:y+h, x:x+w]
            
            # 파일명 생성
            face_filename = f"face_{base_name}_{timestamp}_{i:02d}_conf{detection['confidence']:.2f}.jpg"
            face_path = output_path / face_filename
            
            # 얼굴 저장
            cv2.imwrite(str(face_path), face_crop)
            
            # 메타데이터 저장
            metadata = {
                'timestamp': timestamp,
                'source_file': str(image_path),
                'source_type': 'image',
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(detection['confidence']),
                'quality_score': float(detection['quality_score']),
                'face_path': str(face_path)
            }
            
            metadata_path = face_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.stats['saved_faces'] += 1
        
        self.logger.info(f"처리 완료: {image_path.name} ({len(detections)}개 얼굴)")
    
    def _process_video(self, video_path: Path, output_path: Path):
        """비디오 처리"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"비디오 로드 실패: {video_path}")
        
        frame_count = 0
        detected_count = 0
        
        # 비디오에서 프레임 추출 (1초에 1프레임)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps)) if fps > 0 else 30
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        base_name = video_path.stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 간격에 따라 처리
            if frame_count % frame_interval == 0:
                detections = self.face_detector.detect_faces(frame)
                
                if detections:
                    self.stats['detected_faces'] += len(detections)
                    
                    for i, detection in enumerate(detections):
                        bbox = detection['bbox']
                        x, y, w, h = bbox
                        
                        # 얼굴 영역 추출
                        face_crop = frame[y:y+h, x:x+w]
                        
                        # 파일명 생성
                        face_filename = f"face_{base_name}_frame{frame_count:06d}_{timestamp}_{i:02d}_conf{detection['confidence']:.2f}.jpg"
                        face_path = output_path / face_filename
                        
                        # 얼굴 저장
                        cv2.imwrite(str(face_path), face_crop)
                        
                        # 메타데이터 저장
                        metadata = {
                            'timestamp': timestamp,
                            'source_file': str(video_path),
                            'source_type': 'video',
                            'frame_number': frame_count,
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'confidence': float(detection['confidence']),
                            'quality_score': float(detection['quality_score']),
                            'face_path': str(face_path)
                        }
                        
                        metadata_path = face_path.with_suffix('.json')
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        self.stats['saved_faces'] += 1
                        detected_count += 1
            
            frame_count += 1
        
        cap.release()
        self.logger.info(f"처리 완료: {video_path.name} ({detected_count}개 얼굴, {frame_count}프레임)")
    
    def _print_summary(self):
        """처리 결과 요약"""
        print("\n" + "="*60)
        print("📊 배치 처리 결과 요약")
        print("="*60)
        print(f"🤖 사용 모델: {self.face_detector.model_config['name']}")
        print(f"📁 처리된 파일: {self.stats['processed_files']}개")
        print(f"👤 검출된 얼굴: {self.stats['detected_faces']}개")
        print(f"💾 저장된 얼굴: {self.stats['saved_faces']}개")
        print(f"❌ 오류: {self.stats['errors']}개")
        print("="*60)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 배치 얼굴 처리기")
    parser.add_argument(
        "--source", 
        choices=['captured', 'uploads', 'both'], 
        default='both',
        help="처리할 소스 폴더 (기본값: both)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="설정 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # 설정 로드
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # 배치 처리기 초기화 및 실행
        processor = UnifiedBatchProcessor(config)
        processor.process_batch(args.source)
        
        logger.info("배치 처리 완료")
        
    except Exception as e:
        logger.error(f"배치 처리 실패: {e}")
        print(f"❌ 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 