#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 얼굴 캡처 시스템

새로운 도메인 구조를 사용하고 모든 요청사항을 구현:
1. i키로 정보 토글 (같은 창 사용)
2. 자동 모드에서 얼굴 자동 저장 (각도/조명 변화 감지)
3. 명확한 데이터 플로우
4. 체계적인 폴더 구조
5. 환경 분석을 통한 최적 모델 선택
6. GUI 이름 입력
7. 모델 설정 실시간 조절
"""

import os
import sys
import cv2
import time
import json
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading

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
        yunet_path = models_dir / 'face_detection_yunet_2023mar.onnx'
        print('[DEBUG] YuNet ONNX exists:', yunet_path.exists(), yunet_path)
        if yunet_path.exists():
            available_models.append({
                'name': 'YuNet',
                'type': 'yunet',
                'path': str(yunet_path),
                'speed': 'fast',
                'accuracy': 'high',
                'gpu_required': False,
                'supports_landmarks': True
            })
        print('[DEBUG] Available models:', available_models)
        return available_models
    
    def _select_optimal_model(self, env_info: Dict) -> Dict:
        """최적 모델 선택"""
        available_models = env_info['available_models']
        print('[DEBUG] Selecting model from:', available_models)
        if available_models:
            print('[DEBUG] Selected model:', available_models[0])
            return available_models[0]
        return None

class FaceDetector:
    """통합 얼굴 검출기"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model_type = model_config['type']
        self.logger = get_logger(__name__)
        
        # 모델별 설정
        self.confidence_threshold = 0.5
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        print('[DEBUG] Initializing model:', self.model_config)
        try:
            if self.model_type == 'yunet':
                self.detector = cv2.dnn.readNetFromONNX(self.model_config['path'])
                print('[DEBUG] YuNet loaded:', not self.detector.empty())
                return not self.detector.empty()
            else:
                print('[DEBUG] Unknown model type:', self.model_type)
                return False
        except Exception as e:
            print('[DEBUG] Model init error:', e)
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """얼굴 검출"""
        if self.detector is None:
            self.logger.error("검출기가 초기화되지 않음")
            return []
        
        try:
            if self.model_type == 'yunet':
                return self._detect_yunet(image)
            else:
                self.logger.error(f"지원하지 않는 모델 타입: {self.model_type}")
                return []
        except Exception as e:
            self.logger.error(f"검출 중 오류: {e}")
            return []
    
    def _detect_yunet(self, image: np.ndarray) -> List[Dict]:
        """YuNet 검출 - ONNX Runtime 버전"""
        import onnxruntime as ort
        
        h, w = image.shape[:2]
        
        # ONNX Runtime 세션 생성 (한 번만)
        if not hasattr(self, 'onnx_session'):
            try:
                self.onnx_session = ort.InferenceSession(self.model_config['path'])
                print('[DEBUG] YuNet ONNX Runtime session created')
            except Exception as e:
                print(f'[DEBUG] YuNet ONNX Runtime session error: {e}')
                return []
        
        # 입력 전처리
        input_size = (640, 640)
        blob = cv2.dnn.blobFromImage(image, 1.0, input_size, (104, 117, 123), swapRB=True, crop=False)
        
        try:
            # ONNX Runtime으로 추론 - 모든 출력 가져오기
            input_name = self.onnx_session.get_inputs()[0].name
            output_names = [output.name for output in self.onnx_session.get_outputs()]
            
            print(f'[DEBUG] YuNet input name: {input_name}')
            print(f'[DEBUG] YuNet output names: {output_names}')
            
            outputs = self.onnx_session.run(output_names, {input_name: blob})
            
            print('[DEBUG] YuNet ONNX Runtime outputs:')
            for i, (name, output) in enumerate(zip(output_names, outputs)):
                print(f'  {name}: shape={output.shape}, sample={output.flatten()[:5]}')
            
            # 첫 번째 출력을 기본으로 사용
            outputs = outputs[0]
        except Exception as e:
            print(f'[DEBUG] YuNet ONNX Runtime inference error: {e}')
            return []
        
        detections = []
        
        # 출력 형태 확인 및 처리
        if len(outputs.shape) == 3:
            detections_data = outputs[0] if outputs.shape[0] == 1 else outputs
        elif len(outputs.shape) == 2:
            detections_data = outputs
        else:
            print('[DEBUG] Unexpected YuNet output shape:', outputs.shape)
            return []
        
        for i, detection in enumerate(detections_data):
            if i < 2:
                print(f'[DEBUG] detection[{i}]:', detection)
            
            # 출력 길이에 따른 처리
            if len(detection) < 5:
                continue
                
            # 다양한 출력 구조 시도
            confidence = None
            x1, y1, x2, y2 = 0, 0, 0, 0
            
            # 구조 1: [x, y, w, h, confidence, ...]
            if len(detection) >= 5:
                try:
                    x = float(detection[0])
                    y = float(detection[1])
                    width = float(detection[2])
                    height = float(detection[3])
                    confidence = float(detection[4])
                    
                    # 좌표를 이미지 크기에 맞게 스케일링
                    x1 = int(x * w)
                    y1 = int(y * h)
                    x2 = int((x + width) * w)
                    y2 = int((y + height) * h)
                except:
                    pass
            
            # 구조 2: [x1, y1, x2, y2, confidence, ...]
            if confidence is None and len(detection) >= 5:
                try:
                    x1 = int(float(detection[0]) * w)
                    y1 = int(float(detection[1]) * h)
                    x2 = int(float(detection[2]) * w)
                    y2 = int(float(detection[3]) * h)
                    confidence = float(detection[4])
                except:
                    pass
            
            if confidence is None:
                continue
                
            # 좌표 검증
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            
            # 랜드마크 추출 (가능한 경우)
            landmarks = []
            if len(detection) >= 15:
                try:
                    for j in range(5):
                        lx = int(detection[5 + j * 2] * w)
                        ly = int(detection[6 + j * 2] * h)
                        landmarks.append((lx, ly))
                except:
                    pass
            
            if i < 2:
                print(f'[DEBUG] detection[{i}] 파싱 결과:')
                print(f'  confidence={confidence:.3f}')
                print(f'  bbox=({x1}, {y1}, {x2-x1}, {y2-y1})')
                print(f'  landmarks={landmarks}')
            
            # 신뢰도 필터링
            if confidence > self.confidence_threshold:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'landmarks': landmarks,
                    'quality_score': self._calculate_quality_score(x1, y1, x2 - x1, y2 - y1, image.shape)
                })
        
        return detections
    
    def _generate_haar_landmarks(self, x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        """Haar Cascade용 랜드마크 생성 (5점)"""
        landmarks = []
        # 왼쪽 눈
        landmarks.append((x + int(w * 0.3), y + int(h * 0.35)))
        # 오른쪽 눈
        landmarks.append((x + int(w * 0.7), y + int(h * 0.35)))
        # 코
        landmarks.append((x + int(w * 0.5), y + int(h * 0.5)))
        # 왼쪽 입꼬리
        landmarks.append((x + int(w * 0.3), y + int(h * 0.7)))
        # 오른쪽 입꼬리
        landmarks.append((x + int(w * 0.7), y + int(h * 0.7)))
        return landmarks
    
    def _generate_ultraface_landmarks(self, x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        """UltraFace용 랜드마크 생성 (5점)"""
        landmarks = []
        # 왼쪽 눈
        landmarks.append((x + int(w * 0.3), y + int(h * 0.35)))
        # 오른쪽 눈
        landmarks.append((x + int(w * 0.7), y + int(h * 0.35)))
        # 코
        landmarks.append((x + int(w * 0.5), y + int(h * 0.5)))
        # 왼쪽 입꼬리
        landmarks.append((x + int(w * 0.3), y + int(h * 0.7)))
        # 오른쪽 입꼬리
        landmarks.append((x + int(w * 0.7), y + int(h * 0.7)))
        return landmarks
    
    def _calculate_quality_score(self, x: int, y: int, w: int, h: int, image_shape: Tuple) -> float:
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
    
    def adjust_confidence_threshold(self, delta: float):
        """신뢰도 임계값 조절"""
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold + delta))
        self.logger.info(f"신뢰도 임계값: {self.confidence_threshold:.2f}")
    
    def adjust_scale_factor(self, delta: float):
        """스케일 팩터 조절 (Haar Cascade용)"""
        if self.model_type in ['haar', 'haar_default']:
            self.scale_factor = max(1.05, min(2.0, self.scale_factor + delta))
            self.logger.info(f"스케일 팩터: {self.scale_factor:.2f}")
    
    def adjust_min_neighbors(self, delta: int):
        """최소 이웃 수 조절 (Haar Cascade용)"""
        if self.model_type in ['haar', 'haar_default']:
            self.min_neighbors = max(1, min(20, self.min_neighbors + delta))
            self.logger.info(f"최소 이웃 수: {self.min_neighbors}")

class FaceQualityAnalyzer:
    """얼굴 품질 분석기 (자동 모드용)"""
    
    def __init__(self):
        self.face_history = []
        self.similarity_threshold = 0.8
        self.quality_threshold = 0.6
        
    def should_save_face(self, face_data: Dict, current_faces: List[Dict]) -> bool:
        """얼굴 저장 여부 결정"""
        # 품질 점수 확인
        if face_data['quality_score'] < self.quality_threshold:
            return False
        
        # 기존 저장된 얼굴과 비교
        for saved_face in self.face_history[-10:]:  # 최근 10개만 비교
            if self._calculate_similarity(face_data, saved_face) > self.similarity_threshold:
                # 유사한 얼굴이 있지만 품질이 더 좋으면 저장
                if face_data['quality_score'] > saved_face['quality_score'] + 0.1:
                    return True
                return False
        
        return True
    
    def _calculate_similarity(self, face1: Dict, face2: Dict) -> float:
        """얼굴 유사도 계산 (간단한 바운딩 박스 기반)"""
        bbox1 = face1['bbox']
        bbox2 = face2['bbox']
        
        # IoU 계산
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def add_saved_face(self, face_data: Dict):
        """저장된 얼굴 기록"""
        self.face_history.append(face_data)
        if len(self.face_history) > 50:  # 최대 50개 기록
            self.face_history.pop(0)

class NameInputDialog:
    """GUI 이름 입력 대화상자"""
    
    @staticmethod
    def get_person_name(face_count: int = 1) -> Optional[str]:
        """이름 입력 받기"""
        try:
            root = tk.Tk()
            root.withdraw()  # 메인 창 숨기기
            
            title = f"{face_count}개 얼굴의 인물 이름 입력"
            prompt = "인물 이름을 입력하세요 (취소하면 건너뛰기):"
            
            name = simpledialog.askstring(title, prompt)
            root.destroy()
            
            return name.strip() if name else None
            
        except Exception as e:
            print(f"GUI 입력 오류: {e}")
            return None

class EnhancedFaceCaptureSystem:
    """개선된 얼굴 캡처 시스템"""
    
    def __init__(self, camera_id: int = 0):
        # 로깅 설정
        setup_logging()
        self.logger = get_logger(__name__)
        
        # 환경 분석 및 최적 모델 선택
        self.env_analyzer = EnvironmentAnalyzer()
        self.env_info = self.env_analyzer.analyze_environment()
        
        # 카메라 설정
        self.camera_id = camera_id
        self.cap = None
        
        # 모델 관리
        self.available_models = self.env_info['available_models']
        self.current_model_index = 0
        self.current_model = self.env_info['recommended_model']
        if self.current_model:
            for i, model in enumerate(self.available_models):
                if model['name'] == self.current_model['name']:
                    self.current_model_index = i
                    break
        
        # 얼굴 검출기 초기화
        self.face_detector = FaceDetector(self.current_model)
        if not self.face_detector._initialize_model():
            print(f"❌ 모델 초기화 실패: {self.current_model['name']}")
            # 다음 모델로 시도
            for i, model in enumerate(self.available_models):
                if i != self.current_model_index:
                    print(f"🔄 다른 모델 시도: {model['name']}")
                    self.current_model = model
                    self.current_model_index = i
                    self.face_detector = FaceDetector(self.current_model)
                    if self.face_detector._initialize_model():
                        print(f"✅ 모델 초기화 성공: {model['name']}")
                        break
                    else:
                        print(f"❌ 모델 초기화 실패: {model['name']}")
            else:
                print("❌ 모든 모델 초기화 실패")
                return
        
        # 얼굴 품질 분석기
        self.quality_analyzer = FaceQualityAnalyzer()
        
        # 경로 설정
        self.domain_root = project_root / 'data' / 'domains' / 'face_recognition'
        
        self.paths = {
            # 입력 경로 (s키용)
            'raw_captured': self.domain_root / 'raw_input' / 'captured',
            
            # 검출된 얼굴 저장 경로
            'detected_manual': self.domain_root / 'detected_faces' / 'from_manual',
            'detected_auto': self.domain_root / 'detected_faces' / 'auto_collected',
            
            # 스테이징 경로 (c키 + 이름 입력용)
            'staging_named': self.domain_root / 'staging' / 'named'
        }
        
        # 폴더 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # 모드 설정
        self.is_auto_mode = False  # 기본값: 수동 모드
        self.show_info = True
        self.show_landmarks = False  # 기본값: 랜드마크 표시 안함
        self.auto_save_enabled = True  # 자동 모드에서 자동 저장 활성화
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # 세션 통계
        self.session_stats = {
            'auto_saved': 0,
            'manual_captured': 0,
            'named_saved': 0
        }
        
        self.logger.info("✅ 개선된 얼굴 캡처 시스템 초기화 완료")
        self.logger.info(f"📁 도메인 루트: {self.domain_root}")
        self.logger.info(f"🤖 선택된 모델: {self.current_model['name'] if self.current_model else 'None'}")
    
    def start_capture(self):
        """캡처 시작"""
        self.logger.info("🚀 개선된 얼굴 캡처 시스템 시작")
        print("="*60)
        
        # 환경 정보 출력
        self._print_environment_info()
        
        # 카메라 초기화
        if not self._initialize_camera():
            return False
        
        print("\n📹 카메라 시작됨. 키보드 명령어:")
        self._print_help()
        
        try:
            # 메인 캡처 루프
            self._capture_loop()
            
        except KeyboardInterrupt:
            print("\n⚠️  사용자가 중단했습니다.")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            self.logger.error(f"캡처 루프 오류: {e}")
        finally:
            self._cleanup()
        
        return True
    
    def _print_environment_info(self):
        """환경 정보 출력"""
        print(f"🔍 환경 정보:")
        print(f"   카메라 ID: {self.camera_id}")
        print(f"   검출 모델: {self.current_model['name'] if self.current_model else 'None'}")
        print(f"   GPU 사용 가능: {'YES' if self.env_info['gpu_available'] else 'NO'}")
        print(f"   CPU 코어 수: {self.env_info['cpu_cores']}")
        print(f"   사용 가능 모델 수: {len(self.available_models)}")
        print(f"   모드: {'🤖 자동' if self.is_auto_mode else '👤 수동'}")
        print(f"   정보 표시: {'ON' if self.show_info else 'OFF'}")
        print(f"   랜드마크 표시: {'ON' if self.show_landmarks else 'OFF'}")
        print(f"   자동 저장: {'ON' if self.auto_save_enabled else 'OFF'}")
    
    def _initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"❌ 카메라 {self.camera_id} 연결 실패")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 테스트 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 카메라에서 프레임을 읽을 수 없습니다")
                return False
            
            print(f"✅ 카메라 {self.camera_id} 연결 성공 ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {str(e)}")
            return False
    
    def _capture_loop(self):
        """메인 캡처 루프"""
        window_name = "Enhanced Face Capture System"  # 고정된 창 이름
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                break
            
            # 얼굴 검출
            detections = self.face_detector.detect_faces(frame)
            
            # 🤖 자동 모드에서 자동 저장 처리
            if self.is_auto_mode and self.auto_save_enabled:
                self._handle_auto_save(frame, detections)
            
            # 검출 결과 시각화
            display_frame = self._visualize_detections(frame, detections)
            
            # 정보 오버레이
            if self.show_info:
                display_frame = self._draw_info_overlay(display_frame, detections)
            
            # 프레임 표시
            cv2.imshow(window_name, display_frame)
            
            # FPS 업데이트
            self._update_fps()
            
            # 키보드 입력 처리
            action = self._handle_keyboard_input(cv2.waitKey(1) & 0xFF, frame, detections)
            if action == 'quit':
                break
            elif action == 'save_frame':
                self._save_frame_manual(frame, detections)
            elif action == 'capture_face':
                self._capture_face_with_name(frame, detections)
            elif action == 'toggle_mode':
                self._toggle_auto_manual_mode()
            elif action == 'toggle_info':
                self.show_info = not self.show_info
            elif action == 'increase_threshold':
                self.face_detector.adjust_confidence_threshold(0.05)
            elif action == 'decrease_threshold':
                self.face_detector.adjust_confidence_threshold(-0.05)
            elif action == 'increase_scale':
                self.face_detector.adjust_scale_factor(0.05)
            elif action == 'decrease_scale':
                self.face_detector.adjust_scale_factor(-0.05)
            elif action == 'increase_neighbors':
                self.face_detector.adjust_min_neighbors(1)
            elif action == 'decrease_neighbors':
                self.face_detector.adjust_min_neighbors(-1)
            elif action == 'next_model':
                self._switch_to_next_model()
            elif action == 'prev_model':
                self._switch_to_previous_model()
            elif action == 'help':
                self._print_help()
    
    def _handle_auto_save(self, frame: np.ndarray, detections: List[Dict]):
        """자동 모드에서 얼굴 자동 저장"""
        for detection in detections:
            if self.quality_analyzer.should_save_face(detection, detections):
                self._auto_save_detected_face(frame, detection)
                self.quality_analyzer.add_saved_face(detection)
    
    def _save_full_frame(self, frame: np.ndarray):
        """전체 프레임 저장 (s키 전용)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # raw_input/captured에 전체 프레임 저장
        frame_filename = f"captured_frame_{timestamp}.jpg"
        frame_path = self.paths['raw_captured'] / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # 메타데이터 저장
        metadata = {
            'timestamp': timestamp,
            'capture_type': 'manual_frame',
            'frame_path': str(frame_path),
            'frame_size': list(frame.shape)
        }
        
        metadata_path = frame_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 전체 프레임 저장: {frame_filename}")
        self.logger.info(f"Full frame saved: {frame_filename}")
    
    def _handle_manual_face_capture(self, frame: np.ndarray, detections: List[Dict]):
        """수동 얼굴 캡처 (c키)"""
        if not detections:
            print("❌ 검출된 얼굴이 없습니다")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # 1단계: detected_faces/from_manual에 얼굴 저장
        saved_faces = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w]
            
            face_filename = f"manual_face_{timestamp}_{i:02d}_conf{detection['confidence']:.2f}.jpg"
            face_path = self.paths['detected_manual'] / face_filename
            cv2.imwrite(str(face_path), face_crop)
            
            # 메타데이터 저장
            metadata = {
                'timestamp': timestamp,
                'capture_type': 'manual',
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(detection['confidence']),
                'quality_score': float(detection['quality_score']),
                'face_path': str(face_path)
            }
            
            metadata_path = face_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            saved_faces.append({
                'face_path': face_path,
                'face_crop': face_crop,
                'metadata': metadata
            })
        
        print(f"✅ {len(saved_faces)}개 얼굴을 from_manual에 저장")
        
        # 2단계: GUI로 이름 입력 후 staging/named에 저장
        self._prompt_name_and_save_to_staging(saved_faces)
    
    def _prompt_name_and_save_to_staging(self, saved_faces: List[Dict]):
        """GUI로 이름 입력 후 staging/named에 저장"""
        try:
            # tkinter 창을 별도 스레드에서 실행
            def get_name():
                root = tk.Tk()
                root.withdraw()  # 메인 창 숨기기
                root.attributes('-topmost', True)  # 항상 위에 표시
                
                name = simpledialog.askstring(
                    "이름 입력", 
                    f"{len(saved_faces)}개 얼굴의 이름을 입력하세요:",
                    parent=root
                )
                root.destroy()
                return name
            
            # 이름 입력 받기
            person_name = get_name()
            
            if not person_name or not person_name.strip():
                print("❌ 이름이 입력되지 않아 staging 저장을 취소합니다")
                return
            
            person_name = person_name.strip()
            
            # staging/named/{person_name} 폴더 생성
            person_dir = self.paths['staging_named'] / person_name
            person_dir.mkdir(exist_ok=True)
            
            # 각 얼굴을 person 폴더에 저장
            for i, face_data in enumerate(saved_faces):
                face_crop = face_data['face_crop']
                original_metadata = face_data['metadata']
                
                # 새 파일명 생성
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                staged_filename = f"{person_name}_{timestamp}_{i:02d}.jpg"
                staged_path = person_dir / staged_filename
                
                # 얼굴 이미지 저장
                cv2.imwrite(str(staged_path), face_crop)
                
                # staging 메타데이터 생성
                staged_metadata = {
                    'person_name': person_name,
                    'staged_at': timestamp,
                    'original_face_path': str(face_data['face_path']),
                    'staged_face_path': str(staged_path),
                    'original_metadata': original_metadata
                }
                
                # 메타데이터 저장
                staged_metadata_path = staged_path.with_suffix('.json')
                with open(staged_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(staged_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(saved_faces)}개 얼굴을 staging/named/{person_name}에 저장")
            self.session_stats['named_saved'] += len(saved_faces)
            self.logger.info(f"Manual capture with name: {person_name} ({len(saved_faces)} faces)")
            
        except Exception as e:
            print(f"❌ 이름 입력 중 오류: {e}")
            self.logger.error(f"Name input error: {e}")
    
    def _delete_unwanted_faces(self):
        """불필요한 얼굴 삭제 기능 (별도 스레드로 실행)"""
        def delete_in_thread():
            try:
                # 삭제 가능한 폴더들
                deletable_folders = {
                    'detected_manual': self.paths['detected_manual'],
                    'detected_auto': self.paths['detected_auto'],
                    'staging_named': self.paths['staging_named']
                }
                
                print("\n🗑️  얼굴 삭제 모드 (별도 창에서 실행)")
                print("삭제 가능한 폴더:")
                for i, (name, path) in enumerate(deletable_folders.items(), 1):
                    file_count = len(list(path.glob('*.jpg')))
                    print(f"  {i}. {name} ({file_count}개 파일)")
                
                print("  0. 취소")
                
                choice = input("폴더 선택 (번호): ").strip()
                
                if choice == '0':
                    print("❌ 삭제 취소")
                    return
                
                try:
                    folder_idx = int(choice) - 1
                    folder_names = list(deletable_folders.keys())
                    
                    if 0 <= folder_idx < len(folder_names):
                        selected_folder = folder_names[folder_idx]
                        selected_path = deletable_folders[selected_folder]
                        
                        self._delete_faces_in_folder(selected_folder, selected_path)
                    else:
                        print("❌ 잘못된 선택")
                
                except ValueError:
                    print("❌ 숫자를 입력하세요")
            
            except Exception as e:
                print(f"❌ 삭제 모드 오류: {e}")
        
        # 별도 스레드에서 삭제 기능 실행
        thread = threading.Thread(target=delete_in_thread, daemon=True)
        thread.start()
        print("🗑️  삭제 모드가 별도 창에서 실행됩니다 (카메라 화면 유지)")
    
    def _delete_faces_in_folder(self, folder_name: str, folder_path: Path):
        """특정 폴더의 얼굴들 삭제"""
        try:
            # 이미지 파일들 찾기
            image_files = list(folder_path.glob('*.jpg'))
            
            if not image_files:
                print(f"❌ {folder_name}에 삭제할 파일이 없습니다")
                return
            
            print(f"\n📁 {folder_name} 폴더의 파일들:")
            for i, img_file in enumerate(image_files[:10], 1):  # 최대 10개만 표시
                print(f"  {i}. {img_file.name}")
            
            if len(image_files) > 10:
                print(f"  ... 총 {len(image_files)}개 파일")
            
            print("\n삭제 옵션:")
            print("  1. 전체 삭제")
            print("  2. 개별 선택 삭제")
            print("  0. 취소")
            
            option = input("옵션 선택: ").strip()
            
            if option == '1':
                # 전체 삭제
                confirm = input(f"정말로 {len(image_files)}개 파일을 모두 삭제하시겠습니까? (y/N): ").strip().lower()
                if confirm == 'y':
                    deleted_count = 0
                    for img_file in image_files:
                        try:
                            img_file.unlink()  # 이미지 삭제
                            # 메타데이터도 삭제
                            metadata_file = img_file.with_suffix('.json')
                            if metadata_file.exists():
                                metadata_file.unlink()
                            deleted_count += 1
                        except Exception as e:
                            print(f"❌ {img_file.name} 삭제 실패: {e}")
                    
                    print(f"✅ {deleted_count}개 파일 삭제 완료")
                else:
                    print("❌ 삭제 취소")
            
            elif option == '2':
                # 개별 선택 삭제
                print("삭제할 파일 번호들을 입력하세요 (예: 1,3,5 또는 1-5)")
                indices_input = input("번호: ").strip()
                
                try:
                    indices = self._parse_indices(indices_input, len(image_files))
                    
                    if indices:
                        deleted_count = 0
                        for idx in sorted(indices, reverse=True):  # 역순으로 삭제
                            try:
                                img_file = image_files[idx]
                                img_file.unlink()  # 이미지 삭제
                                # 메타데이터도 삭제
                                metadata_file = img_file.with_suffix('.json')
                                if metadata_file.exists():
                                    metadata_file.unlink()
                                deleted_count += 1
                                print(f"✅ {img_file.name} 삭제")
                            except Exception as e:
                                print(f"❌ {image_files[idx].name} 삭제 실패: {e}")
                        
                        print(f"✅ 총 {deleted_count}개 파일 삭제 완료")
                    else:
                        print("❌ 유효한 번호가 없습니다")
                
                except Exception as e:
                    print(f"❌ 번호 파싱 오류: {e}")
            
            else:
                print("❌ 삭제 취소")
        
        except Exception as e:
            print(f"❌ 폴더 삭제 오류: {e}")
    
    def _parse_indices(self, indices_input: str, max_count: int) -> List[int]:
        """인덱스 문자열 파싱 (예: "1,3,5" 또는 "1-5")"""
        indices = []
        
        for part in indices_input.split(','):
            part = part.strip()
            
            if '-' in part:
                # 범위 (예: "1-5")
                try:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start-1, min(end, max_count)))
                except ValueError:
                    continue
            else:
                # 단일 번호
                try:
                    idx = int(part) - 1
                    if 0 <= idx < max_count:
                        indices.append(idx)
                except ValueError:
                    continue
        
        return list(set(indices))  # 중복 제거
    
    def _adjust_confidence_threshold(self, delta: float):
        """신뢰도 임계값 조절"""
        self.face_detector.adjust_confidence_threshold(delta)
        print(f"🎯 신뢰도 임계값: {self.face_detector.confidence_threshold:.2f}")
    
    def _adjust_scale_factor(self, delta: float):
        """스케일 팩터 조절"""
        self.face_detector.adjust_scale_factor(delta)
        print(f"📏 스케일 팩터: {self.face_detector.scale_factor:.2f}")
    
    def _adjust_min_neighbors(self, delta: int):
        """최소 이웃 수 조절"""
        self.face_detector.adjust_min_neighbors(delta)
        print(f"👥 최소 이웃 수: {self.face_detector.min_neighbors}")
    
    def _switch_to_next_model(self):
        """다음 모델로 변경"""
        self._switch_model(1)
    
    def _switch_to_previous_model(self):
        """이전 모델로 변경"""
        self._switch_model(-1)
    
    def _switch_model(self, direction: int):
        """모델 전환"""
        if len(self.available_models) <= 1:
            print("사용 가능한 모델이 1개뿐입니다.")
            return
        
        self.current_model_index = (self.current_model_index + direction) % len(self.available_models)
        self.current_model = self.available_models[self.current_model_index]
        
        # 새 모델로 검출기 재초기화
        self.face_detector = FaceDetector(self.current_model)
        
        print(f"🔄 모델 변경: {self.current_model['name']}")
        self.logger.info(f"모델 변경: {self.current_model['name']}")
    
    def _update_fps(self):
        """FPS 업데이트"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def _auto_save_detected_face(self, frame: np.ndarray, detection: Dict):
        """자동 모드에서 검출된 얼굴 저장 (얼굴만 저장, 전체 프레임 저장 안함)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # 얼굴 영역 추출
        bbox = detection['bbox']
        x, y, w, h = bbox
        face_crop = frame[y:y+h, x:x+w]
        
        # detected_faces/auto_collected에만 저장 (전체 프레임 저장 안함)
        face_filename = f"auto_face_{timestamp}_conf{detection['confidence']:.2f}_qual{detection['quality_score']:.2f}.jpg"
        face_path = self.paths['detected_auto'] / face_filename
        cv2.imwrite(str(face_path), face_crop)
        
        # 메타데이터 저장
        metadata = {
            'timestamp': timestamp,
            'capture_type': 'auto',
            'bbox': [int(x), int(y), int(w), int(h)],
            'confidence': float(detection['confidence']),
            'quality_score': float(detection['quality_score']),
            'face_path': str(face_path)
        }
        
        metadata_path = face_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.session_stats['auto_saved'] += 1
        self.logger.info(f"Auto saved face: {face_filename}")
    
    def _visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """검출 결과 시각화"""
        display_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            quality_score = detection.get('quality_score', 0.0)
            landmarks = detection.get('landmarks', [])
            
            x, y, w, h = bbox
            
            # 바운딩 박스 그리기
            color = (0, 255, 0) if quality_score > 0.6 else (0, 255, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도와 품질 점수 표시
            label = f"Conf: {confidence:.2f}, Qual: {quality_score:.2f}"
            cv2.putText(display_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 랜드마크 표시 (L키 토글 상태에 따라)
            if hasattr(self, 'show_landmarks') and self.show_landmarks and landmarks:
                landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                for i, (lx, ly) in enumerate(landmarks):
                    color = landmark_colors[i % len(landmark_colors)]
                    cv2.circle(display_frame, (lx, ly), 3, color, -1)
                    cv2.putText(display_frame, str(i+1), (lx+5, ly-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return display_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """정보 오버레이 그리기"""
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # 상단 정보 오버레이 (크기 축소)
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (10, 10), (500, 80), (0, 0, 0), -1)
        overlay_frame = cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0)
        
        # 텍스트 정보 (간소화)
        texts = [
            f"Model: {self.current_model['name'] if self.current_model else 'None'} | FPS: {self.current_fps:.1f}",
            f"Faces: {len(detections)} | Auto: {'ON' if self.auto_save_enabled else 'OFF'} | Landmarks: {'ON' if getattr(self, 'show_landmarks', False) else 'OFF'}",
            f"Conf: {self.face_detector.confidence_threshold:.2f} | Auto: {self.session_stats['auto_saved']} | Manual: {self.session_stats['manual_captured']}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 25 + i * 18
            cv2.putText(overlay_frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 하단 토글키 정보
        self._draw_toggle_keys(overlay_frame, width, height)
        
        return overlay_frame
    
    def _draw_toggle_keys(self, frame: np.ndarray, width: int, height: int):
        """토글키 정보를 화면 하단에 표시"""
        # 배경 박스 (반투명)
        box_height = 140
        box_y = height - box_height - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, box_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 토글키 정보
        y_start = box_y + 20
        line_height = 25
        
        # 첫 번째 줄: 기본 명령어
        cv2.putText(frame, "Toggle Keys:", (10, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 두 번째 줄: 모드 및 정보
        cv2.putText(frame, "A: Auto/Manual  I: Info  L: Landmarks  H: Help  Q: Quit", (10, y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 세 번째 줄: 캡처 명령어
        cv2.putText(frame, "S: Save Frame  C: Capture Face  D: Delete Faces", (10, y_start + line_height * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 네 번째 줄: 모델 설정
        cv2.putText(frame, "Model Settings: +/- Confidence  [/] Scale  ,/. Neighbors  N/M: Model", (10, y_start + line_height * 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 다섯 번째 줄: 현재 설정값
        current_settings = f"Current: Conf={self.face_detector.confidence_threshold:.2f} Scale={self.face_detector.scale_factor:.2f} Neighbors={self.face_detector.min_neighbors}"
        cv2.putText(frame, current_settings, (10, y_start + line_height * 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _handle_keyboard_input(self, key: int, frame: np.ndarray, detections: List[Dict]) -> str:
        """키보드 입력 처리"""
        if key == ord('q'):  # 종료
            return 'quit'
        
        elif key == ord('i'):  # 정보 표시 토글
            self.show_info = not self.show_info
            status = "ON" if self.show_info else "OFF"
            print(f"ℹ️  정보 표시: {status}")
            return 'continue'
        
        elif key == ord('l'):  # 랜드마크 표시 토글
            self.show_landmarks = not getattr(self, 'show_landmarks', False)
            status = "ON" if self.show_landmarks else "OFF"
            print(f"🎯 랜드마크 표시: {status}")
            return 'continue'
        
        elif key == ord('a'):  # 자동/수동 모드 전환
            self.is_auto_mode = not self.is_auto_mode
            mode = "🤖 자동" if self.is_auto_mode else "👤 수동"
            print(f"🔄 모드 변경: {mode}")
            return 'continue'
        
        elif key == ord('s'):  # 전체 프레임 저장
            self._save_full_frame(frame)
            return 'continue'
        
        elif key == ord('c'):  # 얼굴 캡처 + 이름 입력
            self._handle_manual_face_capture(frame, detections)
            return 'continue'
        
        elif key == ord('d'):  # 얼굴 삭제 모드
            self._delete_unwanted_faces()
            return 'continue'
        
        elif key == ord('h'):  # 도움말
            self._print_help()
            return 'continue'
        
        # 모델 설정 조절
        elif key == ord('+') or key == ord('='):  # 신뢰도 증가
            self._adjust_confidence_threshold(0.05)
            return 'continue'
        
        elif key == ord('-'):  # 신뢰도 감소
            self._adjust_confidence_threshold(-0.05)
            return 'continue'
        
        elif key == ord('['):  # 스케일 팩터 감소
            self._adjust_scale_factor(-0.05)
            return 'continue'
        
        elif key == ord(']'):  # 스케일 팩터 증가
            self._adjust_scale_factor(0.05)
            return 'continue'
        
        elif key == ord(','):  # 최소 이웃 수 감소
            self._adjust_min_neighbors(-1)
            return 'continue'
        
        elif key == ord('.'):  # 최소 이웃 수 증가
            self._adjust_min_neighbors(1)
            return 'continue'
        
        # 모델 변경
        elif key == ord('n'):  # 다음 모델
            self._switch_to_next_model()
            return 'continue'
        
        elif key == ord('m'):  # 이전 모델
            self._switch_to_previous_model()
            return 'continue'
        
        return 'continue'
    
    def _print_help(self):
        """도움말 출력"""
        print("\n" + "="*60)
        print("📋 개선된 얼굴 캡처 시스템 - 키보드 명령어")
        print("="*60)
        print("🔧 공통 명령어:")
        print("   i  - 정보 표시 토글")
        print("   l  - 랜드마크 표시 토글")
        print("   a  - 자동/수동 모드 전환")
        print("   h  - 도움말 표시")
        print("   q  - 종료")
        print()
        print("👤 캡처 명령어:")
        print("   s  - 전체 프레임 저장 (raw_input/captured/)")
        print("   c  - 얼굴 캡처 + GUI 이름 입력 (staging/named/)")
        print("   d  - 불필요한 얼굴 삭제")
        print()
        print("⚙️  모델 설정 조절:")
        print("   +/- - 신뢰도 임계값 조절")
        print("   [/] - 스케일 팩터 조절 (Haar Cascade)")
        print("   ,/. - 최소 이웃 수 조절 (Haar Cascade)")
        print()
        print("🔄 모델 변경:")
        print("   n  - 다음 모델")
        print("   m  - 이전 모델")
        print()
        print("📁 데이터 플로우:")
        print("   1. s키: raw_input/captured/ (전체 프레임)")
        print("   2. c키: detected_faces/from_manual/ → staging/named/{이름}/")
        print("   3. 자동모드: detected_faces/auto_collected/ (얼굴만)")
        print("   4. d키: 저장된 얼굴 삭제")
        print("="*60)
    
    def _print_session_stats(self):
        """세션 통계 출력"""
        print("\n📊 세션 통계:")
        print(f"   자동 저장: {self.session_stats['auto_saved']}개")
        print(f"   수동 캡처: {self.session_stats['manual_captured']}개")
        print(f"   이름 지정: {self.session_stats['named_saved']}개")
        print(f"   사용된 모델: {self.current_model['name'] if self.current_model else 'None'}")
    
    def _cleanup(self):
        """정리 작업"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # 세션 통계 출력
        self._print_session_stats()
        
        self.logger.info("얼굴 캡처 시스템 종료")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="개선된 얼굴 캡처 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    args = parser.parse_args()
    
    try:
        # 하드웨어 연결 확인
        print("🔍 하드웨어 연결 상태 확인 중...")
        cap_test = cv2.VideoCapture(args.camera)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            cap_test.release()
            if ret:
                print(f"✅ 카메라 {args.camera} 연결 확인됨")
            else:
                print(f"❌ 카메라 {args.camera}에서 프레임을 읽을 수 없음")
                return
        else:
            print(f"❌ 카메라 {args.camera} 연결 실패")
            return
        
        # 캡처 시스템 시작
        capture_system = EnhancedFaceCaptureSystem(camera_id=args.camera)
        capture_system.start_capture()
        
    except KeyboardInterrupt:
        print("\n⚠️  프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 