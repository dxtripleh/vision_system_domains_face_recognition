#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 AI 얼굴 그룹핑 처리기 (전처리 및 크로스체크 강화)

detected_faces 폴더의 얼굴들을 AI로 분석하여 같은 사람끼리 그룹핑하고
staging/grouped 폴더에 저장합니다.

특징:
- 얼굴 정렬 및 전처리 강화
- 다중 모델 크로스체크 시스템
- 품질 기반 필터링
- 최신 얼굴 인식 모델 활용

사용법:
    python run_unified_ai_grouping_processor.py [--source from_captured|from_uploads|auto_collected|all]
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from datetime import datetime
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenCV import (표준 방식)
try:
    import cv2
except ImportError:
    print("❌ OpenCV가 설치되어 있지 않습니다. pip install opencv-python")
    sys.exit(1)

# 필수 모듈 import (실패 시 즉시 종료)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    print("⚠️ MediaPipe가 설치되어 있지 않습니다. pip install mediapipe")
    MP_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("⚠️ dlib이 설치되어 있지 않습니다. pip install dlib")
    DLIB_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️ onnxruntime이 설치되어 있지 않습니다. pip install onnxruntime")
    ONNX_AVAILABLE = False

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from common.logging import setup_logging, get_logger
    from common.config import load_config
except ImportError as e:
    print(f"❌ 공통 모듈 import 실패: {e}")
    print("프로젝트 구조를 확인해주세요.")
    sys.exit(1)

class FacePreprocessor:
    """얼굴 전처리 및 정렬 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.face_mesh = None
        self.face_detector = None
        self.shape_predictor = None
        self.mp_available = MP_AVAILABLE
        self.dlib_available = DLIB_AVAILABLE
        self._initialize_face_processing()
    
    def _initialize_face_processing(self):
        """얼굴 처리 모델 초기화"""
        # MediaPipe Face Mesh (얼굴 랜드마크)
        if self.mp_available:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
                self.logger.info("MediaPipe Face Mesh 초기화 완료")
            except Exception as e:
                self.logger.warning(f"MediaPipe 초기화 실패: {e}")
                self.face_mesh = None
                self.mp_available = False
        
        # Dlib 얼굴 랜드마크 (대안)
        if self.dlib_available:
            try:
                predictor_path = project_root / 'models' / 'weights' / 'shape_predictor_68_face_landmarks.dat'
                if predictor_path.exists():
                    self.face_detector = dlib.get_frontal_face_detector()
                    self.shape_predictor = dlib.shape_predictor(str(predictor_path))
                    self.logger.info("Dlib 얼굴 랜드마크 초기화 완료")
                else:
                    self.logger.warning(f"Dlib 랜드마크 모델 파일 없음: {predictor_path}")
                    self.dlib_available = False
            except Exception as e:
                self.logger.warning(f"Dlib 초기화 실패: {e}")
                self.dlib_available = False
        
        # 사용 가능한 처리기 확인
        if not self.mp_available and not self.dlib_available:
            self.logger.warning("얼굴 정렬을 위한 라이브러리가 없습니다. 기본 전처리만 수행됩니다.")
    
    def preprocess_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 전처리 및 정렬"""
        if face_image is None or face_image.size == 0:
            self.logger.warning("유효하지 않은 얼굴 이미지")
            return None
        
        try:
            # 1. 얼굴 정렬
            aligned_face = self._align_face(face_image)
            if aligned_face is None:
                self.logger.debug("얼굴 정렬 실패, 원본 이미지 사용")
                aligned_face = face_image
            
            # 2. 품질 향상
            enhanced_face = self._enhance_quality(aligned_face)
            
            # 3. 정규화
            normalized_face = self._normalize_face(enhanced_face)
            
            return normalized_face
            
        except Exception as e:
            self.logger.error(f"얼굴 전처리 오류: {e}")
            # 오류 시 기본 정규화만 수행
            try:
                return self._normalize_face(face_image)
            except:
                return face_image  # 최후의 수단으로 원본 반환
    
    def _align_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 정렬 (랜드마크 기반)"""
        if self.mp_available and self.face_mesh is not None:
            return self._align_with_mediapipe(face_image)
        elif self.dlib_available and self.shape_predictor is not None:
            return self._align_with_dlib(face_image)
        else:
            self.logger.debug("랜드마크 모델 없음. 정렬 생략")
            return face_image
    
    def _align_with_mediapipe(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe를 이용한 얼굴 정렬"""
        try:
            # BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # 랜드마크 검출
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                self.logger.debug("MediaPipe: 얼굴 랜드마크 검출 실패")
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = face_image.shape[:2]
            
            # 눈 랜드마크 추출 (MediaPipe 인덱스)
            left_eye = landmarks.landmark[33]   # 왼쪽 눈 중심
            right_eye = landmarks.landmark[263] # 오른쪽 눈 중심
            
            # 픽셀 좌표로 변환
            left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))
            
            # 정렬 각도 계산
            angle = np.degrees(np.arctan2(
                right_eye_px[1] - left_eye_px[1],
                right_eye_px[0] - left_eye_px[0]
            ))
            
            # 회전 중심점
            center = ((left_eye_px[0] + right_eye_px[0]) // 2,
                     (left_eye_px[1] + right_eye_px[1]) // 2)
            
            # 회전 행렬
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 회전 적용
            aligned = cv2.warpAffine(face_image, rotation_matrix, (w, h))
            
            return aligned
            
        except Exception as e:
            self.logger.error(f"MediaPipe 정렬 오류: {e}")
            return None
    
    def _align_with_dlib(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Dlib을 이용한 얼굴 정렬"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = self.face_detector(gray)
            if not faces:
                self.logger.debug("Dlib: 얼굴 검출 실패")
                return None
            
            # 랜드마크 추출
            shape = self.shape_predictor(gray, faces[0])
            
            # 눈 랜드마크 (Dlib 인덱스)
            left_eye = (shape.part(36).x, shape.part(36).y)   # 왼쪽 눈 시작
            right_eye = (shape.part(45).x, shape.part(45).y)  # 오른쪽 눈 시작
            
            # 정렬 각도 계산
            angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            # 회전 중심점
            center = ((left_eye[0] + right_eye[0]) // 2,
                     (left_eye[1] + right_eye[1]) // 2)
            
            # 회전 행렬
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 회전 적용
            h, w = face_image.shape[:2]
            aligned = cv2.warpAffine(face_image, rotation_matrix, (w, h))
            
            return aligned
            
        except Exception as e:
            self.logger.error(f"Dlib 정렬 오류: {e}")
            return None
    
    def _enhance_quality(self, face_image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        try:
            # 입력 검증
            if face_image is None or face_image.size == 0:
                return face_image
            
            # 1. 히스토그램 평활화 (CLAHE)
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. 노이즈 제거 (간단한 가우시안 블러로 대체)
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 3. 선명도 향상
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"품질 향상 오류: {e}")
            return face_image
    
    def _normalize_face(self, face_image: np.ndarray) -> np.ndarray:
        """얼굴 정규화"""
        try:
            # 크기 정규화 (ArcFace 표준 크기)
            normalized = cv2.resize(face_image, (112, 112))
            
            # 밝기 정규화
            normalized = cv2.convertScaleAbs(normalized, alpha=1.1, beta=5)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"정규화 오류: {e}")
            return face_image

class CrossCheckRecognizer:
    """다중 모델 크로스체크 인식 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models = {}
        self.preprocessor = FacePreprocessor()
        self.model_config = {'name': 'CrossCheck (Multi-Model)'}  # 기본 설정 추가
        self._initialize_models()
    
    def _initialize_models(self):
        """다중 모델 초기화"""
        models_dir = project_root / 'models' / 'weights'
        
        # ONNX 모델들 (ONNX Runtime이 사용 가능한 경우에만)
        if ONNX_AVAILABLE:
            # 1. ArcFace 모델
            arcface_path = models_dir / 'arcface_glint360k_20250628.onnx'
            if arcface_path.exists():
                try:
                    self.models['arcface'] = {
                        'session': ort.InferenceSession(str(arcface_path)),
                        'name': 'ArcFace',
                        'weight': 1.0,
                        'threshold': 0.6
                    }
                    self.logger.info("ArcFace 모델 로드 완료")
                except Exception as e:
                    self.logger.error(f"ArcFace 로드 실패: {e}")
            else:
                self.logger.debug(f"ArcFace 모델 파일 없음: {arcface_path}")
            
            # 2. InsightFace 모델 (대안)
            insightface_path = models_dir / 'insightface_glint360k_20250628.onnx'
            if insightface_path.exists():
                try:
                    self.models['insightface'] = {
                        'session': ort.InferenceSession(str(insightface_path)),
                        'name': 'InsightFace',
                        'weight': 0.8,
                        'threshold': 0.65
                    }
                    self.logger.info("InsightFace 모델 로드 완료")
                except Exception as e:
                    self.logger.error(f"InsightFace 로드 실패: {e}")
            else:
                self.logger.debug(f"InsightFace 모델 파일 없음: {insightface_path}")
        else:
            self.logger.warning("ONNX Runtime 없음. ONNX 모델들을 건너뜁니다.")
        
        # 3. OpenFace 모델 (OpenCV DNN)
        openface_path = models_dir / 'openface_nn4.small2.v1.t7'
        if openface_path.exists():
            try:
                self.models['openface'] = {
                    'net': cv2.dnn.readNetFromTorch(str(openface_path)),
                    'name': 'OpenFace',
                    'weight': 0.6,
                    'threshold': 0.7
                }
                self.logger.info("OpenFace 모델 로드 완료")
            except Exception as e:
                self.logger.error(f"OpenFace 로드 실패: {e}")
        else:
            self.logger.debug(f"OpenFace 모델 파일 없음: {openface_path}")
        
        # 4. ORB 특징 (항상 사용 가능한 폴백)
        try:
            self.models['orb'] = {
                'detector': cv2.ORB_create(nfeatures=500),
                'matcher': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
                'name': 'ORB',
                'weight': 0.3,
                'threshold': 0.8
            }
            self.logger.info("ORB 특징 검출기 초기화 완료")
        except Exception as e:
            self.logger.error(f"ORB 초기화 실패: {e}")
        
        # 모델 로드 결과 확인
        if not self.models:
            self.logger.error("사용 가능한 모델이 없습니다!")
            # 최소한의 ORB라도 사용할 수 있도록 재시도
            try:
                self.models['orb'] = {
                    'detector': cv2.ORB_create(nfeatures=200),  # 더 적은 특징점
                    'matcher': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
                    'name': 'ORB-Minimal',
                    'weight': 1.0,
                    'threshold': 0.9
                }
                self.logger.warning("최소한의 ORB 모델로 폴백")
            except Exception as e:
                self.logger.critical(f"ORB 폴백마저 실패: {e}")
                raise RuntimeError("사용 가능한 얼굴 인식 모델이 없습니다.")
        
        self.logger.info(f"총 {len(self.models)}개 모델 로드 완료")
        
        # 최종 모델명 설정
        if len(self.models) >= 2:
            model_names = [model['name'] for model in self.models.values()]
            self.model_config['name'] = f"CrossCheck ({' + '.join(model_names)})"
        elif len(self.models) == 1:
            single_model_name = list(self.models.values())[0]['name']
            self.model_config['name'] = f"Single Model ({single_model_name})"
        else:
            self.model_config['name'] = "No Model Available"
    
    def extract_features(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """다중 모델로 특징 추출"""
        # 전처리
        preprocessed = self.preprocessor.preprocess_face(face_image)
        if preprocessed is None:
            return {}
        
        features = {}
        
        # 각 모델로 특징 추출
        for model_name, model_info in self.models.items():
            try:
                if model_name == 'arcface':
                    features[model_name] = self._extract_arcface_features(preprocessed, model_info)
                elif model_name == 'insightface':
                    features[model_name] = self._extract_insightface_features(preprocessed, model_info)
                elif model_name == 'openface':
                    features[model_name] = self._extract_openface_features(preprocessed, model_info)
                elif model_name == 'orb':
                    features[model_name] = self._extract_orb_features(preprocessed, model_info)
                    
            except Exception as e:
                self.logger.error(f"{model_name} 특징 추출 오류: {e}")
        
        return features
    
    def _extract_arcface_features(self, face_image: np.ndarray, model_info: Dict) -> np.ndarray:
        """ArcFace 특징 추출"""
        session = model_info['session']
        input_name = session.get_inputs()[0].name
        
        # 전처리
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        normalized = (rgb_image.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # 추론
        outputs = session.run(None, {input_name: input_tensor})
        embedding = outputs[0][0]
        
        # L2 정규화
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _extract_insightface_features(self, face_image: np.ndarray, model_info: Dict) -> np.ndarray:
        """InsightFace 특징 추출"""
        session = model_info['session']
        input_name = session.get_inputs()[0].name
        
        # 전처리 (InsightFace 표준)
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # 추론
        outputs = session.run(None, {input_name: input_tensor})
        embedding = outputs[0][0]
        
        # L2 정규화
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _extract_openface_features(self, face_image: np.ndarray, model_info: Dict) -> np.ndarray:
        """OpenFace 특징 추출"""
        net = model_info['net']
        
        # 전처리
        blob = cv2.dnn.blobFromImage(face_image, 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        
        # 추론
        net.setInput(blob)
        embedding = net.forward()
        
        return embedding.flatten()
    
    def _extract_orb_features(self, face_image: np.ndarray, model_info: Dict) -> np.ndarray:
        """ORB 특징 추출"""
        try:
            detector = model_info['detector']
            
            # 입력 검증
            if face_image is None or face_image.size == 0:
                return np.zeros(32)  # ORB descriptor는 32차원
            
            # 그레이스케일 변환
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 이미지 크기 확인 및 조정
            if gray.shape[0] < 50 or gray.shape[1] < 50:
                gray = cv2.resize(gray, (112, 112))
            
            # 특징점 검출
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                self.logger.debug("ORB: 특징점을 찾을 수 없음")
                return np.zeros(32)  # ORB descriptor 크기
            
            # 특징 벡터 정규화
            if len(descriptors) > 0:
                # 평균 특징 벡터 (모든 descriptor의 평균)
                feature_vector = np.mean(descriptors, axis=0).astype(np.float32)
                
                # 정규화
                norm = np.linalg.norm(feature_vector)
                if norm > 0:
                    feature_vector = feature_vector / norm
                
                return feature_vector
            else:
                return np.zeros(32)
                
        except Exception as e:
            self.logger.error(f"ORB 특징 추출 오류: {e}")
            return np.zeros(32)
    
    def calculate_similarity(self, features1: Dict[str, np.ndarray], 
                           features2: Dict[str, np.ndarray]) -> Dict[str, float]:
        """다중 모델 유사도 계산"""
        similarities = {}
        
        for model_name in features1.keys():
            if model_name in features2:
                try:
                    if model_name in ['arcface', 'insightface']:
                        # 코사인 유사도
                        similarity = self._cosine_similarity(
                            features1[model_name], features2[model_name]
                        )
                    elif model_name == 'openface':
                        # 유클리드 거리 기반 유사도
                        similarity = self._euclidean_similarity(
                            features1[model_name], features2[model_name]
                        )
                    elif model_name == 'orb':
                        # 해밍 거리 기반 유사도
                        similarity = self._hamming_similarity(
                            features1[model_name], features2[model_name]
                        )
                    else:
                        similarity = 0.0
                    
                    similarities[model_name] = similarity
                    
                except Exception as e:
                    self.logger.error(f"{model_name} 유사도 계산 오류: {e}")
                    similarities[model_name] = 0.0
        
        return similarities
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """유클리드 거리 기반 유사도"""
        distance = np.linalg.norm(vec1 - vec2)
        max_distance = np.sqrt(len(vec1) * 255 * 255)
        return max(0.0, 1.0 - (distance / max_distance))
    
    def _hamming_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """해밍 거리 기반 유사도"""
        distance = np.linalg.norm(vec1 - vec2)
        max_distance = np.sqrt(len(vec1) * 255 * 255)
        return max(0.0, 1.0 - (distance / max_distance))
    
    def get_consensus_similarity(self, similarities: Dict[str, float], debug_info: Dict = None) -> float:
        """다중 모델 합의 유사도"""
        if not similarities:
            return 0.0
        
        # 가중 평균 계산
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, similarity in similarities.items():
            weight = self.models[model_name]['weight']
            weighted_sum += similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        consensus = weighted_sum / total_weight
        
        # 신뢰도 계산 (모델 간 일관성)
        if len(similarities) > 1:
            values = list(similarities.values())
            std_dev = np.std(values)
            confidence = max(0.0, 1.0 - std_dev)  # 표준편차가 작을수록 신뢰도 높음
            original_consensus = consensus
            consensus *= confidence
            
            # 디버깅 정보 로깅
            if debug_info:
                face1 = debug_info.get('face1', 'unknown')
                face2 = debug_info.get('face2', 'unknown')
                self.logger.info(f"🔍 유사도 분석: {face1} vs {face2}")
                for model_name, similarity in similarities.items():
                    if model_name in self.models:
                        weight = self.models[model_name]['weight']
                        self.logger.info(f"  - {model_name}: {similarity:.3f} (가중치: {weight})")
                self.logger.info(f"  📊 가중평균: {original_consensus:.3f}")
                self.logger.info(f"  📊 표준편차: {std_dev:.3f}, 신뢰도: {confidence:.3f}")
                self.logger.info(f"  📊 최종 컨센서스: {consensus:.3f}")
        
        return consensus

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
            'recommended_recognition_model': None
        }
        
        # 최적 인식 모델 선택
        env_info['recommended_recognition_model'] = self._select_optimal_recognition_model(env_info)
        
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
        """사용 가능한 인식 모델 스캔"""
        models_dir = project_root / 'models' / 'weights'
        available_models = []
        
        # ArcFace 모델들 스캔
        for model_file in models_dir.glob('*arcface*.onnx'):
            available_models.append({
                'name': f'ArcFace ({model_file.stem})',
                'type': 'arcface',
                'path': str(model_file),
                'speed': 'medium',
                'accuracy': 'high',
                'gpu_required': False
            })
        
        # OpenFace 모델들 스캔
        for model_file in models_dir.glob('*openface*.onnx'):
            available_models.append({
                'name': f'OpenFace ({model_file.stem})',
                'type': 'openface',
                'path': str(model_file),
                'speed': 'fast',
                'accuracy': 'medium',
                'gpu_required': False
            })
        
        # 기본 OpenCV 특징 추출기 (항상 사용 가능)
        available_models.append({
            'name': 'OpenCV ORB Features',
            'type': 'orb',
            'path': 'builtin',
            'speed': 'fast',
            'accuracy': 'low',
            'gpu_required': False
        })
        
        return available_models
    
    def _select_optimal_recognition_model(self, env_info: Dict) -> Dict:
        """최적 인식 모델 선택"""
        available_models = env_info['available_models']
        gpu_available = env_info['gpu_available']
        
        # GPU 사용 가능하면 고성능 모델 우선
        if gpu_available:
            for model in available_models:
                if model['accuracy'] == 'high':
                    return model
        
        # CPU만 사용 가능하면 중간 성능 모델 우선
        for model in available_models:
            if not model['gpu_required'] and model['accuracy'] in ['high', 'medium']:
                return model
        
        # 기본값 (ORB)
        return available_models[-1] if available_models else None

class FaceGrouper:
    """얼굴 그룹핑 엔진 (크로스체크 시스템 적용)"""
    
    def __init__(self, recognizer: CrossCheckRecognizer):
        self.recognizer = recognizer
        self.logger = get_logger(__name__)
        
        # 그룹핑 설정 (극도로 엄격한 임계값으로 조정)
        self.similarity_threshold = 0.90  # 매우 높은 임계값으로 잘못된 그룹핑 방지
        self.min_group_size = 2
        self.consensus_threshold = 0.85  # 컨센서스 임계값도 매우 높게 조정
        self.debug_mode = True  # 디버깅 모드 활성화
        self.max_group_size = 20  # 최대 그룹 크기 제한 추가
        
        # 추가 안전장치
        self.strict_mode = True  # 엄격 모드 활성화
        self.require_multiple_models = True  # 다중 모델 합의 필수
    
    def group_faces(self, face_data_list: List[Dict]) -> List[List[Dict]]:
        """얼굴들을 그룹핑 (크로스체크 적용)"""
        if not face_data_list:
            return []
        
        self.logger.info(f"{len(face_data_list)}개 얼굴을 크로스체크 그룹핑 중...")
        
        # 파일 경로 기반 중복 제거
        unique_faces = []
        seen_paths = set()
        
        for face_data in face_data_list:
            face_path = face_data['face_path']
            if face_path not in seen_paths:
                unique_faces.append(face_data)
                seen_paths.add(face_path)
            else:
                self.logger.warning(f"중복 파일 제거: {face_path}")
        
        self.logger.info(f"중복 제거 후 유효한 얼굴: {len(unique_faces)}개")
        
        # 다중 모델 특징 추출
        valid_faces = []
        for i, face_data in enumerate(unique_faces):
            try:
                face_image = cv2.imread(face_data['face_path'])
                if face_image is not None:
                    # 크로스체크 특징 추출
                    features = self.recognizer.extract_features(face_image)
                    if features:
                        face_data['features'] = features
                        valid_faces.append(face_data)
                        
                        # 진행 상황 로깅
                        if i % 5 == 0:
                            model_count = len(features)
                            self.logger.info(f"특징 추출 진행: {i+1}/{len(unique_faces)} (모델 {model_count}개)")
                    else:
                        self.logger.warning(f"특징 추출 실패: {face_data['face_path']}")
                else:
                    self.logger.warning(f"이미지 로드 실패: {face_data['face_path']}")
            except Exception as e:
                self.logger.error(f"얼굴 처리 오류 {face_data['face_path']}: {e}")
        
        self.logger.info(f"유효한 얼굴: {len(valid_faces)}/{len(unique_faces)}")
        
        if not valid_faces:
            return []
        
        # 크로스체크 그룹핑 수행
        groups = self._perform_crosscheck_clustering(valid_faces)
        
        # 필터링 (크기 및 안전장치)
        filtered_groups = []
        for group in groups:
            # 최소 그룹 크기 확인
            if len(group) < self.min_group_size:
                self.logger.debug(f"그룹 크기 부족으로 제거: {len(group)}개")
                continue
            
            # 최대 그룹 크기 제한 (의심스러운 대형 그룹 방지)
            if len(group) > self.max_group_size:
                self.logger.warning(f"⚠️ 의심스러운 대형 그룹 감지: {len(group)}개 - 분할 필요")
                # 대형 그룹을 더 작은 그룹들로 분할
                sub_groups = self._split_large_group(group)
                filtered_groups.extend(sub_groups)
            else:
                # 엄격 모드에서 그룹 일관성 재검증
                if self.strict_mode and self.validate_group_consistency(group):
                    filtered_groups.append(group)
                elif not self.strict_mode:
                    filtered_groups.append(group)
                else:
                    self.logger.warning(f"그룹 일관성 검증 실패로 제거: {len(group)}개")
        
        self.logger.info(f"크로스체크 그룹핑 완료: {len(filtered_groups)}개 그룹 생성")
        
        return filtered_groups
    
    def _perform_crosscheck_clustering(self, face_data_list: List[Dict]) -> List[List[Dict]]:
        """크로스체크 클러스터링 수행"""
        N = len(face_data_list)
        
        # 1. 모든 얼굴 쌍의 크로스체크 유사도 계산
        self.logger.info("크로스체크 유사도 계산 중...")
        similarity_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i+1, N):
                try:
                    # 다중 모델 유사도 계산
                    similarities = self.recognizer.calculate_similarity(
                        face_data_list[i]['features'], 
                        face_data_list[j]['features']
                    )
                    
                    # 디버깅 정보 준비
                    debug_info = {
                        'face1': face_data_list[i]['filename'],
                        'face2': face_data_list[j]['filename']
                    }
                    
                    # 합의 유사도 계산 (디버깅 정보 포함)
                    consensus_similarity = self.recognizer.get_consensus_similarity(similarities, debug_info if self.debug_mode else None)
                    
                    # 임계값 초과 시 주의 로깅
                    if consensus_similarity >= self.similarity_threshold:
                        self.logger.warning(f"⚠️ 임계값 초과: {face_data_list[i]['filename']} vs {face_data_list[j]['filename']} = {consensus_similarity:.3f}")
                    
                    similarity_matrix[i, j] = similarity_matrix[j, i] = consensus_similarity
                    
                except Exception as e:
                    self.logger.error(f"크로스체크 유사도 계산 오류: {e}")
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0.0
        
        # 2. 연결성 기반 그룹핑 (크로스체크 임계값 적용)
        adjacency_matrix = similarity_matrix >= self.similarity_threshold
        
        # 3. 연결된 컴포넌트(그룹) 찾기 (DFS)
        visited = [False] * N
        groups = []
        
        for i in range(N):
            if not visited[i]:
                group = []
                self._dfs_clustering(i, N, adjacency_matrix, visited, group, face_data_list)
                
                if len(group) >= self.min_group_size:
                    groups.append(group)
        
        return groups
    
    def _dfs_clustering(self, node: int, N: int, adjacency_matrix: np.ndarray, 
                       visited: List[bool], group: List[Dict], face_data_list: List[Dict]):
        """DFS를 이용한 클러스터링"""
        visited[node] = True
        group.append(face_data_list[node])
        
        for neighbor in range(N):
            if not visited[neighbor] and adjacency_matrix[node, neighbor]:
                self._dfs_clustering(neighbor, N, adjacency_matrix, visited, group, face_data_list)
    
    def validate_group_consistency(self, group: List[Dict]) -> bool:
        """그룹 내 일관성 검증 (매우 엄격한 검증)"""
        if len(group) < 2:
            return True
        
        failed_pairs = 0
        total_pairs = 0
        
        # 그룹 내 모든 쌍의 유사도 확인
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                total_pairs += 1
                similarities = self.recognizer.calculate_similarity(
                    group[i]['features'], group[j]['features']
                )
                
                debug_info = {
                    'face1': group[i]['filename'],
                    'face2': group[j]['filename']
                }
                
                consensus = self.recognizer.get_consensus_similarity(similarities, debug_info if self.debug_mode else None)
                
                if consensus < self.consensus_threshold:
                    failed_pairs += 1
                    self.logger.warning(f"⚠️ 일관성 부족: {group[i]['filename']} vs {group[j]['filename']} = {consensus:.3f} < {self.consensus_threshold}")
        
        # 80% 이상의 쌍이 임계값을 만족해야 함
        consistency_rate = (total_pairs - failed_pairs) / total_pairs if total_pairs > 0 else 0
        min_consistency_rate = 0.8
        
        is_consistent = consistency_rate >= min_consistency_rate
        
        if not is_consistent:
            self.logger.warning(f"그룹 일관성 검증 실패: {consistency_rate:.2%} < {min_consistency_rate:.0%}")
        
        return is_consistent
    
    def _split_large_group(self, large_group: List[Dict]) -> List[List[Dict]]:
        """대형 그룹을 더 작은 그룹들로 분할"""
        self.logger.info(f"대형 그룹 분할 시작: {len(large_group)}개 얼굴")
        
        # 더 높은 임계값으로 재그룹핑
        original_threshold = self.similarity_threshold
        self.similarity_threshold = min(0.95, self.similarity_threshold + 0.05)  # 임계값 상향
        
        try:
            # 대형 그룹을 다시 클러스터링
            sub_groups = self._perform_crosscheck_clustering(large_group)
            
            # 각 서브그룹이 최대 크기를 초과하지 않도록 추가 분할
            final_groups = []
            for sub_group in sub_groups:
                if len(sub_group) <= self.max_group_size:
                    final_groups.append(sub_group)
                else:
                    # 여전히 크다면 강제로 분할
                    chunks = [sub_group[i:i + self.max_group_size] 
                             for i in range(0, len(sub_group), self.max_group_size)]
                    final_groups.extend(chunks)
            
            self.logger.info(f"대형 그룹 분할 완료: {len(final_groups)}개 서브그룹 생성")
            return final_groups
            
        finally:
            # 원래 임계값 복원
            self.similarity_threshold = original_threshold

class UnifiedAIGroupingProcessor:
    """통합 AI 얼굴 그룹핑 처리기 (크로스체크 시스템)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.dry_run = config.get('dry_run', False)
        self.verbose = config.get('verbose', False)
        
        # 환경 분석
        self.env_analyzer = EnvironmentAnalyzer()
        env_info = self.env_analyzer.analyze_environment()
        
        # 크로스체크 인식기 초기화
        self.recognizer = CrossCheckRecognizer()
        
        # 그룹핑 엔진 초기화
        self.grouper = FaceGrouper(self.recognizer)
        
        # 사용자 설정 적용
        if 'similarity_threshold' in config:
            self.grouper.similarity_threshold = config['similarity_threshold']
            self.logger.info(f"유사도 임계값 조정: {config['similarity_threshold']}")
        
        if 'debug' in config:
            self.grouper.debug_mode = config['debug']
        
        # 경로 설정
        self.project_root = project_root
        self.detected_faces_dir = self.project_root / 'data' / 'domains' / 'face_recognition' / 'detected_faces'
        self.staging_dir = self.project_root / 'data' / 'domains' / 'face_recognition' / 'staging'
        self.grouped_dir = self.staging_dir / 'grouped'
        
        # 그룹핑 디렉토리 생성 (dry-run이 아닌 경우에만)
        if not self.dry_run:
            self.grouped_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logger.info("DRY RUN 모드: 디렉토리 생성 건너뜀")
        
        # 그룹핑 통계
        self.stats = {
            'total_faces': 0,
            'processed_faces': 0,
            'groups_created': 0,
            'largest_group_size': 0,
            'errors': 0
        }
        
        self.logger.info("크로스체크 AI 그룹핑 처리기 초기화 완료")
        if self.dry_run:
            self.logger.info("🧪 DRY RUN 모드 활성화 - 실제 파일 이동 없음")
        self.logger.info(f"사용 가능한 모델: {len(self.recognizer.models)}개")
        for model_name, model_info in self.recognizer.models.items():
            self.logger.info(f"  - {model_info['name']} (가중치: {model_info['weight']})")

    def process_grouping(self, source: str = 'all'):
        """그룹핑 처리 실행"""
        self.logger.info(f"AI 그룹핑 처리 시작 - 소스: {source}")
        self.logger.info(f"사용 인식 모델: {self.recognizer.model_config['name']}")
        
        # 얼굴 데이터 수집
        face_data_list = self._collect_face_data(source)
        
        if not face_data_list:
            self.logger.warning("처리할 얼굴 데이터가 없습니다")
            return
        
        self.stats['total_faces'] = len(face_data_list)
        
        # 그룹핑 수행
        groups = self.grouper.group_faces(face_data_list)
        
        if not groups:
            self.logger.warning("생성된 그룹이 없습니다")
            return
        
        # 그룹 저장
        self._save_groups(groups)
        
        self._print_summary()
    
    def _collect_face_data(self, source: str) -> List[Dict]:
        """얼굴 데이터 수집"""
        face_data_list = []
        
        # 소스별 폴더 결정
        source_folders = []
        if source in ['from_captured', 'all']:
            source_folders.append(('from_captured', self.detected_faces_dir / 'from_captured'))
        if source in ['from_uploads', 'all']:
            source_folders.append(('from_uploads', self.detected_faces_dir / 'from_uploads'))
        if source in ['from_manual', 'all']:
            source_folders.append(('from_manual', self.detected_faces_dir / 'from_manual'))
        if source in ['auto_collected', 'all']:
            source_folders.append(('auto_collected', self.detected_faces_dir / 'auto_collected'))
        
        # 이미 그룹핑된 얼굴들 확인
        grouped_faces = self._get_grouped_faces()
        
        # 각 소스 폴더에서 얼굴 데이터 수집
        for source_name, source_path in source_folders:
            self.logger.info(f"수집 중: {source_path}")
            
            # 이미지 파일들만 찾기 (JSON 파일 제외)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(source_path.glob(f'*{ext}'))
                image_files.extend(source_path.glob(f'*{ext.upper()}'))
            
            # JSON 파일 제외 (이미지 파일만 처리)
            image_files = [f for f in image_files if not f.name.endswith('.json')]
            
            self.logger.info(f"{source_name}에서 {len(image_files)}개 얼굴 이미지 발견")
            
            # 중복 처리 방지: 이미 그룹핑된 얼굴 제외
            new_faces = []
            skipped_faces = []
            
            for image_path in image_files:
                if self._is_already_grouped(image_path, grouped_faces):
                    skipped_faces.append(image_path.name)
                else:
                    new_faces.append(image_path)
            
            if skipped_faces:
                self.logger.info(f"이미 그룹핑된 얼굴 {len(skipped_faces)}개 건너뜀: {', '.join(skipped_faces[:5])}{'...' if len(skipped_faces) > 5 else ''}")
            
            for image_path in new_faces:
                # 메타데이터 파일 확인
                metadata_path = image_path.with_suffix('.json')
                metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"메타데이터 로드 실패 {metadata_path}: {e}")
                
                face_data = {
                    'face_path': str(image_path),
                    'source': source_name,
                    'metadata': metadata,
                    'filename': image_path.name
                }
                
                face_data_list.append(face_data)
        
        self.logger.info(f"총 {len(face_data_list)}개 얼굴 데이터 수집됨 (새로운 얼굴만)")
        return face_data_list
    
    def _get_grouped_faces(self) -> Set[str]:
        """이미 그룹핑된 얼굴들의 경로 수집"""
        grouped_faces = set()
        
        if not self.grouped_dir.exists():
            return grouped_faces
        
        # 모든 그룹 폴더에서 얼굴 파일들 수집
        for group_dir in self.grouped_dir.iterdir():
            if group_dir.is_dir():
                # 그룹 메타데이터 확인
                metadata_path = group_dir / 'group_metadata.json'
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            group_metadata = json.load(f)
                        
                        # 그룹에 포함된 얼굴들의 원본 경로 수집
                        for face_info in group_metadata.get('faces', []):
                            original_path = face_info.get('original_path', '')
                            if original_path:
                                grouped_faces.add(original_path)
                    except Exception as e:
                        self.logger.debug(f"그룹 메타데이터 로드 실패 {metadata_path}: {e}")
        
        return grouped_faces
    
    def _is_already_grouped(self, face_path: Path, grouped_faces: Set[str]) -> bool:
        """얼굴이 이미 그룹핑되었는지 확인"""
        return str(face_path) in grouped_faces
    
    def _save_groups(self, groups: List[List[Dict]]):
        """그룹들을 파일로 저장 (그룹핑 엔진 결과를 그대로 저장)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 기존 그룹들 로드 (참고용)
        existing_groups = self._load_existing_groups()
        
        for group_idx, group in enumerate(groups):
            # 그룹핑 엔진의 결과를 그대로 새 그룹으로 생성
            group_name = f"group_{len(existing_groups) + group_idx:03d}_{timestamp}"
            self._create_new_group(group, group_name)
            self.logger.info(f"새 그룹 생성: {group_name} ({len(group)}개 얼굴)")
            
            # 통계 업데이트
            if len(group) > self.stats['largest_group_size']:
                self.stats['largest_group_size'] = len(group)
        
        self.stats['groups_created'] = len(groups)
    
    def _load_existing_groups(self) -> List[Dict]:
        """기존 그룹들 로드"""
        existing_groups = []
        
        if not self.grouped_dir.exists():
            return existing_groups
        
        for group_dir in self.grouped_dir.iterdir():
            if group_dir.is_dir():
                metadata_path = group_dir / 'group_metadata.json'
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            group_metadata = json.load(f)
                        existing_groups.append(group_metadata)
                    except Exception as e:
                        self.logger.debug(f"그룹 메타데이터 로드 실패 {metadata_path}: {e}")
        
        return existing_groups
    
    def _create_new_group(self, group: List[Dict], group_name: str):
        """새 그룹 생성"""
        group_dir = self.grouped_dir / group_name
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: 그룹 디렉토리 생성 시뮬레이션 - {group_dir}")
        else:
            group_dir.mkdir(exist_ok=True)
        
        # 그룹 메타데이터
        group_metadata = {
            'group_id': group_name,
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'face_count': len(group),
            'recognition_model': self.recognizer.model_config['name'],
            'dry_run': self.dry_run,
            'faces': []
        }
        
        # 각 얼굴 이미지를 그룹 폴더로 복사
        for face_idx, face_data in enumerate(group):
            source_path = Path(face_data['face_path'])
            
            # 새 파일명 생성
            new_filename = f"face_{face_idx:03d}_{source_path.name}"
            dest_path = group_dir / new_filename
            
            # 그룹 메타데이터에 추가
            face_info = {
                'original_path': str(source_path),
                'new_path': str(dest_path),
                'source': face_data['source'],
                'original_metadata': face_data['metadata']
            }
            group_metadata['faces'].append(face_info)
            
            if self.dry_run:
                # DRY RUN 모드: 실제 복사는 하지 않고 로그만
                self.logger.debug(f"DRY RUN: 파일 복사 시뮬레이션 - {source_path} -> {dest_path}")
                self.stats['processed_faces'] += 1
            else:
                # 실제 파일 복사
                try:
                    if not source_path.exists():
                        self.logger.error(f"소스 파일이 존재하지 않음: {source_path}")
                        self.stats['errors'] += 1
                        continue
                    
                    shutil.copy2(source_path, dest_path)
                    self.stats['processed_faces'] += 1
                    
                    if self.verbose:
                        self.logger.debug(f"파일 복사 완료: {source_path} -> {dest_path}")
                    
                except Exception as e:
                    self.logger.error(f"파일 복사 실패 {source_path} -> {dest_path}: {e}")
                    self.stats['errors'] += 1
        
        # 그룹 메타데이터 저장
        metadata_path = group_dir / 'group_metadata.json'
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: 메타데이터 저장 시뮬레이션 - {metadata_path}")
            if self.verbose:
                self.logger.debug(f"DRY RUN: 메타데이터 내용 - {len(group_metadata['faces'])}개 얼굴")
        else:
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(group_metadata, f, indent=2, ensure_ascii=False)
                
                if self.verbose:
                    self.logger.debug(f"그룹 메타데이터 저장 완료: {metadata_path}")
                    
            except Exception as e:
                self.logger.error(f"그룹 메타데이터 저장 실패 {metadata_path}: {e}")
                self.stats['errors'] += 1
    
    def _print_summary(self):
        """처리 결과 요약"""
        print("\n" + "="*60)
        print("📊 AI 그룹핑 처리 결과 요약")
        if self.dry_run:
            print("🧪 DRY RUN 모드 - 실제 파일 이동 없음")
        print("="*60)
        print(f"🤖 사용 인식 모델: {self.recognizer.model_config['name']}")
        print(f"👤 총 얼굴 수: {self.stats['total_faces']}개")
        print(f"✅ 처리된 얼굴: {self.stats['processed_faces']}개")
        print(f"👥 {'시뮬레이션된' if self.dry_run else '생성된'} 그룹: {self.stats['groups_created']}개")
        print(f"📈 최대 그룹 크기: {self.stats['largest_group_size']}개")
        print(f"❌ 오류: {self.stats['errors']}개")
        if self.dry_run:
            print("💡 실제 실행하려면 --dry-run 옵션을 제거하세요")
        print("="*60)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 AI 얼굴 그룹핑 처리기")
    parser.add_argument(
        "--source", 
        choices=['from_captured', 'from_uploads', 'from_manual', 'auto_collected', 'all'], 
        default='all',
        help="처리할 소스 폴더 (기본값: all)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 파일 이동 없이 테스트만 수행"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세한 로그 출력"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버깅 모드 활성화 (유사도 상세 분석)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="유사도 임계값 설정 (기본값: 0.75)"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = get_logger(__name__)
    
    # 의존성 검사 결과 출력
    print("\n🔍 시스템 의존성 검사:")
    print(f"  - OpenCV: ✅ 사용 가능")
    print(f"  - MediaPipe: {'✅ 사용 가능' if MP_AVAILABLE else '❌ 설치되지 않음'}")
    print(f"  - Dlib: {'✅ 사용 가능' if DLIB_AVAILABLE else '❌ 설치되지 않음'}")
    print(f"  - ONNX Runtime: {'✅ 사용 가능' if ONNX_AVAILABLE else '❌ 설치되지 않음'}")
    
    if not MP_AVAILABLE and not DLIB_AVAILABLE:
        print("\n⚠️ 경고: 얼굴 정렬 라이브러리가 없습니다.")
        print("   더 나은 결과를 위해 다음 중 하나를 설치하세요:")
        print("   - pip install mediapipe")
        print("   - pip install dlib")
    
    if not ONNX_AVAILABLE:
        print("\n⚠️ 경고: ONNX Runtime이 없어 고성능 모델을 사용할 수 없습니다.")
        print("   설치: pip install onnxruntime")
    
    try:
        # 설정 로드
        config = {}
        if args.config:
            if Path(args.config).exists():
                config = load_config(args.config)
                logger.info(f"설정 파일 로드됨: {args.config}")
            else:
                logger.warning(f"설정 파일을 찾을 수 없습니다: {args.config}")
        
        # 기본 설정 추가
        config.setdefault('dry_run', args.dry_run)
        config.setdefault('verbose', args.verbose)
        config.setdefault('debug', args.debug)
        config.setdefault('similarity_threshold', args.threshold)
        
        if args.debug:
            logger.info("🔍 디버깅 모드 활성화 - 유사도 상세 분석을 출력합니다")
        
        logger.info(f"🎯 유사도 임계값: {args.threshold}")
        
        print(f"\n🚀 AI 그룹핑 처리 시작 (소스: {args.source})")
        if args.dry_run:
            print("   📋 DRY RUN 모드: 실제 파일 이동은 하지 않습니다.")
        
        # AI 그룹핑 처리기 초기화 및 실행
        processor = UnifiedAIGroupingProcessor(config)
        processor.process_grouping(args.source)
        
        logger.info("AI 그룹핑 처리 완료")
        print("\n✅ AI 그룹핑 처리가 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 130  # SIGINT exit code
        
    except FileNotFoundError as e:
        logger.error(f"파일/폴더 없음: {e}")
        print(f"❌ 파일/폴더를 찾을 수 없습니다: {e}")
        return 2
        
    except PermissionError as e:
        logger.error(f"권한 오류: {e}")
        print(f"❌ 파일 권한 오류: {e}")
        return 3
        
    except Exception as e:
        logger.error(f"AI 그룹핑 처리 실패: {e}", exc_info=True)
        print(f"❌ 오류: {e}")
        
        if args.verbose:
            import traceback
            print("\n상세 오류 정보:")
            traceback.print_exc()
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 