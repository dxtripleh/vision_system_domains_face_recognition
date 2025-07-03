#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
사용자 맞춤 얼굴 품질평가 시스템.

이 모듈은 사용자가 원하는 방향으로 얼굴 품질을 평가할 수 있도록
확장 가능한 구조를 제공합니다.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import logging

logger = logging.getLogger(__name__)

class CustomFaceQualityAssessor:
    """사용자 맞춤 얼굴 품질평가기"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 품질평가 설정 파일 경로
        """
        self.config_path = config_path or "domains/face_recognition/config/quality_assessment_config.yaml"
        self.config = self._load_config()
        
        # 기본 검증기들
        self.basic_assessors = {
            'size': self._assess_size,
            'blur': self._assess_blur,
            'brightness': self._assess_brightness,
            'contrast': self._assess_contrast
        }
        
        # 고급 검증기들 (설정에 따라 활성화)
        self.advanced_assessors = {
            'face_angle': self._assess_face_angle,
            'eye_quality': self._assess_eye_quality,
            'mouth_quality': self._assess_mouth_quality,
            'symmetry': self._assess_symmetry,
            'age_filter': self._assess_age,
            'emotion_filter': self._assess_emotion
        }
        
        # 얼굴 랜드마크 검출기 (고급 기능용)
        self.face_landmarks = None
        if self._needs_landmarks():
            self._initialize_landmarks()
    
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패: {e}, 기본값 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'quality_thresholds': {
                'min_face_size': 80,
                'blur_threshold': 100,
                'brightness_min': 50,
                'brightness_max': 200,
                'contrast_min': 20
            },
            'weights': {
                'size': 0.3,
                'blur': 0.3,
                'brightness': 0.2,
                'contrast': 0.2
            },
            'quality_grades': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            },
            'custom_assessment': {
                'enable_face_angle': False,
                'enable_eye_quality': False,
                'enable_mouth_quality': False,
                'enable_symmetry': False,
                'enable_age_filter': False,
                'enable_emotion_filter': False
            }
        }
    
    def _needs_landmarks(self) -> bool:
        """랜드마크가 필요한 고급 기능이 활성화되어 있는지 확인"""
        custom = self.config.get('custom_assessment', {})
        return any([
            custom.get('enable_face_angle', False),
            custom.get('enable_eye_quality', False),
            custom.get('enable_mouth_quality', False),
            custom.get('enable_symmetry', False)
        ])
    
    def _initialize_landmarks(self):
        """얼굴 랜드마크 검출기 초기화"""
        try:
            # MediaPipe 또는 dlib 사용 가능
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_landmarks = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe 얼굴 랜드마크 초기화 완료")
        except ImportError:
            logger.warning("MediaPipe 설치되지 않음, 고급 기능 비활성화")
            self.face_landmarks = None
    
    def assess_face_quality(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        종합 얼굴 품질 평가
        
        Args:
            image: 전체 이미지
            face_bbox: 얼굴 영역 (x, y, w, h)
        
        Returns:
            품질 평가 결과 딕셔너리
        """
        x, y, w, h = face_bbox
        face_crop = image[y:y+h, x:x+w]
        
        assessment_results = {}
        total_score = 0.0
        total_weight = 0.0
        
        # 1️⃣ 기본 품질 평가
        for assessor_name, assessor_func in self.basic_assessors.items():
            score = assessor_func(face_crop, face_bbox)
            weight = self.config['weights'].get(assessor_name, 0.0)
            
            assessment_results[f'{assessor_name}_score'] = score
            total_score += score * weight
            total_weight += weight
        
        # 2️⃣ 고급 품질 평가 (설정에 따라)
        custom_config = self.config.get('custom_assessment', {})
        
        for feature_name, assessor_func in self.advanced_assessors.items():
            if custom_config.get(f'enable_{feature_name}', False):
                try:
                    result = assessor_func(image, face_crop, face_bbox)
                    assessment_results[feature_name] = result
                    
                    # 고급 기능은 필터 역할 (통과/차단)
                    if not result.get('passed', True):
                        assessment_results['blocked_by'] = feature_name
                        assessment_results['overall_quality'] = 'poor'
                        assessment_results['quality_score'] = 0.0
                        return assessment_results
                        
                except Exception as e:
                    logger.warning(f"{feature_name} 평가 실패: {e}")
        
        # 3️⃣ 최종 점수 및 등급 계산
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        assessment_results['quality_score'] = final_score
        assessment_results['overall_quality'] = self._determine_quality_grade(final_score)
        
        return assessment_results
    
    # 기본 평가 함수들
    def _assess_size(self, face_crop: np.ndarray, face_bbox: Tuple) -> float:
        """크기 평가"""
        _, _, w, h = face_bbox
        min_size = self.config['quality_thresholds']['min_face_size']
        size_score = min(w, h) / min_size
        return min(size_score, 1.0)
    
    def _assess_blur(self, face_crop: np.ndarray, face_bbox: Tuple) -> float:
        """블러 평가 (선명도)"""
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        threshold = self.config['quality_thresholds']['blur_threshold']
        return min(blur_score / threshold, 1.0)
    
    def _assess_brightness(self, face_crop: np.ndarray, face_bbox: Tuple) -> float:
        """밝기 평가"""
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        
        min_bright = self.config['quality_thresholds']['brightness_min']
        max_bright = self.config['quality_thresholds']['brightness_max']
        
        if min_bright <= brightness <= max_bright:
            # 이상적인 범위 내
            ideal_brightness = (min_bright + max_bright) / 2
            score = 1.0 - abs(brightness - ideal_brightness) / (max_bright - min_bright)
        else:
            # 범위 밖 - 패널티
            if brightness < min_bright:
                score = brightness / min_bright
            else:
                score = max_bright / brightness
        
        return max(score, 0.0)
    
    def _assess_contrast(self, face_crop: np.ndarray, face_bbox: Tuple) -> float:
        """대비 평가"""
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_face)
        min_contrast = self.config['quality_thresholds']['contrast_min']
        return min(contrast / min_contrast, 1.0)
    
    # 고급 평가 함수들
    def _assess_face_angle(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """얼굴 각도 평가"""
        if not self.face_landmarks:
            return {'passed': True, 'reason': 'landmarks_unavailable'}
        
        try:
            # MediaPipe로 얼굴 랜드마크 추출
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_landmarks.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {'passed': False, 'reason': 'no_landmarks'}
            
            landmarks = results.multi_face_landmarks[0]
            
            # 얼굴 각도 계산 (간단한 버전)
            # 실제로는 더 정교한 계산 필요
            nose_tip = landmarks.landmark[1]  # 코끝
            left_eye = landmarks.landmark[33]  # 왼쪽 눈
            right_eye = landmarks.landmark[133]  # 오른쪽 눈
            
            # Yaw (좌우 회전) 각도 추정
            eye_center_x = (left_eye.x + right_eye.x) / 2
            yaw_angle = abs(nose_tip.x - eye_center_x) * 180  # 근사값
            
            limits = self.config.get('face_angle_limits', {})
            max_yaw = limits.get('max_yaw', 30)
            
            passed = yaw_angle <= max_yaw
            
            return {
                'passed': passed,
                'yaw_angle': yaw_angle,
                'max_allowed': max_yaw,
                'reason': 'angle_ok' if passed else 'angle_too_large'
            }
            
        except Exception as e:
            logger.warning(f"얼굴 각도 평가 실패: {e}")
            return {'passed': True, 'reason': 'assessment_failed'}
    
    def _assess_eye_quality(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """눈 품질 평가"""
        # 눈 검출 및 품질 평가 로직
        # 구현 예시: 눈 영역 검출, 눈 뜸 여부, 눈 크기 등
        return {'passed': True, 'reason': 'not_implemented'}
    
    def _assess_mouth_quality(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """입 품질 평가"""
        # 입 영역 품질 평가 로직
        return {'passed': True, 'reason': 'not_implemented'}
    
    def _assess_symmetry(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """얼굴 대칭성 평가"""
        # 얼굴 대칭성 분석 로직
        return {'passed': True, 'reason': 'not_implemented'}
    
    def _assess_age(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """나이 필터"""
        # 나이 추정 모델 연동 또는 다른 방법
        return {'passed': True, 'reason': 'not_implemented'}
    
    def _assess_emotion(self, image: np.ndarray, face_crop: np.ndarray, face_bbox: Tuple) -> Dict:
        """감정 필터"""
        # 감정 인식 모델 연동
        return {'passed': True, 'reason': 'not_implemented'}
    
    def _determine_quality_grade(self, score: float) -> str:
        """점수에 따른 품질 등급 결정"""
        grades = self.config['quality_grades']
        
        if score >= grades['excellent']:
            return 'excellent'
        elif score >= grades['good']:
            return 'good'
        elif score >= grades['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def update_config(self, new_config: Dict):
        """설정 업데이트"""
        self.config.update(new_config)
        
        # 설정 저장
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info("품질평가 설정 업데이트 완료")
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def get_config_template(self) -> Dict:
        """설정 템플릿 반환 (사용자 가이드용)"""
        return {
            'description': '사용자 맞춤 품질평가 설정 가이드',
            'basic_settings': {
                'weights': '각 평가 항목의 중요도 (합계 1.0)',
                'quality_thresholds': '기본 임계값들',
                'quality_grades': '품질 등급 기준'
            },
            'advanced_features': {
                'face_angle': '얼굴 각도 제한',
                'eye_quality': '눈 품질 검사',
                'age_filter': '나이 필터링',
                'emotion_filter': '감정 필터링'
            },
            'customization_examples': {
                '선명도_중시': {'weights': {'blur': 0.6, 'size': 0.2, 'brightness': 0.1, 'contrast': 0.1}},
                '정면_얼굴만': {'custom_assessment': {'enable_face_angle': True}, 'face_angle_limits': {'max_yaw': 10}},
                '고품질만': {'quality_grades': {'excellent': 0.9, 'good': 0.8, 'fair': 0.7}}
            }
        }

# 사용 예시 함수
def create_quality_assessor_for_user(user_preferences: str) -> CustomFaceQualityAssessor:
    """
    사용자 선호도에 따른 품질평가기 생성
    
    Args:
        user_preferences: 'strict', 'balanced', 'permissive', 'custom'
    """
    assessor = CustomFaceQualityAssessor()
    
    if user_preferences == 'strict':
        # 엄격한 기준
        assessor.update_config({
            'quality_grades': {'excellent': 0.9, 'good': 0.8, 'fair': 0.7, 'poor': 0.0},
            'weights': {'blur': 0.4, 'size': 0.3, 'brightness': 0.15, 'contrast': 0.15}
        })
    elif user_preferences == 'permissive':
        # 관대한 기준
        assessor.update_config({
            'quality_grades': {'excellent': 0.6, 'good': 0.4, 'fair': 0.2, 'poor': 0.0},
            'weights': {'size': 0.5, 'blur': 0.2, 'brightness': 0.15, 'contrast': 0.15}
        })
    elif user_preferences == 'balanced':
        # 균형잡힌 기준 (기본값)
        pass
    
    return assessor 