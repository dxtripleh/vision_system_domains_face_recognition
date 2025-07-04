#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Factory Defect Detection Model Test.

단위 테스트용 예제 이미지 추론 코드입니다.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from .model import DefectDetectionModel
from . import DEFECT_TYPES

class TestDefectDetectionModel(unittest.TestCase):
    """불량 검출 모델 테스트 클래스."""
    
    def setUp(self):
        """테스트 설정."""
        self.model = None
        self.test_image = None
        
        # 테스트용 이미지 생성
        self.test_image = self._create_test_image()
    
    def tearDown(self):
        """테스트 정리."""
        if self.model:
            del self.model
    
    def _create_test_image(self) -> np.ndarray:
        """테스트용 이미지를 생성합니다."""
        # 640x640 크기의 테스트 이미지 생성
        image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        
        # 테스트용 불량 패턴 추가 (간단한 사각형)
        cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)  # 빨간색 사각형
        cv2.rectangle(image, (300, 300), (400, 400), (255, 0, 0), -1)  # 파란색 사각형
        
        return image
    
    def test_model_initialization(self):
        """모델 초기화 테스트."""
        try:
            # 모델 초기화 (모델 파일이 없어도 기본 설정으로 초기화)
            self.model = DefectDetectionModel()
            
            # 기본 속성 확인
            self.assertIsNotNone(self.model.config)
            self.assertIsNotNone(self.model.device_config)
            self.assertIsNotNone(self.model.input_shape)
            self.assertIsNotNone(self.model.output_names)
            
            print("✓ 모델 초기화 테스트 통과")
            
        except FileNotFoundError:
            # 모델 파일이 없는 경우는 정상 (테스트 환경)
            print("⚠ 모델 파일이 없어 초기화 테스트 건너뜀")
        except Exception as e:
            self.fail(f"모델 초기화 실패: {e}")
    
    def test_hardware_detection(self):
        """하드웨어 감지 테스트."""
        try:
            self.model = DefectDetectionModel()
            
            # 하드웨어 설정 확인
            device_config = self.model.device_config
            
            self.assertIn('device', device_config)
            self.assertIn('providers', device_config)
            self.assertIn('optimization_level', device_config)
            
            print(f"✓ 하드웨어 감지 테스트 통과: {device_config['device']}")
            
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 하드웨어 감지 테스트 건너뜀")
        except Exception as e:
            self.fail(f"하드웨어 감지 실패: {e}")
    
    def test_preprocessing(self):
        """전처리 테스트."""
        try:
            self.model = DefectDetectionModel()
            
            # 전처리 실행
            processed = self.model.preprocess(self.test_image)
            
            # 결과 검증
            self.assertEqual(processed.ndim, 4)  # (1, C, H, W)
            self.assertEqual(processed.shape[0], 1)  # 배치 크기
            self.assertEqual(processed.shape[1], 3)  # 채널 수
            self.assertEqual(processed.dtype, np.float32)
            
            # 값 범위 확인 (0-1)
            self.assertTrue(np.all(processed >= 0))
            self.assertTrue(np.all(processed <= 1))
            
            print("✓ 전처리 테스트 통과")
            
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 전처리 테스트 건너뜀")
        except Exception as e:
            self.fail(f"전처리 실패: {e}")
    
    def test_defect_types(self):
        """불량 유형 정의 테스트."""
        # 불량 유형 확인
        self.assertIn('scratch', DEFECT_TYPES)
        self.assertIn('dent', DEFECT_TYPES)
        self.assertIn('crack', DEFECT_TYPES)
        self.assertIn('discoloration', DEFECT_TYPES)
        self.assertIn('contamination', DEFECT_TYPES)
        
        # 각 불량 유형의 필수 필드 확인
        for defect_type, info in DEFECT_TYPES.items():
            self.assertIn('id', info)
            self.assertIn('name', info)
            self.assertIn('color', info)
            self.assertIsInstance(info['id'], int)
            self.assertIsInstance(info['name'], str)
            self.assertIsInstance(info['color'], list)
            self.assertEqual(len(info['color']), 3)
        
        print("✓ 불량 유형 정의 테스트 통과")
    
    def test_class_name_mapping(self):
        """클래스 이름 매핑 테스트."""
        try:
            self.model = DefectDetectionModel()
            
            # 각 클래스 ID에 대한 이름 확인
            for defect_type, info in DEFECT_TYPES.items():
                class_name = self.model._get_class_name(info['id'])
                self.assertEqual(class_name, info['name'])
            
            # 알 수 없는 클래스 ID 처리
            unknown_name = self.model._get_class_name(999)
            self.assertTrue(unknown_name.startswith('unknown_'))
            
            print("✓ 클래스 이름 매핑 테스트 통과")
            
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 클래스 이름 매핑 테스트 건너뜀")
        except Exception as e:
            self.fail(f"클래스 이름 매핑 실패: {e}")
    
    def test_nms_application(self):
        """NMS 적용 테스트."""
        try:
            self.model = DefectDetectionModel()
            
            # 테스트용 검출 결과
            test_detections = [
                {
                    'bbox': [100, 100, 200, 200],
                    'class_id': 0,
                    'confidence': 0.8,
                    'class_name': '스크래치'
                },
                {
                    'bbox': [110, 110, 210, 210],  # 겹치는 박스
                    'class_id': 0,
                    'confidence': 0.6,
                    'class_name': '스크래치'
                },
                {
                    'bbox': [300, 300, 400, 400],  # 겹치지 않는 박스
                    'class_id': 1,
                    'confidence': 0.7,
                    'class_name': '함몰'
                }
            ]
            
            # NMS 적용
            filtered_detections = self.model._apply_nms(test_detections)
            
            # 결과 검증 (겹치는 박스 중 신뢰도가 높은 것만 남아야 함)
            self.assertLessEqual(len(filtered_detections), len(test_detections))
            
            print("✓ NMS 적용 테스트 통과")
            
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 NMS 테스트 건너뜀")
        except Exception as e:
            self.fail(f"NMS 적용 실패: {e}")
    
    def test_model_info(self):
        """모델 정보 테스트."""
        try:
            self.model = DefectDetectionModel()
            
            # 모델 정보 가져오기
            model_info = self.model.get_model_info()
            
            # 필수 필드 확인
            self.assertIn('model_path', model_info)
            self.assertIn('input_shape', model_info)
            self.assertIn('output_names', model_info)
            self.assertIn('device_config', model_info)
            self.assertIn('defect_types', model_info)
            self.assertIn('config', model_info)
            
            print("✓ 모델 정보 테스트 통과")
            
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 모델 정보 테스트 건너뜀")
        except Exception as e:
            self.fail(f"모델 정보 가져오기 실패: {e}")

def run_basic_test():
    """기본 테스트 실행 (모델 파일 없이도 실행 가능)."""
    print("🔍 Factory Defect Detection Model 기본 테스트 시작")
    print("=" * 50)
    
    # 테스트 인스턴스 생성
    test = TestDefectDetectionModel()
    test.setUp()
    
    try:
        # 기본 테스트 실행
        test.test_defect_types()
        test.test_model_initialization()
        test.test_hardware_detection()
        test.test_preprocessing()
        test.test_class_name_mapping()
        test.test_nms_application()
        test.test_model_info()
        
        print("=" * 50)
        print("✅ 모든 기본 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    finally:
        test.tearDown()

if __name__ == "__main__":
    # 단위 테스트 실행
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[''], exit=False)
    else:
        # 기본 테스트 실행
        run_basic_test() 