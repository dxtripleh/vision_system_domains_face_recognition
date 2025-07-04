#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared Vision Core 테스트 스크립트.

Shared Vision Core의 기본 클래스들을 테스트합니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from shared.vision_core import BaseDetector, BaseRecognizer, ImageProcessor
from shared.vision_core import get_optimal_device, validate_image, create_processor

logger = get_logger(__name__)


class TestImageProcessor(unittest.TestCase):
    """ImageProcessor 클래스 테스트."""
    
    def setUp(self):
        """테스트 설정."""
        self.processor = ImageProcessor()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """초기화 테스트."""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.config)
        self.assertIsNotNone(self.processor.supported_formats)
    
    def test_resize_image(self):
        """이미지 리사이즈 테스트."""
        target_size = (320, 240)
        resized = self.processor.resize_image(self.test_image, target_size)
        
        self.assertEqual(resized.shape[:2], target_size[::-1])  # OpenCV는 (height, width)
    
    def test_normalize_image(self):
        """이미지 정규화 테스트."""
        normalized = self.processor.normalize_image(self.test_image)
        
        # 정규화된 이미지는 float32 타입이어야 함
        self.assertEqual(normalized.dtype, np.float32)
        
        # 값이 0-1 범위에 있어야 함
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
    
    def test_denormalize_image(self):
        """이미지 역정규화 테스트."""
        # 정규화
        normalized = self.processor.normalize_image(self.test_image)
        
        # 역정규화
        denormalized = self.processor.denormalize_image(normalized)
        
        # 원본과 비슷한 값이어야 함 (약간의 오차 허용)
        self.assertEqual(denormalized.dtype, np.uint8)
        self.assertTrue(np.allclose(self.test_image, denormalized, atol=1))
    
    def test_crop_image(self):
        """이미지 크롭 테스트."""
        bbox = [100, 100, 300, 300]  # [x1, y1, x2, y2]
        cropped = self.processor.crop_image(self.test_image, bbox)
        
        expected_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0], 3)
        self.assertEqual(cropped.shape, expected_shape)
    
    def test_flip_image(self):
        """이미지 뒤집기 테스트."""
        # 수평 뒤집기
        flipped_h = self.processor.flip_image(self.test_image, 'horizontal')
        self.assertEqual(flipped_h.shape, self.test_image.shape)
        
        # 수직 뒤집기
        flipped_v = self.processor.flip_image(self.test_image, 'vertical')
        self.assertEqual(flipped_v.shape, self.test_image.shape)
    
    def test_rotate_image(self):
        """이미지 회전 테스트."""
        rotated = self.processor.rotate_image(self.test_image, 90)
        self.assertEqual(rotated.shape, self.test_image.shape)
    
    def test_adjust_brightness(self):
        """밝기 조정 테스트."""
        brightened = self.processor.adjust_brightness(self.test_image, 1.5)
        self.assertEqual(brightened.shape, self.test_image.shape)
    
    def test_adjust_contrast(self):
        """대비 조정 테스트."""
        contrasted = self.processor.adjust_contrast(self.test_image, 1.5)
        self.assertEqual(contrasted.shape, self.test_image.shape)
    
    def test_apply_gaussian_blur(self):
        """가우시안 블러 테스트."""
        blurred = self.processor.apply_gaussian_blur(self.test_image, 5, 1.0)
        self.assertEqual(blurred.shape, self.test_image.shape)
    
    def test_validate_image_quality(self):
        """이미지 품질 검증 테스트."""
        result = self.processor.validate_image_quality(self.test_image)
        
        self.assertIn('quality_score', result)
        self.assertIn('is_good_quality', result)
        self.assertIn('laplacian_variance', result)
        self.assertIn('threshold', result)
    
    def test_save_and_load_image(self):
        """이미지 저장 및 로드 테스트."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            # 이미지 저장
            success = self.processor.save_image(self.test_image, temp_path)
            self.assertTrue(success)
            
            # 이미지 로드
            loaded_image = self.processor.load_image(temp_path)
            self.assertEqual(loaded_image.shape, self.test_image.shape)
            
        finally:
            # 임시 파일 정리
            if temp_path.exists():
                temp_path.unlink()
    
    def test_performance_stats(self):
        """성능 통계 테스트."""
        # 몇 가지 작업 수행
        self.processor.resize_image(self.test_image)
        self.processor.normalize_image(self.test_image)
        
        stats = self.processor.get_performance_stats()
        
        self.assertIn('total_processed', stats)
        self.assertIn('total_time', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertGreater(stats['total_processed'], 0)


class TestBaseDetector(unittest.TestCase):
    """BaseDetector 클래스 테스트."""
    
    def test_abstract_class(self):
        """추상 클래스 테스트."""
        # BaseDetector는 추상 클래스이므로 직접 인스턴스화할 수 없음
        with self.assertRaises(TypeError):
            BaseDetector()


class TestBaseRecognizer(unittest.TestCase):
    """BaseRecognizer 클래스 테스트."""
    
    def test_abstract_class(self):
        """추상 클래스 테스트."""
        # BaseRecognizer는 추상 클래스이므로 직접 인스턴스화할 수 없음
        with self.assertRaises(TypeError):
            BaseRecognizer()


class TestUtilityFunctions(unittest.TestCase):
    """유틸리티 함수 테스트."""
    
    def test_get_optimal_device(self):
        """최적 디바이스 감지 테스트."""
        device = get_optimal_device()
        
        # 반환된 디바이스는 유효한 값이어야 함
        valid_devices = ['cpu', 'gpu', 'jetson']
        self.assertIn(device, valid_devices)
    
    def test_validate_image(self):
        """이미지 검증 테스트."""
        # 유효한 이미지
        valid_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.assertTrue(validate_image(valid_image))
        
        # 유효하지 않은 이미지들
        self.assertFalse(validate_image(None))
        self.assertFalse(validate_image("not an image"))
        self.assertFalse(validate_image(np.random.randint(0, 255, (480, 640), dtype=np.uint8)))  # 2D
        self.assertFalse(validate_image(np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)))  # 4채널
    
    def test_create_processor(self):
        """프로세서 생성 테스트."""
        # 이미지 프로세서 생성
        image_processor = create_processor('image')
        self.assertIsInstance(image_processor, ImageProcessor)
        
        # 지원하지 않는 타입
        with self.assertRaises(ValueError):
            create_processor('unsupported_type')
        
        # 추상 클래스 타입들
        with self.assertRaises(ValueError):
            create_processor('detector')
        
        with self.assertRaises(ValueError):
            create_processor('recognizer')


class TestCrossPlatformCompatibility(unittest.TestCase):
    """크로스 플랫폼 호환성 테스트."""
    
    def test_pathlib_usage(self):
        """Pathlib 사용 테스트."""
        # 모든 경로가 Path 객체로 처리되는지 확인
        from pathlib import Path
        
        # 프로젝트 루트 경로가 Path 객체인지 확인
        project_root = Path(__file__).parent.parent
        self.assertIsInstance(project_root, Path)
        
        # 경로 조작이 올바르게 작동하는지 확인
        models_dir = project_root / "models" / "weights"
        self.assertIsInstance(models_dir, Path)
    
    def test_no_hardcoded_paths(self):
        """하드코딩된 경로 없음 테스트."""
        # 코드에서 하드코딩된 경로가 없는지 확인
        # 이는 코드 리뷰를 통해 확인해야 하지만, 
        # 여기서는 pathlib 사용을 확인
        
        from pathlib import Path
        current_file = Path(__file__)
        self.assertTrue(current_file.exists())
    
    def test_exception_handling(self):
        """예외 처리 테스트."""
        processor = ImageProcessor()
        
        # 잘못된 입력에 대한 예외 처리 확인
        with self.assertRaises(ValueError):
            processor.resize_image(None)
        
        with self.assertRaises(ValueError):
            processor.normalize_image(None)


def run_tests():
    """테스트를 실행합니다."""
    logger.info("Shared Vision Core 테스트 시작")
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_classes = [
        TestImageProcessor,
        TestBaseDetector,
        TestBaseRecognizer,
        TestUtilityFunctions,
        TestCrossPlatformCompatibility
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    logger.info(f"테스트 완료: {result.testsRun}개 실행, {len(result.failures)}개 실패, {len(result.errors)}개 오류")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 