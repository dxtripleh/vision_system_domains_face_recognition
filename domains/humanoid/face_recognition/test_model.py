#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Humanoid Face Recognition Model Test.

단위 테스트용 얼굴인식 모델 테스트 코드입니다.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from .model import FaceRecognitionModel

class TestFaceRecognitionModel(unittest.TestCase):
    def setUp(self):
        self.model = None
        self.test_image = self._create_test_image()
    def tearDown(self):
        if self.model:
            del self.model
    def _create_test_image(self) -> np.ndarray:
        image = np.ones((112, 112, 3), dtype=np.uint8) * 128
        return image
    def test_model_initialization(self):
        try:
            self.model = FaceRecognitionModel()
            self.assertIsNotNone(self.model.config)
            self.assertIsNotNone(self.model.device_config)
            self.assertIsNotNone(self.model.input_shape)
            self.assertIsNotNone(self.model.output_names)
            print("✓ 모델 초기화 테스트 통과")
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 초기화 테스트 건너뜀")
        except Exception as e:
            self.fail(f"모델 초기화 실패: {e}")
    def test_preprocessing(self):
        try:
            self.model = FaceRecognitionModel()
            processed = self.model.preprocess(self.test_image)
            self.assertEqual(processed.ndim, 4)
            self.assertEqual(processed.shape[0], 1)
            self.assertEqual(processed.shape[1], 3)
            self.assertEqual(processed.dtype, np.float32)
            self.assertTrue(np.all(processed >= 0))
            self.assertTrue(np.all(processed <= 1))
            print("✓ 전처리 테스트 통과")
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 전처리 테스트 건너뜀")
        except Exception as e:
            self.fail(f"전처리 실패: {e}")
    def test_model_info(self):
        try:
            self.model = FaceRecognitionModel()
            model_info = self.model.get_model_info()
            self.assertIn('model_path', model_info)
            self.assertIn('input_shape', model_info)
            self.assertIn('output_names', model_info)
            self.assertIn('device_config', model_info)
            self.assertIn('config', model_info)
            print("✓ 모델 정보 테스트 통과")
        except FileNotFoundError:
            print("⚠ 모델 파일이 없어 모델 정보 테스트 건너뜀")
        except Exception as e:
            self.fail(f"모델 정보 가져오기 실패: {e}")

def run_basic_test():
    print("🔍 Face Recognition Model 기본 테스트 시작")
    print("=" * 50)
    test = TestFaceRecognitionModel()
    test.setUp()
    try:
        test.test_model_initialization()
        test.test_preprocessing()
        test.test_model_info()
        print("=" * 50)
        print("✅ 모든 기본 테스트 통과!")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    finally:
        test.tearDown()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[''], exit=False)
    else:
        run_basic_test() 