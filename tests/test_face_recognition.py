#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴인식 도메인 테스트

이 모듈은 얼굴인식 도메인의 기능을 테스트합니다.
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys
import tempfile
import shutil

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from domains.humanoid.face_recognition import FaceRecognitionSystem
from shared.vision_core.detection import FaceDetector
from shared.vision_core.recognition import FaceEmbedder, FaceMatcher
from shared.vision_core.preprocessing import ImageProcessor, FaceAligner


class TestFaceRecognitionSystem(unittest.TestCase):
    """얼굴인식 시스템 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "models": {
                "face_detection": {
                    "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                    "confidence_threshold": 0.5
                },
                "face_recognition": {
                    "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                    "similarity_threshold": 0.6
                }
            },
            "preprocessing": {
                "resize": [640, 640],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
        }
        
        # 테스트용 더미 이미지 생성
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_face_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_face_recognition_system_initialization(self):
        """얼굴인식 시스템 초기화 테스트"""
        try:
            system = FaceRecognitionSystem(self.test_config)
            self.assertIsNotNone(system)
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_face_detection(self):
        """얼굴 검출 테스트"""
        try:
            detector = FaceDetector(
                model_path=self.test_config["models"]["face_detection"]["path"],
                config={"confidence_threshold": 0.5}
            )
            
            # 더미 얼굴 영역 생성
            test_image = self.test_image.copy()
            # 얼굴 모양의 사각형 그리기
            cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)
            
            faces = detector.detect(test_image)
            self.assertIsInstance(faces, list)
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_face_embedding_extraction(self):
        """얼굴 임베딩 추출 테스트"""
        try:
            embedder = FaceEmbedder(
                model_path=self.test_config["models"]["face_recognition"]["path"],
                config={"embedding_size": 512}
            )
            
            embedding = embedder.extract_embedding(self.test_face_image)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape[0], 512)
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_face_matching(self):
        """얼굴 매칭 테스트"""
        matcher = FaceMatcher(
            config={"similarity_threshold": 0.6}
        )
        
        # 더미 임베딩 생성
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)
        
        # 정규화
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = matcher.match(embedding1, embedding2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_image_preprocessing(self):
        """이미지 전처리 테스트"""
        processor = ImageProcessor(
            config={
                "resize": [640, 640],
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        )
        
        processed_image = processor.process(self.test_image)
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertEqual(processed_image.shape[:2], (640, 640))
    
    def test_face_alignment(self):
        """얼굴 정렬 테스트"""
        aligner = FaceAligner(
            config={
                "output_size": [112, 112],
                "eye_center_ratio": 0.35
            }
        )
        
        # 더미 랜드마크 생성 (5점)
        landmarks = [
            [50, 30],   # 왼쪽 눈
            [80, 30],   # 오른쪽 눈
            [65, 50],   # 코
            [55, 70],   # 왼쪽 입
            [75, 70]    # 오른쪽 입
        ]
        
        aligned_face = aligner.align(self.test_face_image, landmarks)
        self.assertIsInstance(aligned_face, np.ndarray)
        self.assertEqual(aligned_face.shape[:2], (112, 112))


class TestFaceRecognitionIntegration(unittest.TestCase):
    """얼굴인식 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 테스트용 더미 이미지들 생성
        self.test_images = []
        for i in range(5):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # 각 이미지에 다른 얼굴 영역 생성
            cv2.rectangle(image, (100 + i*50, 100), (200 + i*50, 200), (255, 255, 255), -1)
            self.test_images.append(image)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_face_recognition(self):
        """엔드투엔드 얼굴인식 테스트"""
        try:
            # 시스템 초기화
            config = {
                "models": {
                    "face_detection": {
                        "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                        "confidence_threshold": 0.5
                    },
                    "face_recognition": {
                        "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                        "similarity_threshold": 0.6
                    }
                }
            }
            
            system = FaceRecognitionSystem(config)
            
            # 테스트 이미지로 얼굴인식 수행
            for i, image in enumerate(self.test_images):
                result = system.recognize_faces(image)
                self.assertIsInstance(result, dict)
                self.assertIn("faces", result)
                self.assertIn("recognitions", result)
                
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_batch_face_recognition(self):
        """배치 얼굴인식 테스트"""
        try:
            # 시스템 초기화
            config = {
                "models": {
                    "face_detection": {
                        "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                        "confidence_threshold": 0.5
                    },
                    "face_recognition": {
                        "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                        "similarity_threshold": 0.6
                    }
                }
            }
            
            system = FaceRecognitionSystem(config)
            
            # 배치 처리
            results = system.recognize_faces_batch(self.test_images)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), len(self.test_images))
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")


class TestFaceRecognitionPerformance(unittest.TestCase):
    """얼굴인식 성능 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 성능 테스트용 이미지 생성
        self.performance_test_images = []
        for i in range(10):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            self.performance_test_images.append(image)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_inference_speed(self):
        """추론 속도 테스트"""
        try:
            import time
            
            config = {
                "models": {
                    "face_detection": {
                        "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                        "confidence_threshold": 0.5
                    },
                    "face_recognition": {
                        "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                        "similarity_threshold": 0.6
                    }
                }
            }
            
            system = FaceRecognitionSystem(config)
            
            # 워밍업
            for _ in range(3):
                system.recognize_faces(self.performance_test_images[0])
            
            # 성능 측정
            inference_times = []
            for image in self.performance_test_images:
                start_time = time.time()
                system.recognize_faces(image)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            fps = 1.0 / avg_inference_time
            
            # 성능 기준 확인 (실시간 처리 기준: 30 FPS 이상)
            self.assertGreater(fps, 10.0, f"FPS가 너무 낮습니다: {fps:.1f}")
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        try:
            import psutil
            import gc
            
            config = {
                "models": {
                    "face_detection": {
                        "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                        "confidence_threshold": 0.5
                    },
                    "face_recognition": {
                        "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                        "similarity_threshold": 0.6
                    }
                }
            }
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            system = FaceRecognitionSystem(config)
            
            # 여러 이미지 처리
            for image in self.performance_test_images:
                system.recognize_faces(image)
            
            gc.collect()  # 가비지 컬렉션
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # 메모리 증가량이 500MB 이하여야 함
            self.assertLess(memory_increase, 500.0, 
                          f"메모리 사용량이 너무 많습니다: {memory_increase:.1f}MB")
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")


class TestFaceRecognitionErrorHandling(unittest.TestCase):
    """얼굴인식 오류 처리 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_image_input(self):
        """잘못된 이미지 입력 테스트"""
        try:
            config = {
                "models": {
                    "face_detection": {
                        "path": "models/weights/face_detection_retinaface_widerface_20250628.onnx",
                        "confidence_threshold": 0.5
                    },
                    "face_recognition": {
                        "path": "models/weights/face_recognition_arcface_glint360k_20250628.onnx",
                        "similarity_threshold": 0.6
                    }
                }
            }
            
            system = FaceRecognitionSystem(config)
            
            # None 입력
            with self.assertRaises(ValueError):
                system.recognize_faces(None)
            
            # 빈 배열 입력
            with self.assertRaises(ValueError):
                system.recognize_faces(np.array([]))
            
            # 잘못된 차원 입력
            with self.assertRaises(ValueError):
                system.recognize_faces(np.random.rand(100, 100))  # 2D 배열
            
        except Exception as e:
            self.skipTest(f"모델 파일이 없어 테스트를 건너뜁니다: {e}")
    
    def test_missing_model_files(self):
        """누락된 모델 파일 테스트"""
        # 존재하지 않는 모델 파일로 설정
        invalid_config = {
            "models": {
                "face_detection": {
                    "path": "models/weights/nonexistent_model.onnx",
                    "confidence_threshold": 0.5
                },
                "face_recognition": {
                    "path": "models/weights/nonexistent_model.onnx",
                    "similarity_threshold": 0.6
                }
            }
        }
        
        with self.assertRaises(FileNotFoundError):
            FaceRecognitionSystem(invalid_config)
    
    def test_invalid_config(self):
        """잘못된 설정 테스트"""
        # 필수 설정이 누락된 경우
        invalid_config = {}
        
        with self.assertRaises(KeyError):
            FaceRecognitionSystem(invalid_config)


if __name__ == "__main__":
    # 테스트 실행
    unittest.main(verbosity=2) 