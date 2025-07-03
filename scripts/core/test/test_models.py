#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
다운로드된 모델 테스트 스크립트.

이 스크립트는 다운로드된 얼굴인식 모델들이 제대로 로드되고 작동하는지 테스트합니다.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import onnxruntime as ort
import cv2

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

# 로깅 설정
logger = logging.getLogger(__name__)

class ModelTester:
    """모델 테스트 클래스"""
    
    def __init__(self):
        self.models_dir = Path("models/weights")
        self.test_results = {}
        
    def test_all_models(self):
        """모든 다운로드된 모델 테스트"""
        print("🧪 Model Testing Started")
        print("=" * 60)
        
        # 얼굴 검출 모델 테스트
        self.test_detection_models()
        
        # 얼굴 인식 모델 테스트
        self.test_recognition_models()
        
        # 결과 출력
        self.print_test_results()
        
    def test_detection_models(self):
        """얼굴 검출 모델 테스트"""
        print("\n📸 Testing Face Detection Models:")
        print("-" * 40)
        
        detection_models = [
            {
                "name": "SCRFD 10G",
                "path": "face_detection_scrfd_10g_20250628.onnx",
                "input_size": (640, 640)
            },
            {
                "name": "RetinaFace MNet025", 
                "path": "face_detection_retinaface_mnet025_20250628.onnx",
                "input_size": (640, 640)
            }
        ]
        
        for model_info in detection_models:
            result = self.test_onnx_model(
                model_info["name"],
                model_info["path"], 
                model_info["input_size"],
                "detection"
            )
            self.test_results[model_info["name"]] = result
    
    def test_recognition_models(self):
        """얼굴 인식 모델 테스트"""
        print("\n🔍 Testing Face Recognition Models:")
        print("-" * 40)
        
        recognition_models = [
            {
                "name": "ArcFace R100 (Buffalo_L)",
                "path": "face_recognition_arcface_r100_20250628.onnx",
                "input_size": (112, 112)
            },
            {
                "name": "ArcFace R50 (Buffalo_S)",
                "path": "face_recognition_arcface_r50_20250628.onnx", 
                "input_size": (112, 112)
            },
            {
                "name": "MobileFaceNet",
                "path": "face_recognition_mobilefacenet_20250628.onnx",
                "input_size": (112, 112)
            }
        ]
        
        for model_info in recognition_models:
            result = self.test_onnx_model(
                model_info["name"],
                model_info["path"],
                model_info["input_size"], 
                "recognition"
            )
            self.test_results[model_info["name"]] = result
    
    def test_onnx_model(self, model_name: str, model_path: str, input_size: tuple, model_type: str) -> dict:
        """ONNX 모델 테스트"""
        full_path = self.models_dir / model_path
        result = {
            "status": "FAILED",
            "error": None,
            "file_exists": False,
            "file_size_mb": 0,
            "loadable": False,
            "input_shape": None,
            "output_shape": None,
            "inference_time_ms": None
        }
        
        try:
            # 파일 존재 확인
            if not full_path.exists():
                result["error"] = f"Model file not found: {full_path}"
                print(f"❌ {model_name}: File not found")
                return result
                
            result["file_exists"] = True
            result["file_size_mb"] = round(full_path.stat().st_size / (1024 * 1024), 1)
            
            # ONNX 모델 로드 테스트
            try:
                # CPU 제공자로 세션 생성
                providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(str(full_path), providers=providers)
                result["loadable"] = True
                
                # 입력/출력 정보 확인
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()[0]
                result["input_shape"] = input_info.shape
                result["output_shape"] = output_info.shape
                
                # 더미 입력으로 추론 테스트
                if model_type == "detection":
                    # 검출 모델: [batch, channels, height, width]
                    dummy_input = np.random.rand(1, 3, input_size[1], input_size[0]).astype(np.float32)
                else:
                    # 인식 모델: [batch, channels, height, width] 
                    dummy_input = np.random.rand(1, 3, input_size[1], input_size[0]).astype(np.float32)
                
                # 추론 시간 측정
                import time
                start_time = time.time()
                
                # 추론 실행
                input_name = input_info.name
                outputs = session.run(None, {input_name: dummy_input})
                
                end_time = time.time()
                result["inference_time_ms"] = round((end_time - start_time) * 1000, 2)
                
                result["status"] = "SUCCESS"
                print(f"✅ {model_name}: OK ({result['file_size_mb']}MB, {result['inference_time_ms']}ms)")
                
            except Exception as e:
                result["error"] = f"Model loading/inference failed: {str(e)}"
                print(f"❌ {model_name}: Loading failed - {str(e)}")
                
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            print(f"❌ {model_name}: Unexpected error - {str(e)}")
            
        return result
    
    def test_opencv_model(self):
        """OpenCV Haar Cascade 모델 테스트"""
        model_path = self.models_dir / "face_detection_opencv_haarcascade_20250628.xml"
        
        try:
            if not model_path.exists():
                print(f"❌ OpenCV Haar Cascade: File not found")
                return False
                
            # OpenCV로 로드 테스트
            face_cascade = cv2.CascadeClassifier(str(model_path))
            
            if face_cascade.empty():
                print(f"❌ OpenCV Haar Cascade: Failed to load")
                return False
                
            # 더미 이미지로 테스트
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
            
            # 검출 테스트
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            
            file_size_mb = round(model_path.stat().st_size / (1024 * 1024), 1)
            print(f"✅ OpenCV Haar Cascade: OK ({file_size_mb}MB)")
            
            self.test_results["OpenCV Haar Cascade"] = {
                "status": "SUCCESS",
                "file_size_mb": file_size_mb
            }
            return True
            
        except Exception as e:
            print(f"❌ OpenCV Haar Cascade: {str(e)}")
            self.test_results["OpenCV Haar Cascade"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def print_test_results(self):
        """테스트 결과 요약 출력"""
        print("\n📊 Test Results Summary:")
        print("=" * 60)
        
        success_count = 0
        total_count = len(self.test_results)
        
        for model_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
            print(f"{status_icon} {model_name}")
            
            if result["status"] == "SUCCESS":
                success_count += 1
                if "file_size_mb" in result:
                    print(f"   📁 Size: {result['file_size_mb']}MB")
                if "inference_time_ms" in result:
                    print(f"   ⏱️  Inference: {result['inference_time_ms']}ms")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            print()
        
        # OpenCV 모델도 테스트
        self.test_opencv_model()
        if "OpenCV Haar Cascade" in self.test_results:
            if self.test_results["OpenCV Haar Cascade"]["status"] == "SUCCESS":
                success_count += 1
            total_count += 1
        
        print(f"🎯 Overall Result: {success_count}/{total_count} models passed")
        
        if success_count == total_count:
            print("🎉 All models are working correctly!")
        else:
            print("⚠️  Some models have issues. Check the errors above.")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 테스트 도구")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 모델 테스터 실행
    tester = ModelTester()
    tester.test_all_models()


if __name__ == "__main__":
    main() 