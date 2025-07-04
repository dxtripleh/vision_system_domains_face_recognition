#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Service Test.

얼굴인식 서비스를 테스트하는 스크립트입니다.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from ..models.face_detection_model import FaceDetectionModel
from ..models.face_recognition_model import FaceRecognitionModel
from ..services.service import FaceRecognitionService

logger = logging.getLogger(__name__)

class FaceRecognitionTester:
    """얼굴인식 테스트 클래스"""
    
    def __init__(self):
        self.test_results = []
        print("얼굴인식 테스터 초기화")
    
    def create_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """테스트용 가짜 이미지 생성"""
        # 랜덤 색상 배경
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 중앙에 사각형 그리기 (얼굴 영역 시뮬레이션)
        center_x, center_y = width // 2, height // 2
        face_size = min(width, height) // 4
        
        cv2.rectangle(image, 
                     (center_x - face_size//2, center_y - face_size//2),
                     (center_x + face_size//2, center_y + face_size//2),
                     (255, 255, 255), -1)
        
        # 눈, 코, 입 표시
        cv2.circle(image, (center_x - 20, center_y - 10), 5, (0, 0, 0), -1)
        cv2.circle(image, (center_x + 20, center_y - 10), 5, (0, 0, 0), -1)
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 0), -1)
        cv2.ellipse(image, (center_x, center_y + 15), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        return image
    
    def test_face_detection_model(self) -> Dict:
        """얼굴 검출 모델 테스트"""
        print("=== 얼굴 검출 모델 테스트 시작 ===")
        
        try:
            detector = FaceDetectionModel()
            model_info = detector.get_model_info()
            print(f"모델 정보: {model_info}")
            
            test_image = self.create_test_image()
            
            start_time = time.time()
            detections = detector.detect_faces(test_image)
            processing_time = time.time() - start_time
            
            result = {
                'test_name': 'face_detection_model',
                'success': True,
                'detections_count': len(detections),
                'processing_time_ms': processing_time * 1000,
                'model_info': model_info,
                'detections': detections
            }
            
            print(f"검출 결과: {len(detections)}개 얼굴")
            print(f"처리 시간: {processing_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            print(f"얼굴 검출 모델 테스트 실패: {e}")
            return {
                'test_name': 'face_detection_model',
                'success': False,
                'error': str(e)
            }
    
    def test_face_recognition_model(self) -> Dict:
        """얼굴 인식 모델 테스트"""
        print("=== 얼굴 인식 모델 테스트 시작 ===")
        
        try:
            recognizer = FaceRecognitionModel()
            model_info = recognizer.get_model_info()
            print(f"모델 정보: {model_info}")
            
            test_face = self.create_test_image(112, 112)
            
            start_time = time.time()
            embedding = recognizer.infer(test_face)
            processing_time = time.time() - start_time
            
            result = {
                'test_name': 'face_recognition_model',
                'success': True,
                'embedding_shape': embedding.shape if embedding is not None else None,
                'processing_time_ms': processing_time * 1000,
                'model_info': model_info
            }
            
            print(f"임베딩 생성 완료: {embedding.shape if embedding is not None else 'None'}")
            print(f"처리 시간: {processing_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            print(f"얼굴 인식 모델 테스트 실패: {e}")
            return {
                'test_name': 'face_recognition_model',
                'success': False,
                'error': str(e)
            }
    
    def test_face_recognition_service(self) -> Dict:
        """통합 얼굴인식 서비스 테스트"""
        print("=== 얼굴인식 서비스 테스트 시작 ===")
        
        try:
            service = FaceRecognitionService()
            service_info = service.get_service_info()
            print(f"서비스 정보: {service_info}")
            
            test_image = self.create_test_image()
            
            start_time = time.time()
            result = service.process_frame(test_image)
            total_time = time.time() - start_time
            
            test_result = {
                'test_name': 'face_recognition_service',
                'success': True,
                'faces_detected': len(result['faces']),
                'processing_time_ms': result['processing_time'] * 1000,
                'total_time_ms': total_time * 1000,
                'service_stats': result['stats'],
                'service_info': service_info
            }
            
            print(f"검출된 얼굴: {len(result['faces'])}개")
            print(f"처리 시간: {result['processing_time']*1000:.1f}ms")
            print(f"전체 시간: {total_time*1000:.1f}ms")
            
            return test_result
            
        except Exception as e:
            print(f"얼굴인식 서비스 테스트 실패: {e}")
            return {
                'test_name': 'face_recognition_service',
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> List[Dict]:
        """모든 테스트 실행"""
        print("=== 전체 테스트 시작 ===")
        
        tests = [
            self.test_face_detection_model,
            self.test_face_recognition_model,
            self.test_face_recognition_service
        ]
        
        results = []
        
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                print(f"테스트 실행 중 오류: {e}")
                results.append({
                    'test_name': 'unknown',
                    'success': False,
                    'error': str(e)
                })
        
        self._print_test_summary(results)
        return results
    
    def _print_test_summary(self, results: List[Dict]):
        """테스트 결과 요약 출력"""
        print("=== 테스트 결과 요약 ===")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('success', False))
        failed_tests = total_tests - passed_tests
        
        print(f"전체 테스트: {total_tests}")
        print(f"성공: {passed_tests}")
        print(f"실패: {failed_tests}")
        
        for result in results:
            test_name = result.get('test_name', 'unknown')
            success = result.get('success', False)
            status = "✅ PASS" if success else "❌ FAIL"
            
            if success:
                if 'processing_time_ms' in result:
                    print(f"{status} {test_name} ({result['processing_time_ms']:.1f}ms)")
                else:
                    print(f"{status} {test_name}")
            else:
                error = result.get('error', 'Unknown error')
                print(f"{status} {test_name} - {error}")

def main():
    """메인 함수"""
    print("얼굴인식 모델 테스트 시작")
    
    try:
        tester = FaceRecognitionTester()
        results = tester.run_all_tests()
        
        # 성공 여부 반환
        all_passed = all(r.get('success', False) for r in results)
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 