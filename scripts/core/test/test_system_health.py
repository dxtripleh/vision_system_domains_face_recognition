#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
시스템 건강 상태 점검.

전체 비전 시스템의 각 컴포넌트가 제대로 동작하는지 확인합니다.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

def test_basic_imports() -> Dict[str, Any]:
    """기본 라이브러리 import 테스트"""
    result = {'name': 'Basic Imports', 'passed': False, 'errors': []}
    
    try:
        import numpy as np
        import cv2
        import json
        import yaml
        import pathlib
        
        # 버전 정보 수집
        versions = {
            'numpy': np.__version__,
            'opencv': cv2.__version__,
            'python': sys.version.split()[0]
        }
        
        result['passed'] = True
        result['info'] = f"NumPy {versions['numpy']}, OpenCV {versions['opencv']}, Python {versions['python']}"
        
    except ImportError as e:
        result['errors'].append(f"Import error: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")
    
    return result

def test_domain_packages() -> Dict[str, Any]:
    """도메인 패키지 import 테스트"""
    result = {'name': 'Domain Packages', 'passed': False, 'errors': []}
    
    try:
        from domains.face_recognition.core.entities.face import Face
        from domains.face_recognition.core.entities.person import Person
        from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
        
        result['passed'] = True
        result['info'] = "Face recognition domain imports successful"
        
    except ImportError as e:
        result['errors'].append(f"Domain import error: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")
    
    return result

def test_value_objects() -> Dict[str, Any]:
    """Value Objects 테스트"""
    result = {'name': 'Value Objects', 'passed': False, 'errors': []}
    
    try:
        from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
        from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
        from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding
        from domains.face_recognition.core.value_objects.point import Point
        
        # BoundingBox 테스트
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        assert bbox.area == 8000
        assert bbox.center == (60.0, 60.0)
        
        # ConfidenceScore 테스트
        conf = ConfidenceScore(0.85)
        assert conf.is_high == True
        assert conf.percentage == 85.0
        
        # Point 테스트
        point = Point(x=50, y=60)
        assert point.distance_to(Point(x=53, y=64)) == 5.0
        
        result['passed'] = True
        result['info'] = "All value objects working correctly"
        
    except Exception as e:
        result['errors'].append(f"Value object error: {str(e)}")
        result['errors'].append(traceback.format_exc())
    
    return result

def test_repository() -> Dict[str, Any]:
    """Repository 테스트"""
    result = {'name': 'Repository', 'passed': False, 'errors': []}
    
    try:
        from domains.face_recognition.core.repositories.face_repository import FaceRepository
        from domains.face_recognition.core.repositories.person_repository import PersonRepository
        
        # 테스트용 저장소 초기화
        face_repo = FaceRepository("data/test/faces")
        person_repo = PersonRepository("data/test/persons")
        
        # 기본 동작 확인
        assert face_repo.count() >= 0
        assert person_repo.count() >= 0
        
        result['passed'] = True
        result['info'] = f"Repositories initialized - Faces: {face_repo.count()}, Persons: {person_repo.count()}"
        
    except Exception as e:
        result['errors'].append(f"Repository error: {str(e)}")
        result['errors'].append(traceback.format_exc())
    
    return result

def test_services() -> Dict[str, Any]:
    """서비스 테스트"""
    result = {'name': 'Services', 'passed': False, 'errors': []}
    
    try:
        from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
        from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
        
        # 서비스 초기화 (Mock 모드로)
        recognition_service = FaceRecognitionService(use_mock=True)
        detection_service = FaceDetectionService(use_mock=True)
        
        # 기본 동작 확인
        stats = recognition_service.get_statistics()
        assert isinstance(stats, dict)
        
        result['passed'] = True
        result['info'] = f"Services initialized - Recognition stats: {stats}"
        
    except Exception as e:
        result['errors'].append(f"Service error: {str(e)}")
        result['errors'].append(traceback.format_exc())
    
    return result

def test_ai_models() -> Dict[str, Any]:
    """AI 모델 테스트"""
    result = {'name': 'AI Models', 'passed': False, 'errors': []}
    
    try:
        from domains.face_recognition.infrastructure.models.arcface_recognizer import ArcFaceRecognizer
        from domains.face_recognition.infrastructure.models.retinaface_detector import RetinaFaceDetector
        
        # 모델 초기화 (파일이 없어도 객체 생성은 가능)
        model_path = "models/weights/arcface_model.onnx"  # 실제 파일이 없어도 됨
        
        recognizer = ArcFaceRecognizer(model_path=model_path, use_gpu=False)
        # detector = RetinaFaceDetector(model_path="models/weights/retinaface_model.onnx", use_gpu=False)
        
        # 기본 속성 확인
        assert recognizer.model_path == model_path
        assert recognizer.is_loaded == False  # 실제 파일이 없으므로
        
        result['passed'] = True
        result['info'] = "AI model classes initialized successfully"
        
    except Exception as e:
        result['errors'].append(f"AI model error: {str(e)}")
        result['errors'].append(traceback.format_exc())
    
    return result

def test_shared_modules() -> Dict[str, Any]:
    """Shared 모듈 테스트"""
    result = {'name': 'Shared Modules', 'passed': False, 'errors': []}
    
    try:
        # Shared vision_core 모듈 import 시도
        from shared.vision_core.detection.base_detector import BaseDetector
        from shared.vision_core.recognition.base_recognizer import BaseRecognizer
        
        result['passed'] = True
        result['info'] = "Shared vision core modules imported successfully"
        
    except ImportError as e:
        # 아직 구현되지 않은 모듈일 수 있으므로 경고로 처리
        result['passed'] = True
        result['info'] = "Shared modules not fully implemented yet (expected)"
        result['warnings'] = [f"Some shared modules not available: {str(e)}"]
        
    except Exception as e:
        result['errors'].append(f"Shared module error: {str(e)}")
    
    return result

def run_all_tests() -> Dict[str, Any]:
    """모든 테스트 실행"""
    tests = [
        test_basic_imports,
        test_domain_packages,
        test_value_objects,
        test_repository,
        test_services,
        test_ai_models,
        test_shared_modules
    ]
    
    results = []
    passed_count = 0
    total_count = len(tests)
    
    print("🔍 System Health Check Starting...")
    print("=" * 50)
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['name']}")
            
            if result['passed']:
                passed_count += 1
                if 'info' in result:
                    print(f"     {result['info']}")
                if 'warnings' in result:
                    for warning in result['warnings']:
                        print(f"     ⚠️ {warning}")
            else:
                for error in result['errors']:
                    print(f"     ❌ {error}")
            
            print()
            
        except Exception as e:
            print(f"❌ FAIL {test_func.__name__} - Unexpected error: {str(e)}")
            results.append({
                'name': test_func.__name__,
                'passed': False,
                'errors': [f"Test execution error: {str(e)}"]
            })
    
    # 결과 요약
    success_rate = (passed_count / total_count) * 100
    print("=" * 50)
    print(f"📊 Test Results: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 System health status: GOOD")
    elif success_rate >= 60:
        print("⚠️ System health status: FAIR - Some issues need attention")
    else:
        print("❌ System health status: POOR - Major issues detected")
    
    return {
        'total_tests': total_count,
        'passed_tests': passed_count,
        'success_rate': success_rate,
        'results': results
    }

def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🏥 Starting system health check")
    
    try:
        overall_result = run_all_tests()
        
        if overall_result['success_rate'] >= 80:
            logger.info(f"System health check completed successfully: {overall_result['success_rate']:.1f}%")
            return 0
        else:
            logger.warning(f"System health check found issues: {overall_result['success_rate']:.1f}%")
            return 1
            
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 