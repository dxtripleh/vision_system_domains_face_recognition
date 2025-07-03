#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX Runtime 환경 진단 스크립트
"""

import sys
import os
from pathlib import Path

def check_onnx_environment():
    """ONNX Runtime 환경 확인"""
    print("=== ONNX Runtime 환경 진단 ===")
    
    # 1. Python 버전 확인
    print(f"Python 버전: {sys.version}")
    
    # 2. ONNX Runtime 설치 확인
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime 설치됨: {ort.__version__}")
        
        # 3. 사용 가능한 프로바이더 확인
        providers = ort.get_available_providers()
        print(f"사용 가능한 프로바이더: {providers}")
        
        # 4. 기본 프로바이더 확인
        default_provider = ort.get_device()
        print(f"기본 디바이스: {default_provider}")
        
        # 5. 세션 옵션 테스트
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0  # 모든 로그 출력
        
        print("✅ ONNX Runtime 환경 정상")
        return True
        
    except ImportError as e:
        print(f"❌ ONNX Runtime 설치 안됨: {e}")
        print("설치 명령: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ ONNX Runtime 오류: {e}")
        return False

def check_onnx_models():
    """ONNX 모델 파일 확인"""
    print("\n=== ONNX 모델 파일 확인 ===")
    
    model_files = [
        "models/weights/face_detection_retinaface_mobilenet025.onnx",
        "models/weights/face_detection_retinaface_resnet50.onnx",
        "models/weights/face_detection_scrfd_10g_20250628.onnx",
        "models/weights/face_recognition_mobilefacenet_20250628.onnx",
        "models/weights/face_detection_ultraface_rfb_320.onnx"
    ]
    
    for model_path in model_files:
        if Path(model_path).exists():
            size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {model_path} ({size:.1f}MB)")
        else:
            print(f"❌ {model_path} (파일 없음)")
    
    return model_files

def test_onnx_model_loading():
    """ONNX 모델 로딩 테스트"""
    print("\n=== ONNX 모델 로딩 테스트 ===")
    
    try:
        import onnxruntime as ort
        
        # CPU 전용으로 설정
        providers = ['CPUExecutionProvider']
        
        model_files = [
            "models/weights/face_detection_retinaface_resnet50.onnx",
            "models/weights/face_recognition_mobilefacenet_20250628.onnx"
        ]
        
        for model_path in model_files:
            if not Path(model_path).exists():
                print(f"❌ {model_path} 파일 없음")
                continue
                
            try:
                print(f"테스트 중: {model_path}")
                
                # 세션 옵션 설정 (CPU 전용, 로그 레벨 낮춤)
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 2  # 경고만 출력
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                
                # 모델 로딩
                session = ort.InferenceSession(
                    model_path, 
                    providers=providers,
                    sess_options=session_options
                )
                
                # 입력/출력 정보 확인
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()
                
                print(f"  ✅ 로딩 성공")
                print(f"  입력: {input_info.name}, 형태: {input_info.shape}")
                print(f"  출력 개수: {len(output_info)}")
                
                for i, output in enumerate(output_info):
                    print(f"  출력 {i}: {output.name}, 형태: {output.shape}")
                
            except Exception as e:
                print(f"  ❌ 로딩 실패: {e}")
                
    except Exception as e:
        print(f"❌ ONNX Runtime 테스트 실패: {e}")

def main():
    """메인 함수"""
    # 1. 환경 확인
    if not check_onnx_environment():
        return
    
    # 2. 모델 파일 확인
    check_onnx_models()
    
    # 3. 모델 로딩 테스트
    test_onnx_model_loading()
    
    print("\n=== 진단 완료 ===")

if __name__ == "__main__":
    main() 