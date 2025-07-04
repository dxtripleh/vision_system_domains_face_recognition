#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Models Downloader.

얼굴인식 도메인에 필요한 ONNX 모델들을 다운로드합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class FaceModelDownloader:
    """얼굴인식 모델 다운로드 클래스"""
    
    def __init__(self):
        self.models_dir = project_root / "models" / "weights"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드할 모델 정의
        self.models = {
            "face_detection": {
                "filename": "face_detection_retinaface_widerface_20250628.onnx",
                "description": "얼굴 검출 모델 (RetinaFace 기반)",
                "size_mb": 1.2
            },
            "face_recognition": {
                "filename": "face_recognition_arcface_glint360k_20250628.onnx",
                "description": "얼굴 인식 모델 (ArcFace 기반)",
                "size_mb": 249.0
            }
        }
    
    def create_dummy_models(self) -> bool:
        """개발용 더미 모델 생성"""
        print("더미 모델 생성 중... (개발용)")
        
        for model_name, model_info in self.models.items():
            filepath = self.models_dir / model_info["filename"]
            
            if filepath.exists():
                print(f"이미 존재: {filepath}")
                continue
            
            # 더미 ONNX 모델 생성
            dummy_data = b"DUMMY_ONNX_MODEL_FOR_DEVELOPMENT_ONLY"
            
            try:
                with open(filepath, 'wb') as f:
                    f.write(dummy_data)
                
                print(f"더미 모델 생성: {filepath}")
                
            except Exception as e:
                print(f"더미 모델 생성 실패: {e}")
                return False
        
        print("더미 모델 생성 완료. 실제 모델로 교체 필요!")
        return True
    
    def list_models(self) -> List[Dict]:
        """다운로드 가능한 모델 목록"""
        models_list = []
        
        for model_name, model_info in self.models.items():
            filepath = self.models_dir / model_info["filename"]
            
            models_list.append({
                "name": model_name,
                "filename": model_info["filename"],
                "description": model_info["description"],
                "size_mb": model_info["size_mb"],
                "exists": filepath.exists(),
                "path": str(filepath)
            })
        
        return models_list

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="얼굴인식 모델 다운로드")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 표시")
    parser.add_argument("--dummy", action="store_true", help="더미 모델 생성 (개발용)")
    
    args = parser.parse_args()
    
    downloader = FaceModelDownloader()
    
    if args.list:
        models = downloader.list_models()
        print("\n=== 얼굴인식 모델 목록 ===")
        for model in models:
            status = "✓ 존재" if model["exists"] else "✗ 없음"
            print(f"{model['name']}: {model['description']} ({model['size_mb']:.1f}MB) - {status}")
        return
    
    if args.dummy:
        success = downloader.create_dummy_models()
        if success:
            print("✓ 더미 모델 생성 완료")
        else:
            print("✗ 더미 모델 생성 실패")
        return

if __name__ == "__main__":
    main() 