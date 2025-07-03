#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 모델 다운로드 스크립트
안정적인 모델 다운로드 및 무결성 검증
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

def download_file(url: str, filepath: Path, expected_hash: str = None) -> bool:
    """파일 다운로드 및 해시 검증"""
    try:
        print(f"📥 다운로드 중: {url}")
        
        # 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # 해시 검증
        if expected_hash:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            if file_hash != expected_hash:
                print(f"❌ 해시 불일치: {filepath}")
                filepath.unlink()  # 손상된 파일 삭제
                return False
            else:
                print(f"✅ 해시 검증 성공: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def main():
    """메인 함수"""
    models_dir = project_root / 'models' / 'weights'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 다운로드할 모델 목록 (URL, 파일명, 예상 해시)
    models = [
        {
            'name': 'YuNet',
            'url': 'https://github.com/ShiqiYu/libfacedetection/raw/master/models/yunet_120x160.onnx',
            'filename': 'face_detection_yunet_2023mar.onnx',
            'hash': None  # 해시가 알려지지 않은 경우
        },
        {
            'name': 'UltraFace',
            'url': 'https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx',
            'filename': 'ultraface_rfb_320_robust.onnx',
            'hash': None
        }
    ]
    
    print("🚀 ONNX 모델 다운로드 시작")
    print("="*60)
    
    success_count = 0
    total_count = len(models)
    
    for model in models:
        filepath = models_dir / model['filename']
        
        # 이미 존재하는 파일 체크
        if filepath.exists():
            print(f"✅ 이미 존재: {model['name']} ({model['filename']})")
            success_count += 1
            continue
        
        # 다운로드
        if download_file(model['url'], filepath, model['hash']):
            print(f"✅ 다운로드 성공: {model['name']}")
            success_count += 1
        else:
            print(f"❌ 다운로드 실패: {model['name']}")
    
    print("\n" + "="*60)
    print(f"📊 결과: {success_count}/{total_count} 모델 다운로드 성공")
    print("="*60)
    
    if success_count < total_count:
        print("\n⚠️  일부 모델 다운로드 실패")
        print("   - 네트워크 연결 확인")
        print("   - 수동으로 모델 파일을 models/weights/ 폴더에 배치")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 