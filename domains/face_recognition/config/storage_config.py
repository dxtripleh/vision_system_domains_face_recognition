#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Domain Storage Configuration.

얼굴인식 도메인의 데이터 저장소 설정입니다.
"""

from pathlib import Path

# 도메인 루트 경로
DOMAIN_ROOT = Path(__file__).parent.parent

# 데이터 저장 경로들
STORAGE_PATHS = {
    "faces": DOMAIN_ROOT / "data" / "storage" / "faces",
    "persons": DOMAIN_ROOT / "data" / "storage" / "persons",
    "temp": DOMAIN_ROOT / "data" / "temp",
    "logs": DOMAIN_ROOT / "data" / "logs",
    "models": DOMAIN_ROOT / "models",
    "configs": DOMAIN_ROOT / "config"
}

# 저장소 설정
STORAGE_CONFIG = {
    "face_repository": {
        "storage_path": str(STORAGE_PATHS["faces"]),
        "file_format": "json",
        "backup_enabled": True,
        "max_files_per_directory": 1000
    },
    "person_repository": {
        "storage_path": str(STORAGE_PATHS["persons"]), 
        "file_format": "json",
        "backup_enabled": True,
        "max_files_per_directory": 100
    }
}

def get_storage_path(storage_type: str) -> Path:
    """저장소 타입별 경로 반환"""
    return STORAGE_PATHS.get(storage_type, STORAGE_PATHS["temp"])

def ensure_directories():
    """필요한 디렉토리들 생성"""
    for path in STORAGE_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("✅ Face Recognition 도메인 저장소 디렉토리 생성 완료") 