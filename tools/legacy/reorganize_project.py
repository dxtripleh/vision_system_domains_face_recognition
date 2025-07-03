#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프로젝트 구조 정리 스크립트.

잘못 배치된 파일들을 올바른 위치로 이동하고 데이터 저장소 구조를 정리합니다.
"""

import os
import shutil
from pathlib import Path

def create_proper_directories():
    """올바른 디렉토리 구조 생성"""
    directories = [
        # 얼굴인식 도메인 내부 구조
        "domains/face_recognition/data/storage/faces",
        "domains/face_recognition/data/storage/persons", 
        "domains/face_recognition/data/temp",
        "domains/face_recognition/runners/demos",
        "domains/face_recognition/runners/tools",
        
        # 시스템 레벨 도구들
        "tools/setup",
        "tools/testing",
        "tools/deployment",
        
        # 공통 데이터 (시스템 레벨)
        "data/logs",
        "data/temp",
        "data/backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 생성: {directory}")

def move_files():
    """파일들을 올바른 위치로 이동"""
    
    # 이동할 파일들과 목적지 정의
    file_moves = [
        # 실행 스크립트들을 적절한 위치로
        ("run_simple_demo.py", "domains/face_recognition/runners/demos/"),
        ("run_face_recognition_demo.py", "domains/face_recognition/runners/demos/"),
        ("run_face_registration.py", "domains/face_recognition/runners/data_collection/"),
        
        # 도구들을 tools 폴더로
        ("download_models.py", "tools/setup/"),
        
        # 문서들은 그대로 유지 (최상위)
        # CURRENT_STATUS.md, README.md 등은 최상위에 유지
    ]
    
    moved_count = 0
    
    for source, dest_dir in file_moves:
        source_path = Path(source)
        dest_dir_path = Path(dest_dir)
        dest_path = dest_dir_path / source_path.name
        
        if source_path.exists():
            try:
                # 목적지 디렉토리 생성
                dest_dir_path.mkdir(parents=True, exist_ok=True)
                
                # 파일 이동
                shutil.move(str(source_path), str(dest_path))
                print(f"📦 이동: {source} → {dest_path}")
                moved_count += 1
                
            except Exception as e:
                print(f"❌ 이동 실패: {source} → {dest_dir} ({str(e)})")
        else:
            print(f"⏭️ 파일 없음: {source}")
    
    print(f"\n📊 총 {moved_count}개 파일 이동 완료")

def migrate_data_storage():
    """데이터 저장소를 도메인별로 이동"""
    
    # 현재 data/storage의 내용을 domains/face_recognition/data/storage로 이동
    source_storage = Path("data/storage")
    dest_storage = Path("domains/face_recognition/data/storage")
    
    if source_storage.exists():
        print(f"\n📦 데이터 저장소 마이그레이션")
        print(f"   {source_storage} → {dest_storage}")
        
        # 목적지 디렉토리 생성
        dest_storage.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리들 이동
        for item in source_storage.iterdir():
            if item.is_dir():
                dest_item = dest_storage / item.name
                try:
                    if dest_item.exists():
                        # 기존 파일들과 병합
                        for file in item.rglob("*"):
                            if file.is_file():
                                relative_path = file.relative_to(item)
                                dest_file = dest_item / relative_path
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(file), str(dest_file))
                        shutil.rmtree(str(item))
                    else:
                        shutil.move(str(item), str(dest_item))
                    print(f"   ✅ {item.name} 이동 완료")
                except Exception as e:
                    print(f"   ❌ {item.name} 이동 실패: {str(e)}")
        
        # 빈 디렉토리 제거
        try:
            if not any(source_storage.iterdir()):
                source_storage.rmdir()
                print(f"   🗑️ 빈 디렉토리 제거: {source_storage}")
        except:
            pass
    else:
        print(f"⏭️ 데이터 저장소 없음: {source_storage}")

def update_import_paths():
    """이동된 파일들의 import 경로 업데이트"""
    
    files_to_update = [
        "domains/face_recognition/runners/demos/run_simple_demo.py",
        "domains/face_recognition/runners/demos/run_face_recognition_demo.py", 
        "domains/face_recognition/runners/data_collection/run_face_registration.py"
    ]
    
    print(f"\n🔧 Import 경로 업데이트")
    
    for file_path in files_to_update:
        path = Path(file_path)
        if path.exists():
            try:
                # 파일 읽기
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 경로 수정
                # 프로젝트 루트 경로 계산 (도메인 내부에서)
                old_root_path = "project_root = Path(__file__).parent"
                new_root_path = "project_root = Path(__file__).parent.parent.parent.parent.parent"
                
                if old_root_path in content:
                    content = content.replace(old_root_path, new_root_path)
                    
                    # 파일 저장
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"   ✅ 업데이트: {file_path}")
                else:
                    print(f"   ⏭️ 변경 불필요: {file_path}")
                    
            except Exception as e:
                print(f"   ❌ 업데이트 실패: {file_path} ({str(e)})")
        else:
            print(f"   ⏭️ 파일 없음: {file_path}")

def create_domain_storage_config():
    """도메인별 저장소 설정 파일 생성"""
    
    config_content = '''#!/usr/bin/env python
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

def ensure_directories():
    """필요한 디렉토리들 생성"""
    for path in STORAGE_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def get_storage_path(storage_type: str) -> Path:
    """저장소 타입별 경로 반환"""
    return STORAGE_PATHS.get(storage_type, STORAGE_PATHS["temp"])

if __name__ == "__main__":
    ensure_directories()
    print("✅ Face Recognition 도메인 저장소 디렉토리 생성 완료")
'''
    
    config_path = Path("domains/face_recognition/config/storage_config.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 도메인 저장소 설정 파일 생성: {config_path}")

def create_project_launcher():
    """프로젝트 런처 생성 (최상위에 하나만)"""
    
    launcher_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision System Project Launcher.

전체 비전 시스템의 통합 런처입니다.
"""

import sys
import os
from pathlib import Path

def main():
    """메인 런처"""
    print("🎯 Vision System Project Launcher")
    print("=" * 50)
    print("사용 가능한 명령:")
    print()
    print("📊 얼굴인식 (Face Recognition)")
    print("  python domains/face_recognition/runners/demos/run_simple_demo.py")
    print("  python domains/face_recognition/runners/demos/run_face_recognition_demo.py")
    print("  python domains/face_recognition/runners/data_collection/run_face_registration.py")
    print()
    print("🛠️ 시스템 도구")
    print("  python tools/setup/download_models.py")
    print("  python scripts/core/test/test_system_health.py")
    print()
    print("📚 문서")
    print("  CURRENT_STATUS.md - 현재 개발 상태")
    print("  README.md - 프로젝트 개요")
    print("=" * 50)

if __name__ == "__main__":
    main()
'''
    
    launcher_path = Path("launcher.py")
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print(f"✅ 프로젝트 런처 생성: {launcher_path}")

def main():
    """메인 실행 함수"""
    print("🔧 프로젝트 구조 정리 시작")
    print("=" * 60)
    
    # 1. 올바른 디렉토리 구조 생성
    print("1️⃣ 디렉토리 구조 생성")
    create_proper_directories()
    
    # 2. 파일들 이동
    print("\n2️⃣ 파일 이동")
    move_files()
    
    # 3. 데이터 저장소 마이그레이션
    print("\n3️⃣ 데이터 저장소 마이그레이션")
    migrate_data_storage()
    
    # 4. Import 경로 업데이트
    print("\n4️⃣ Import 경로 업데이트")
    update_import_paths()
    
    # 5. 도메인 저장소 설정 생성
    print("\n5️⃣ 도메인 저장소 설정 생성")
    create_domain_storage_config()
    
    # 6. 프로젝트 런처 생성
    print("\n6️⃣ 프로젝트 런처 생성")
    create_project_launcher()
    
    print("\n" + "=" * 60)
    print("✅ 프로젝트 구조 정리 완료!")
    print()
    print("🎯 정리된 구조:")
    print("├── domains/face_recognition/")
    print("│   ├── data/storage/          # 도메인별 데이터 저장소")
    print("│   ├── runners/demos/         # 데모 실행 파일들")
    print("│   └── runners/data_collection/  # 데이터 수집 도구들")
    print("├── tools/setup/               # 설정 도구들")
    print("├── data/                      # 시스템 공통 데이터")
    print("└── launcher.py                # 통합 런처")
    print()
    print("🚀 사용법:")
    print("  python launcher.py           # 사용 가능한 명령 보기")

if __name__ == "__main__":
    main() 