#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임시 폴더 구조 설정 스크립트

얼굴인식 워크플로우에 필요한 모든 임시 폴더들을 자동으로 생성합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

def setup_temp_directories():
    """임시 폴더 구조 설정"""
    import logging
    logger = logging.getLogger(__name__)
    setup_logging()
    
    # 필요한 임시 폴더들 정의
    temp_directories = [
        # 1단계 워크플로우용
        'data/temp/face_staging',           # 공통 허브 (분기점)
        'data/temp/auto_collected',         # 자동 수집기용
        'data/temp/uploads',                # 업로드 파일용
        
        # 처리 단계별 폴더
        'data/temp/processed',              # 처리된 임시 파일
        'data/temp/grouped',                # 그룹핑된 얼굴들
        'data/temp/quality_checked',        # 품질 검사 완료
        
        # 백업 및 로그
        'data/temp/backups',                # 백업 파일
        'data/logs/face_recognition',       # 얼굴인식 로그
        
        # 출력 결과
        'data/output/recognition_results',  # 인식 결과
        'data/output/captured_frames',      # 캡처된 프레임
        
        # 도메인별 임시 폴더
        'domains/face_recognition/data/temp',
        'domains/face_recognition/data/logs'
    ]
    
    created_count = 0
    existing_count = 0
    
    for directory in temp_directories:
        dir_path = project_root / directory
        
        if dir_path.exists():
            existing_count += 1
            logger.info(f"✓ 이미 존재: {directory}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
                logger.info(f"✓ 생성 완료: {directory}")
                
                # .gitkeep 파일 생성 (빈 폴더 유지용)
                gitkeep_file = dir_path / '.gitkeep'
                gitkeep_file.touch()
                
            except Exception as e:
                logger.error(f"✗ 생성 실패: {directory} - {str(e)}")
    
    # README 파일 생성
    create_temp_readme()
    
    logger.info(f"\n📊 결과 요약:")
    logger.info(f"   ✓ 새로 생성: {created_count}개 폴더")
    logger.info(f"   ✓ 기존 존재: {existing_count}개 폴더")
    logger.info(f"   ✓ 총 관리: {len(temp_directories)}개 폴더")
    
    return True

def create_temp_readme():
    """data/temp 폴더에 README 파일 생성"""
    readme_content = """# 📁 TEMP 폴더 - 임시 데이터

## 🎯 **목적**
얼굴인식 워크플로우에서 사용되는 모든 임시 데이터를 관리합니다.
자동 정리 시스템에 의해 주기적으로 정리됩니다.

## 📂 **구조**
```
data/temp/
├── face_staging/        # 🎯 공통 허브 (분기점)
│   ├── person_001/      # 그룹별 정리된 얼굴들
│   ├── person_002/
│   └── ungrouped/       # 미분류 얼굴들
├── auto_collected/      # 🤖 자동 수집기 결과
├── uploads/             # 📤 업로드된 파일들
├── processed/           # 🔄 처리된 임시 파일
├── grouped/             # 👥 그룹핑 작업 중
├── quality_checked/     # ✅ 품질 검사 완료
└── backups/             # 💾 백업 파일
```

## 🔄 **자동 정리 규칙**
- **face_staging/**: 수동 정리 (사용자가 직접 관리)
- **auto_collected/**: 24시간 후 자동 삭제
- **uploads/**: 처리 완료 후 7일 보관
- **processed/**: 24시간 후 자동 삭제
- **grouped/**: 처리 완료 후 즉시 삭제
- **quality_checked/**: 24시간 후 자동 삭제
- **backups/**: 30일 보관

## ⚠️ **주의사항**
1. 중요한 데이터는 반드시 정식 저장소로 이동
2. face_staging 폴더만 장기 보관 (수동 관리)
3. 나머지 폴더는 자동 정리 대상

---
*이 폴더는 자동 설정 스크립트에 의해 생성되었습니다.*
"""
    
    readme_path = project_root / 'data' / 'temp' / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

if __name__ == "__main__":
    print("🚀 임시 폴더 구조 설정 시작...")
    
    if setup_temp_directories():
        print("\n✅ 임시 폴더 구조 설정 완료!")
        print("\n📝 다음 단계:")
        print("   1️⃣ 카메라 연결 테스트")
        print("   2️⃣ 얼굴 캡처 테스트")
        print("   3️⃣ 자동 그룹핑 테스트")
    else:
        print("\n❌ 설정 중 오류가 발생했습니다.")
        sys.exit(1) 