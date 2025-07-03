#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 구조 마이그레이션 스크립트

기존 data/temp 구조를 새로운 도메인별 구조로 이동합니다.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

class DataStructureMigrator:
    """데이터 구조 마이그레이션"""
    
    def __init__(self):
        self.project_root = project_root
        self.backup_dir = self.project_root / 'data' / 'backups' / f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 새로운 구조 정의
        self.new_structure = {
            'data/runtime': [
                'temp',
                'logs', 
                'output'
            ],
            'data/domains/face_recognition': [
                'raw_input/captured',
                'raw_input/uploads', 
                'raw_input/manual',
                'detected_faces/auto_collected',
                'detected_faces/from_captured',
                'detected_faces/from_uploads',
                'staging/grouped',
                'staging/named',
                'staging/rejected',
                'processed/final',
                'processed/embeddings',
                'processed/registered'
            ],
            'data/shared': [
                'models',
                'cache'
            ]
        }
        
        # 마이그레이션 매핑
        self.migration_mapping = {
            'data/temp/captured_frames': 'data/domains/face_recognition/raw_input/captured',
            'data/temp/uploads': 'data/domains/face_recognition/raw_input/uploads',
            'data/temp/auto_collected': 'data/domains/face_recognition/detected_faces/auto_collected',
            'data/temp/face_staging': 'data/domains/face_recognition/staging/named',
            'data/temp/grouped': 'data/domains/face_recognition/staging/grouped',
            'data/temp/processed': 'data/domains/face_recognition/processed/final',
            'data/temp/quality_checked': 'data/domains/face_recognition/processed/final',
            'data/logs': 'data/runtime/logs',
            'data/output': 'data/runtime/output'
        }
    
    def migrate(self, create_backup: bool = True) -> bool:
        """마이그레이션 실행"""
        print("🚀 데이터 구조 마이그레이션 시작")
        print("="*60)
        
        try:
            # 1. 백업 생성
            if create_backup:
                self._create_backup()
            
            # 2. 새 구조 생성
            self._create_new_structure()
            
            # 3. 데이터 이동
            self._migrate_data()
            
            # 4. 설정 파일 생성
            self._create_config_files()
            
            # 5. README 파일 생성
            self._create_readme_files()
            
            # 6. 마이그레이션 보고서 생성
            self._generate_migration_report()
            
            print("\n✅ 마이그레이션 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 마이그레이션 실패: {str(e)}")
            return False
    
    def _create_backup(self):
        """기존 데이터 백업"""
        print("📦 기존 데이터 백업 중...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # data 폴더 전체 백업
        data_dir = self.project_root / 'data'
        if data_dir.exists():
            backup_data_dir = self.backup_dir / 'data'
            shutil.copytree(data_dir, backup_data_dir)
            print(f"   ✅ 백업 완료: {self.backup_dir}")
    
    def _create_new_structure(self):
        """새로운 폴더 구조 생성"""
        print("📁 새로운 폴더 구조 생성 중...")
        
        created_count = 0
        for base_path, subdirs in self.new_structure.items():
            for subdir in subdirs:
                full_path = self.project_root / base_path / subdir
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_count += 1
                    print(f"   📂 생성: {full_path}")
        
        print(f"   ✅ {created_count}개 폴더 생성 완료")
    
    def _migrate_data(self):
        """데이터 이동"""
        print("📦 데이터 이동 중...")
        
        moved_count = 0
        for old_path, new_path in self.migration_mapping.items():
            old_full_path = self.project_root / old_path
            new_full_path = self.project_root / new_path
            
            if old_full_path.exists():
                # 대상 폴더가 없으면 생성
                new_full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일/폴더 이동
                if old_full_path.is_file():
                    shutil.move(str(old_full_path), str(new_full_path))
                else:
                    # 폴더의 경우 내용물만 이동
                    if new_full_path.exists():
                        # 대상이 이미 있으면 내용 병합
                        for item in old_full_path.iterdir():
                            dest_item = new_full_path / item.name
                            if dest_item.exists():
                                if dest_item.is_dir():
                                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(item, dest_item)
                            else:
                                shutil.move(str(item), str(dest_item))
                        
                        # 빈 폴더 제거
                        if not any(old_full_path.iterdir()):
                            old_full_path.rmdir()
                    else:
                        shutil.move(str(old_full_path), str(new_full_path))
                
                moved_count += 1
                print(f"   📦 이동: {old_path} → {new_path}")
        
        print(f"   ✅ {moved_count}개 항목 이동 완료")
    
    def _create_config_files(self):
        """도메인별 설정 파일 생성"""
        print("⚙️  설정 파일 생성 중...")
        
        # 얼굴인식 도메인 설정
        face_config = {
            'domain': 'face_recognition',
            'version': '1.0.0',
            'data_structure': {
                'raw_input': {
                    'captured': 'Camera captured frames',
                    'uploads': 'User uploaded files',
                    'manual': 'Manual capture with c key'
                },
                'detected_faces': {
                    'auto_collected': 'Auto mode detected faces',
                    'from_captured': 'Faces detected from captured frames',
                    'from_uploads': 'Faces detected from uploaded files'
                },
                'staging': {
                    'grouped': 'AI grouped faces waiting for naming',
                    'named': 'Named faces waiting for quality check',
                    'rejected': 'Quality check failed faces'
                },
                'processed': {
                    'final': 'Final processed face data',
                    'embeddings': 'Extracted embeddings',
                    'registered': 'System registered faces'
                }
            },
            'auto_cleanup': {
                'raw_input': {'days': 7},
                'detected_faces': {'days': 30},
                'staging': {'days': 90}
            }
        }
        
        config_file = self.project_root / 'data' / 'domains' / 'face_recognition' / 'config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(face_config, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 설정 파일 생성: {config_file}")
    
    def _create_readme_files(self):
        """README 파일 생성"""
        print("📝 README 파일 생성 중...")
        
        # 메인 데이터 README
        main_readme = """# 데이터 구조 가이드

## 📁 폴더 구조

### runtime/ - 런타임 데이터
- `temp/` - 임시 파일 (24시간 자동 정리)
- `logs/` - 시스템 로그
- `output/` - 최종 결과물

### domains/ - 도메인별 데이터
각 도메인은 독립적인 데이터 구조를 가집니다.

### shared/ - 공유 데이터
- `models/` - 도메인 간 공유 모델
- `cache/` - 공유 캐시

## 🔄 데이터 플로우
1. 입력 → raw_input/
2. 검출 → detected_*/
3. 그룹핑/이름지정 → staging/
4. 최종처리 → processed/
"""
        
        main_readme_file = self.project_root / 'data' / 'README.md'
        with open(main_readme_file, 'w', encoding='utf-8') as f:
            f.write(main_readme)
        
        # 얼굴인식 도메인 README
        face_readme = """# 얼굴인식 도메인 데이터

## 📁 폴더 구조

### raw_input/ - 원본 입력
- `captured/` - s키로 저장된 카메라 캡처 프레임
- `uploads/` - 사용자가 직접 업로드한 파일
- `manual/` - c키로 수동 캡처한 프레임

### detected_faces/ - 얼굴 검출 결과
- `auto_collected/` - 자동 모드에서 검출된 얼굴들
- `from_captured/` - captured에서 검출된 얼굴들
- `from_uploads/` - uploads에서 검출된 얼굴들

### staging/ - 처리 대기
- `grouped/` - AI 그룹핑 완료, 이름 입력 대기
- `named/` - 이름 지정 완료, 품질 검증 대기
- `rejected/` - 품질 검증 실패

### processed/ - 최종 처리 완료
- `final/` - 최종 처리 완료된 얼굴 데이터
- `embeddings/` - 임베딩 추출 완료
- `registered/` - 시스템 등록 완료

## 🔄 데이터 플로우

### 자동 모드
1. 얼굴 감지 → detected_faces/auto_collected/
2. AI 그룹핑 → staging/grouped/
3. 이름 입력 → staging/named/
4. 품질 검증 → processed/final/

### 수동 모드
1. s키 저장 → raw_input/captured/
2. 얼굴 검출 → detected_faces/from_captured/
3. 이름 입력 → staging/named/
4. 품질 검증 → processed/final/

또는

1. c키 캡처 → raw_input/manual/ (바로 이름 지정)
2. 품질 검증 → processed/final/
"""
        
        face_readme_file = self.project_root / 'data' / 'domains' / 'face_recognition' / 'README.md'
        with open(face_readme_file, 'w', encoding='utf-8') as f:
            f.write(face_readme)
        
        print("   ✅ README 파일 생성 완료")
    
    def _generate_migration_report(self):
        """마이그레이션 보고서 생성"""
        report = {
            'migration_date': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'new_structure': self.new_structure,
            'migration_mapping': self.migration_mapping,
            'status': 'completed'
        }
        
        report_file = self.project_root / 'data' / 'migration_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 마이그레이션 보고서: {report_file}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 구조 마이그레이션")
    parser.add_argument("--no-backup", action="store_true", help="백업 생성 안함")
    parser.add_argument("--dry-run", action="store_true", help="실제 실행 없이 계획만 표시")
    args = parser.parse_args()
    
    migrator = DataStructureMigrator()
    
    if args.dry_run:
        print("🔍 마이그레이션 계획 (실제 실행 안함):")
        print("="*50)
        
        print("\n📁 생성될 폴더:")
        for base_path, subdirs in migrator.new_structure.items():
            print(f"  {base_path}/")
            for subdir in subdirs:
                print(f"    └── {subdir}/")
        
        print("\n📦 이동될 데이터:")
        for old_path, new_path in migrator.migration_mapping.items():
            exists = "✅" if (project_root / old_path).exists() else "❌"
            print(f"  {exists} {old_path} → {new_path}")
        
        return
    
    # 실제 마이그레이션 실행
    success = migrator.migrate(create_backup=not args.no_backup)
    
    if success:
        print("\n🎉 마이그레이션이 성공적으로 완료되었습니다!")
        print("💡 이제 새로운 구조로 시스템을 사용할 수 있습니다.")
    else:
        print("\n❌ 마이그레이션이 실패했습니다.")
        print("💡 백업에서 복원하거나 수동으로 수정하세요.")

if __name__ == "__main__":
    main() 