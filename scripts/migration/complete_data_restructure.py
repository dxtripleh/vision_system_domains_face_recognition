#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data 폴더 완전 재구성 스크립트

기존 data 폴더를 새로운 도메인 구조로 완전히 재구성합니다.
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

class CompleteDataRestructurer:
    """Data 폴더 완전 재구성기"""
    
    def __init__(self):
        self.project_root = project_root
        self.data_root = self.project_root / 'data'
        self.backup_dir = self.project_root / 'data' / 'backups' / f'restructure_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 최종 목표 구조
        self.target_structure = {
            'data/runtime': {
                'temp': '임시 파일 (24시간 자동 정리)',
                'logs': '시스템 로그',
                'output': '최종 결과물'
            },
            'data/domains/face_recognition': {
                'raw_input/captured': 's키로 저장된 카메라 캡처',
                'raw_input/uploads': '사용자 직접 업로드',
                'raw_input/manual': 'c키로 수동 캡처',
                'detected_faces/auto_collected': '자동 모드 검출',
                'detected_faces/from_captured': 'captured 처리 결과',
                'detected_faces/from_uploads': 'uploads 처리 결과',
                'staging/grouped': 'AI 그룹핑 완료',
                'staging/named': '이름 지정 완료',
                'staging/rejected': '품질 검증 실패',
                'processed/final': '최종 처리 완료',
                'processed/embeddings': '임베딩 추출 완료',
                'processed/registered': '시스템 등록 완료'
            },
            'data/shared': {
                'models': '도메인 간 공유 모델',
                'cache': '공유 캐시'
            }
        }
        
        # 기존 데이터 매핑
        self.migration_mapping = {
            # 기존 temp 폴더들
            'data/temp/captured_frames': 'data/domains/face_recognition/raw_input/captured',
            'data/temp/uploads': 'data/domains/face_recognition/raw_input/uploads',
            'data/temp/auto_collected': 'data/domains/face_recognition/detected_faces/auto_collected',
            'data/temp/face_staging': 'data/domains/face_recognition/staging/named',
            'data/temp/grouped': 'data/domains/face_recognition/staging/grouped',
            'data/temp/processed': 'data/domains/face_recognition/processed/final',
            'data/temp/quality_checked': 'data/domains/face_recognition/processed/final',
            
            # 기존 최상위 폴더들
            'data/logs': 'data/runtime/logs',
            'data/output': 'data/runtime/output',
            
            # 테스트 데이터
            'data/test': 'data/runtime/temp/test_data'
        }
    
    def restructure(self, create_backup: bool = True) -> bool:
        """완전 재구성 실행"""
        print("🏗️  Data 폴더 완전 재구성 시작")
        print("="*60)
        
        try:
            # 1. 백업 생성
            if create_backup:
                self._create_backup()
            
            # 2. 새 구조 생성
            self._create_target_structure()
            
            # 3. 기존 데이터 이동
            self._migrate_existing_data()
            
            # 4. 기존 구조 정리
            self._cleanup_old_structure()
            
            # 5. 설정 파일 업데이트
            self._update_config_files()
            
            # 6. README 파일 생성
            self._create_documentation()
            
            # 7. 검증
            self._verify_structure()
            
            print("\n✅ Data 폴더 재구성 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 재구성 실패: {str(e)}")
            return False
    
    def _create_backup(self):
        """백업 생성"""
        print("📦 기존 data 폴더 백업 중...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 data 폴더 구조 기록
        structure_info = self._scan_current_structure()
        
        # 백업 정보 저장
        backup_info = {
            'backup_date': datetime.now().isoformat(),
            'original_structure': structure_info,
            'backup_location': str(self.backup_dir)
        }
        
        backup_info_file = self.backup_dir / 'backup_info.json'
        with open(backup_info_file, 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 백업 정보 저장: {backup_info_file}")
    
    def _scan_current_structure(self) -> Dict:
        """현재 구조 스캔"""
        structure = {}
        
        if self.data_root.exists():
            for item in self.data_root.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(self.data_root)
                    structure[str(rel_path)] = {
                        'type': 'file',
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                elif item.is_dir() and item != self.data_root:
                    rel_path = item.relative_to(self.data_root)
                    structure[str(rel_path)] = {
                        'type': 'directory',
                        'items': len(list(item.iterdir())) if item.exists() else 0
                    }
        
        return structure
    
    def _create_target_structure(self):
        """목표 구조 생성"""
        print("📁 새로운 구조 생성 중...")
        
        created_count = 0
        
        for base_path, subdirs in self.target_structure.items():
            for subdir, description in subdirs.items():
                full_path = self.project_root / base_path / subdir
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_count += 1
                    print(f"   📂 생성: {base_path}/{subdir}")
                
                # README 파일 생성
                readme_file = full_path / 'README.md'
                if not readme_file.exists():
                    with open(readme_file, 'w', encoding='utf-8') as f:
                        f.write(f"# {subdir}\n\n{description}\n")
        
        print(f"   ✅ {created_count}개 폴더 생성 완료")
    
    def _migrate_existing_data(self):
        """기존 데이터 이동"""
        print("📦 기존 데이터 이동 중...")
        
        moved_count = 0
        
        for old_path, new_path in self.migration_mapping.items():
            old_full_path = self.project_root / old_path
            new_full_path = self.project_root / new_path
            
            if old_full_path.exists():
                # 대상 폴더 생성
                new_full_path.parent.mkdir(parents=True, exist_ok=True)
                
                if old_full_path.is_file():
                    # 파일 이동
                    if new_full_path.exists():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name_parts = old_full_path.name.rsplit('.', 1)
                        if len(name_parts) == 2:
                            new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                        else:
                            new_name = f"{old_full_path.name}_{timestamp}"
                        new_full_path = new_full_path.parent / new_name
                    
                    shutil.move(str(old_full_path), str(new_full_path))
                    moved_count += 1
                    print(f"   📄 이동: {old_path} → {new_path}")
                
                elif old_full_path.is_dir():
                    # 폴더 내용 이동
                    if new_full_path.exists():
                        # 대상이 있으면 내용 병합
                        for item in old_full_path.iterdir():
                            dest_item = new_full_path / item.name
                            if dest_item.exists():
                                if dest_item.is_dir():
                                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                                else:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    name_parts = item.name.rsplit('.', 1)
                                    if len(name_parts) == 2:
                                        new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                                    else:
                                        new_name = f"{item.name}_{timestamp}"
                                    shutil.move(str(item), str(new_full_path / new_name))
                            else:
                                shutil.move(str(item), str(dest_item))
                        
                        # 빈 폴더 제거
                        if not any(old_full_path.iterdir()):
                            old_full_path.rmdir()
                    else:
                        shutil.move(str(old_full_path), str(new_full_path))
                    
                    moved_count += 1
                    print(f"   📁 이동: {old_path} → {new_path}")
        
        print(f"   ✅ {moved_count}개 항목 이동 완료")
    
    def _cleanup_old_structure(self):
        """기존 구조 정리"""
        print("🧹 기존 구조 정리 중...")
        
        # 빈 temp 폴더 제거
        temp_dir = self.data_root / 'temp'
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            print("   🗑️  빈 temp 폴더 제거")
        
        # 기존 STRUCTURE.md 백업으로 이동
        old_structure_md = self.data_root / 'STRUCTURE.md'
        if old_structure_md.exists():
            backup_structure = self.backup_dir / 'old_STRUCTURE.md'
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_structure_md), str(backup_structure))
            print("   📦 기존 STRUCTURE.md 백업으로 이동")
    
    def _update_config_files(self):
        """설정 파일 업데이트"""
        print("⚙️  설정 파일 업데이트 중...")
        
        # 도메인별 설정 파일 생성
        face_config = {
            'domain': 'face_recognition',
            'version': '2.0.0',
            'restructure_date': datetime.now().isoformat(),
            'data_paths': {
                'raw_input': {
                    'captured': 'data/domains/face_recognition/raw_input/captured',
                    'uploads': 'data/domains/face_recognition/raw_input/uploads',
                    'manual': 'data/domains/face_recognition/raw_input/manual'
                },
                'detected_faces': {
                    'auto_collected': 'data/domains/face_recognition/detected_faces/auto_collected',
                    'from_captured': 'data/domains/face_recognition/detected_faces/from_captured',
                    'from_uploads': 'data/domains/face_recognition/detected_faces/from_uploads'
                },
                'staging': {
                    'grouped': 'data/domains/face_recognition/staging/grouped',
                    'named': 'data/domains/face_recognition/staging/named',
                    'rejected': 'data/domains/face_recognition/staging/rejected'
                },
                'processed': {
                    'final': 'data/domains/face_recognition/processed/final',
                    'embeddings': 'data/domains/face_recognition/processed/embeddings',
                    'registered': 'data/domains/face_recognition/processed/registered'
                }
            },
            'auto_cleanup': {
                'raw_input': {'days': 7},
                'detected_faces': {'days': 30},
                'staging': {'days': 90}
            }
        }
        
        config_file = self.data_root / 'domains' / 'face_recognition' / 'config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(face_config, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 얼굴인식 도메인 설정: {config_file}")
        
        # 런타임 설정 파일
        runtime_config = {
            'version': '1.0.0',
            'paths': {
                'temp': 'data/runtime/temp',
                'logs': 'data/runtime/logs',
                'output': 'data/runtime/output'
            },
            'cleanup': {
                'temp_retention_hours': 24,
                'log_retention_days': 30,
                'output_retention_days': 90
            }
        }
        
        runtime_config_file = self.data_root / 'runtime' / 'config.json'
        with open(runtime_config_file, 'w', encoding='utf-8') as f:
            json.dump(runtime_config, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 런타임 설정: {runtime_config_file}")
    
    def _create_documentation(self):
        """문서 생성"""
        print("📝 문서 생성 중...")
        
        # 메인 data README 업데이트
        main_readme = """# 데이터 구조 가이드 v2.0

## 🎯 새로운 도메인 기반 구조

이 구조는 향후 다양한 비전 도메인(공장 불량 검출, 전선 검사 등) 확장을 고려하여 설계되었습니다.

## 📁 폴더 구조

### runtime/ - 런타임 데이터 (모든 도메인 공통)
- `temp/` - 임시 파일 (24시간 자동 정리)
- `logs/` - 시스템 로그 (30일 보관)
- `output/` - 최종 결과물 (90일 보관)

### domains/ - 도메인별 독립 데이터
각 도메인은 완전히 독립적인 데이터 구조를 가집니다.

#### face_recognition/ - 얼굴인식 도메인
```
raw_input/          # 원본 입력
├── captured/       # s키로 저장된 카메라 캡처
├── uploads/        # 사용자 직접 업로드
└── manual/         # c키로 수동 캡처

detected_faces/     # 얼굴 검출 결과
├── auto_collected/ # 자동 모드에서 검출
├── from_captured/  # captured에서 검출
└── from_uploads/   # uploads에서 검출

staging/            # 처리 대기
├── grouped/        # AI 그룹핑 완료
├── named/          # 이름 지정 완료
└── rejected/       # 품질 검증 실패

processed/          # 최종 처리 완료
├── final/          # 최종 처리 완료
├── embeddings/     # 임베딩 추출 완료
└── registered/     # 시스템 등록 완료
```

### shared/ - 도메인 간 공유 데이터
- `models/` - 도메인 간 공유 모델
- `cache/` - 공유 캐시

## 🔄 데이터 플로우

### 자동 모드
1. 얼굴 감지 → detected_faces/auto_collected/
2. AI 그룹핑 → staging/grouped/
3. 이름 입력 → staging/named/
4. 품질 검증 → processed/final/

### 수동 모드
#### s키 플로우
1. 프레임 저장 → raw_input/captured/
2. 얼굴 검출 → detected_faces/from_captured/
3. 이름 입력 → staging/named/
4. 품질 검증 → processed/final/

#### c키 플로우
1. 얼굴 캡처 → raw_input/manual/
2. 바로 이름 입력 → staging/named/
3. 품질 검증 → processed/final/

#### 파일 업로드 플로우
1. 파일 업로드 → raw_input/uploads/
2. 얼굴 검출 → detected_faces/from_uploads/
3. AI 그룹핑 → staging/grouped/
4. 이름 입력 → staging/named/
5. 품질 검증 → processed/final/

## 🚀 향후 확장

새로운 도메인 추가 시:
```
domains/
├── face_recognition/    # 기존
├── factory_defect/      # 공장 불량 검출
└── powerline_inspection/ # 전선 검사
```

각 도메인은 동일한 구조를 따르되, 도메인 특성에 맞게 커스터마이징 가능합니다.
"""
        
        main_readme_file = self.data_root / 'README.md'
        with open(main_readme_file, 'w', encoding='utf-8') as f:
            f.write(main_readme)
        
        print(f"   ✅ 메인 README 업데이트: {main_readme_file}")
        
        # 구조 다이어그램 생성
        structure_diagram = """# 데이터 구조 다이어그램

```
data/
├── runtime/                    # 런타임 데이터
│   ├── temp/                  # 임시 파일 (24h)
│   ├── logs/                  # 시스템 로그 (30d)
│   └── output/                # 최종 결과물 (90d)
│
├── domains/                   # 도메인별 데이터
│   └── face_recognition/      # 얼굴인식 도메인
│       ├── raw_input/         # 원본 입력
│       │   ├── captured/      # s키 저장
│       │   ├── uploads/       # 파일 업로드
│       │   └── manual/        # c키 캡처
│       ├── detected_faces/    # 얼굴 검출 결과
│       │   ├── auto_collected/    # 자동 모드
│       │   ├── from_captured/     # captured 처리
│       │   └── from_uploads/      # uploads 처리
│       ├── staging/           # 처리 대기
│       │   ├── grouped/       # AI 그룹핑
│       │   ├── named/         # 이름 지정
│       │   └── rejected/      # 품질 실패
│       └── processed/         # 최종 처리
│           ├── final/         # 처리 완료
│           ├── embeddings/    # 임베딩
│           └── registered/    # 시스템 등록
│
├── shared/                    # 공유 데이터
│   ├── models/               # 공유 모델
│   └── cache/                # 공유 캐시
│
└── backups/                  # 백업 데이터
```
"""
        
        structure_file = self.data_root / 'STRUCTURE.md'
        with open(structure_file, 'w', encoding='utf-8') as f:
            f.write(structure_diagram)
        
        print(f"   ✅ 구조 다이어그램: {structure_file}")
    
    def _verify_structure(self):
        """구조 검증"""
        print("🔍 구조 검증 중...")
        
        errors = []
        warnings = []
        
        # 필수 폴더 검증
        required_paths = [
            'data/runtime/temp',
            'data/runtime/logs',
            'data/runtime/output',
            'data/domains/face_recognition/raw_input/captured',
            'data/domains/face_recognition/detected_faces/auto_collected',
            'data/domains/face_recognition/staging/named',
            'data/domains/face_recognition/processed/final',
            'data/shared/models'
        ]
        
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                errors.append(f"필수 폴더 누락: {path}")
        
        # 설정 파일 검증
        config_files = [
            'data/domains/face_recognition/config.json',
            'data/runtime/config.json'
        ]
        
        for config_file in config_files:
            full_path = self.project_root / config_file
            if not full_path.exists():
                warnings.append(f"설정 파일 누락: {config_file}")
        
        # 결과 출력
        if errors:
            print(f"   ❌ {len(errors)}개 오류 발견:")
            for error in errors:
                print(f"     - {error}")
        
        if warnings:
            print(f"   ⚠️  {len(warnings)}개 경고:")
            for warning in warnings:
                print(f"     - {warning}")
        
        if not errors and not warnings:
            print("   ✅ 구조 검증 완료 - 모든 것이 정상입니다!")
        
        return len(errors) == 0

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data 폴더 완전 재구성")
    parser.add_argument("--no-backup", action="store_true", help="백업 생성 안함")
    parser.add_argument("--verify-only", action="store_true", help="검증만 수행")
    args = parser.parse_args()
    
    restructurer = CompleteDataRestructurer()
    
    if args.verify_only:
        print("🔍 구조 검증만 수행합니다.")
        success = restructurer._verify_structure()
        sys.exit(0 if success else 1)
    
    success = restructurer.restructure(create_backup=not args.no_backup)
    
    if success:
        print("\n🎉 Data 폴더 재구성이 성공적으로 완료되었습니다!")
        print("💡 새로운 구조로 시스템을 사용할 수 있습니다.")
    else:
        print("\n❌ 재구성이 실패했습니다.")

if __name__ == "__main__":
    main() 