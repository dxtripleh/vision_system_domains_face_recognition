#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data 폴더 최종 정리 스크립트

기존 중복 구조를 완전히 정리하고 새로운 구조로 통합합니다.
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

class DataFinalizer:
    """Data 폴더 최종 정리기"""
    
    def __init__(self):
        self.project_root = project_root
        self.data_root = self.project_root / 'data'
        
        # 최종 정리 매핑
        self.cleanup_mapping = {
            # 기존 logs → runtime/logs
            'data/logs': 'data/runtime/logs',
            
            # 기존 output → runtime/output  
            'data/output': 'data/runtime/output',
            
            # 기존 temp → runtime/temp (내용만)
            'data/temp': 'data/runtime/temp',
            
            # 기존 test → runtime/temp/test_data
            'data/test': 'data/runtime/temp/test_data'
        }
        
        # 삭제할 빈 폴더들
        self.folders_to_remove = [
            'data/logs',
            'data/output', 
            'data/temp',
            'data/test'
        ]
    
    def finalize_cleanup(self) -> bool:
        """최종 정리 실행"""
        print("🧹 Data 폴더 최종 정리 시작")
        print("="*50)
        
        try:
            # 1. 기존 폴더 내용 이동
            self._move_existing_content()
            
            # 2. 빈 폴더 제거
            self._remove_empty_folders()
            
            # 3. 최종 구조 검증
            self._verify_final_structure()
            
            # 4. README 및 STRUCTURE 업데이트
            self._update_documentation()
            
            print("\n✅ Data 폴더 최종 정리 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 정리 실패: {str(e)}")
            return False
    
    def _move_existing_content(self):
        """기존 폴더 내용 이동"""
        print("📦 기존 폴더 내용 이동 중...")
        
        moved_count = 0
        
        for old_path, new_path in self.cleanup_mapping.items():
            old_full_path = self.project_root / old_path
            new_full_path = self.project_root / new_path
            
            if old_full_path.exists() and old_full_path.is_dir():
                # 대상 폴더 생성
                new_full_path.mkdir(parents=True, exist_ok=True)
                
                # 내용 이동
                for item in old_full_path.iterdir():
                    if item.name in ['README.md', 'STRUCTURE.md']:
                        continue  # 문서 파일은 건드리지 않음
                    
                    dest_item = new_full_path / item.name
                    
                    try:
                        if item.is_dir():
                            if dest_item.exists():
                                # 기존 폴더가 있으면 내용 병합
                                shutil.copytree(item, dest_item, dirs_exist_ok=True)
                                shutil.rmtree(item)
                            else:
                                shutil.move(str(item), str(dest_item))
                        else:
                            if dest_item.exists():
                                # 중복 파일 처리
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                name_parts = item.name.rsplit('.', 1)
                                if len(name_parts) == 2:
                                    new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                                else:
                                    new_name = f"{item.name}_{timestamp}"
                                dest_item = new_full_path / new_name
                            
                            shutil.move(str(item), str(dest_item))
                        
                        moved_count += 1
                        print(f"   📁 이동: {item.relative_to(self.project_root)} → {dest_item.relative_to(self.project_root)}")
                        
                    except Exception as e:
                        print(f"   ❌ 이동 실패: {item.name} - {str(e)}")
        
        print(f"   ✅ {moved_count}개 항목 이동 완료")
    
    def _remove_empty_folders(self):
        """빈 폴더 제거"""
        print("🗑️  빈 폴더 제거 중...")
        
        removed_count = 0
        
        for folder_path in self.folders_to_remove:
            full_path = self.project_root / folder_path
            
            if full_path.exists() and full_path.is_dir():
                # 폴더가 비어있는지 확인 (README 파일만 있어도 빈 것으로 간주)
                contents = [item for item in full_path.iterdir() 
                           if item.name not in ['README.md', 'STRUCTURE.md']]
                
                if not contents:
                    try:
                        # README 파일이 있으면 백업
                        readme_file = full_path / 'README.md'
                        if readme_file.exists():
                            backup_dir = self.data_root / 'backups' / 'old_readmes'
                            backup_dir.mkdir(parents=True, exist_ok=True)
                            backup_name = f"{full_path.name}_README.md"
                            shutil.copy2(readme_file, backup_dir / backup_name)
                        
                        shutil.rmtree(full_path)
                        removed_count += 1
                        print(f"   🗑️  제거: {folder_path}")
                        
                    except Exception as e:
                        print(f"   ❌ 제거 실패: {folder_path} - {str(e)}")
                else:
                    print(f"   ⚠️  건너뜀: {folder_path} (내용 있음: {len(contents)}개)")
        
        print(f"   ✅ {removed_count}개 폴더 제거 완료")
    
    def _verify_final_structure(self):
        """최종 구조 검증"""
        print("🔍 최종 구조 검증 중...")
        
        # 예상 구조
        expected_structure = {
            'data/runtime/temp': '임시 파일',
            'data/runtime/logs': '시스템 로그',
            'data/runtime/output': '결과 출력',
            'data/domains/face_recognition/raw_input/captured': '카메라 캡처',
            'data/domains/face_recognition/raw_input/uploads': '파일 업로드',
            'data/domains/face_recognition/raw_input/manual': '수동 캡처',
            'data/domains/face_recognition/detected_faces/auto_collected': '자동 수집',
            'data/domains/face_recognition/detected_faces/from_captured': '캡처 처리',
            'data/domains/face_recognition/detected_faces/from_uploads': '업로드 처리',
            'data/domains/face_recognition/staging/grouped': 'AI 그룹핑',
            'data/domains/face_recognition/staging/named': '이름 지정',
            'data/domains/face_recognition/staging/rejected': '품질 실패',
            'data/domains/face_recognition/processed/final': '최종 처리',
            'data/domains/face_recognition/processed/embeddings': '임베딩',
            'data/domains/face_recognition/processed/registered': '시스템 등록',
            'data/shared/models': '공유 모델',
            'data/shared/cache': '공유 캐시',
            'data/backups': '백업 파일'
        }
        
        missing_folders = []
        existing_folders = []
        
        for folder_path, description in expected_structure.items():
            full_path = self.project_root / folder_path
            if full_path.exists():
                existing_folders.append(folder_path)
            else:
                missing_folders.append(folder_path)
                # 누락된 폴더 생성
                full_path.mkdir(parents=True, exist_ok=True)
                
                # README 생성
                readme_file = full_path / 'README.md'
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {full_path.name}\n\n{description}\n")
        
        print(f"   ✅ 기존 폴더: {len(existing_folders)}개")
        print(f"   📁 생성된 폴더: {len(missing_folders)}개")
        
        if missing_folders:
            print("   생성된 폴더들:")
            for folder in missing_folders:
                print(f"     - {folder}")
    
    def _update_documentation(self):
        """문서 업데이트"""
        print("📝 문서 업데이트 중...")
        
        # 메인 data README 업데이트
        main_readme = """# 📁 DATA 폴더 - 런타임 데이터 v2.0

## 🎯 **목적**
시스템 실행 중에 생성되는 모든 런타임 데이터를 도메인별로 관리합니다.
학습용 데이터(`datasets/`)와는 구별되며, 도메인별 독립적인 데이터 관리를 제공합니다.

## 📂 **최종 구조**
```
data/
├── runtime/                    # 모든 도메인 공통 런타임 데이터
│   ├── temp/                  # 임시 파일 (24시간 자동 정리)
│   ├── logs/                  # 시스템 로그 (30일 보관)
│   └── output/                # 최종 결과물 (90일 보관)
├── domains/                   # 도메인별 독립 데이터
│   └── face_recognition/      # 얼굴인식 도메인
│       ├── raw_input/         # 원본 입력
│       │   ├── captured/      # s키로 저장된 카메라 캡처
│       │   ├── uploads/       # 사용자 직접 업로드
│       │   └── manual/        # c키로 수동 캡처
│       ├── detected_faces/    # 얼굴 검출 결과
│       │   ├── auto_collected/    # 자동 모드에서 검출
│       │   ├── from_captured/     # captured 처리 결과
│       │   └── from_uploads/      # uploads 처리 결과
│       ├── staging/           # 처리 대기
│       │   ├── grouped/       # AI 그룹핑 완료
│       │   ├── named/         # 이름 지정 완료
│       │   └── rejected/      # 품질 검증 실패
│       └── processed/         # 최종 처리
│           ├── final/         # 최종 처리 완료
│           ├── embeddings/    # 임베딩 추출 완료
│           └── registered/    # 시스템 등록 완료
├── shared/                    # 도메인 간 공유 데이터
│   ├── models/               # 공유 모델
│   └── cache/                # 공유 캐시
└── backups/                  # 백업 데이터
```

## 🔄 **데이터 플로우 (상세)**

### 1. **c키 수동 캡처 플로우**
```
카메라 화면에서 c키 → 얼굴 검출 → 이름 입력
↓
data/domains/face_recognition/raw_input/manual/
↓
품질 검증 → 임베딩 추출
↓
data/domains/face_recognition/processed/final/
↓
시스템 등록 (domains/face_recognition/data/storage/)
```

### 2. **s키 프레임 저장 플로우**
```
카메라 화면에서 s키 → 전체 프레임 저장
↓
data/domains/face_recognition/raw_input/captured/
↓
얼굴 검출 → data/domains/face_recognition/detected_faces/from_captured/
↓
이름 입력 → data/domains/face_recognition/staging/named/
↓
품질 검증 → 임베딩 추출
↓
data/domains/face_recognition/processed/final/
↓
시스템 등록 (domains/face_recognition/data/storage/)
```

### 3. **자동 모드 플로우**
```
자동 얼굴 검출 → data/domains/face_recognition/detected_faces/auto_collected/
↓
AI 그룹핑 → data/domains/face_recognition/staging/grouped/
↓
이름 입력 → data/domains/face_recognition/staging/named/
↓
품질 검증 → 임베딩 추출
↓
data/domains/face_recognition/processed/final/
↓
시스템 등록 (domains/face_recognition/data/storage/)
```

### 4. **파일 업로드 플로우**
```
파일 업로드 → data/domains/face_recognition/raw_input/uploads/
↓
얼굴 검출 → data/domains/face_recognition/detected_faces/from_uploads/
↓
AI 그룹핑 → data/domains/face_recognition/staging/grouped/
↓
이름 입력 → data/domains/face_recognition/staging/named/
↓
품질 검증 → 임베딩 추출
↓
data/domains/face_recognition/processed/final/
↓
시스템 등록 (domains/face_recognition/data/storage/)
```

## 📊 **processed/final → domains/face_recognition/data/storage 처리**

### processed/final 파일 형태
```json
{
  "face_id": "uuid-generated-id",
  "person_name": "사용자입력이름",
  "image_path": "relative/path/to/image.jpg",
  "embedding": [0.1, 0.2, 0.3, ...],  // 512차원 벡터
  "quality_score": 0.95,
  "metadata": {
    "capture_method": "manual|captured|upload|auto",
    "timestamp": "2025-06-29T21:50:00",
    "camera_id": "camera_0",
    "confidence": 0.87
  }
}
```

### domains/face_recognition/data/storage 저장 형태
```json
// faces/{face_id}.json
{
  "id": "uuid-generated-id",
  "embedding": [0.1, 0.2, 0.3, ...],
  "quality_score": 0.95,
  "person_id": "person-uuid",
  "created_at": "2025-06-29T21:50:00",
  "source_method": "manual"
}

// persons/{person_id}.json  
{
  "id": "person-uuid",
  "name": "사용자입력이름",
  "face_ids": ["face-uuid-1", "face-uuid-2"],
  "created_at": "2025-06-29T21:50:00",
  "updated_at": "2025-06-29T21:50:00"
}
```

### 처리 과정
1. **품질 검증**: processed/final에서 quality_score > 0.7 확인
2. **중복 검사**: 기존 임베딩과 유사도 비교 (threshold: 0.8)
3. **Person 매핑**: 같은 이름의 Person 찾거나 새로 생성
4. **Face 등록**: Face 엔티티 생성 및 Person과 연결
5. **인덱스 업데이트**: face_index.json, person_index.json 업데이트

## 🚀 **향후 확장**

새로운 도메인 추가 시:
```
data/domains/
├── face_recognition/     # 기존
├── factory_defect/       # 공장 불량 검출
└── powerline_inspection/ # 전선 검사
```

각 도메인은 동일한 구조를 따르되, 도메인 특성에 맞게 커스터마이징 가능합니다.

---
*최종 업데이트: 2025-06-29 v2.0*
"""
        
        main_readme_file = self.data_root / 'README.md'
        with open(main_readme_file, 'w', encoding='utf-8') as f:
            f.write(main_readme)
        
        print(f"   ✅ 메인 README 업데이트: {main_readme_file}")
        
        # STRUCTURE.md 업데이트
        structure_content = """# 데이터 구조 다이어그램 v2.0

## 🏗️ **최종 구조**

```
data/
├── runtime/                    # 런타임 데이터 (모든 도메인 공통)
│   ├── temp/                  # 임시 파일 (24h 자동 정리)
│   │   ├── processing_cache/   # 처리 중 캐시
│   │   ├── model_outputs/      # 모델 임시 출력
│   │   └── test_data/          # 테스트 데이터
│   ├── logs/                  # 시스템 로그 (30d 보관)
│   │   ├── system_YYYY-MM-DD.log
│   │   ├── error_YYYY-MM-DD.log
│   │   └── face_recognition/   # 도메인별 로그
│   └── output/                # 최종 결과물 (90d 보관)
│       ├── recognition_results/
│       ├── processed_images/
│       └── reports/
│
├── domains/                   # 도메인별 독립 데이터
│   └── face_recognition/      # 얼굴인식 도메인
│       ├── raw_input/         # 원본 입력
│       │   ├── captured/      # s키 저장 (전체 프레임)
│       │   ├── uploads/       # 파일 업로드
│       │   └── manual/        # c키 캡처 (얼굴만)
│       ├── detected_faces/    # 얼굴 검출 결과
│       │   ├── auto_collected/    # 자동 모드 검출
│       │   ├── from_captured/     # captured 처리
│       │   └── from_uploads/      # uploads 처리
│       ├── staging/           # 처리 대기
│       │   ├── grouped/       # AI 그룹핑 완료
│       │   ├── named/         # 이름 지정 완료 ✨
│       │   └── rejected/      # 품질 검증 실패
│       └── processed/         # 최종 처리
│           ├── final/         # 최종 처리 완료 ✨
│           ├── embeddings/    # 임베딩 추출
│           └── registered/    # 시스템 등록 완료
│
├── shared/                    # 도메인 간 공유
│   ├── models/               # 공유 모델 가중치
│   └── cache/                # 공유 캐시
│
└── backups/                  # 백업 데이터
    ├── daily/                # 일일 백업
    ├── weekly/               # 주간 백업
    └── migration_history/    # 마이그레이션 기록
```

## 🔄 **데이터 플로우 맵**

### 입력 경로들
```
📷 Camera (c키) ──────────► raw_input/manual/
📷 Camera (s키) ──────────► raw_input/captured/
📁 File Upload ───────────► raw_input/uploads/
🤖 Auto Detection ───────► detected_faces/auto_collected/
```

### 처리 파이프라인
```
raw_input/* ──► Face Detection ──► detected_faces/*
                     │
detected_faces/* ──► AI Grouping ──► staging/grouped/
                     │
staging/grouped/ ──► Name Input ──► staging/named/ ✨
                     │
staging/named/ ──► Quality Check ──► processed/final/ ✨
                     │                      │
                     └─► rejected/          │
                                           │
processed/final/ ──► Registration ──► domains/face_recognition/data/storage/
```

## 📊 **파일 형태 변화**

### 1. raw_input → detected_faces
```
Input:  full_frame_image.jpg (1920x1080)
Output: face_001.jpg (224x224) + metadata.json
```

### 2. detected_faces → staging/named
```
Input:  face_001.jpg + basic_metadata.json
Output: named_face_person1.jpg + enriched_metadata.json
```

### 3. staging/named → processed/final
```
Input:  named_face_person1.jpg + enriched_metadata.json
Output: processed_face_uuid.json (임베딩 포함)
```

### 4. processed/final → domains/face_recognition/data/storage
```
Input:  processed_face_uuid.json
Output: faces/{uuid}.json + persons/{uuid}.json
```

## 🎯 **핵심 전환점**

- **✨ staging/named/**: 사용자가 이름을 지정한 최종 단계
- **✨ processed/final/**: 시스템이 처리 완료한 최종 단계
- **🎯 data/storage/**: 실제 인식에 사용되는 데이터베이스

---
*구조 버전: v2.0 (2025-06-29)*
"""
        
        structure_file = self.data_root / 'STRUCTURE.md'
        with open(structure_file, 'w', encoding='utf-8') as f:
            f.write(structure_content)
        
        print(f"   ✅ STRUCTURE.md 업데이트: {structure_file}")

def main():
    """메인 함수"""
    finalizer = DataFinalizer()
    success = finalizer.finalize_cleanup()
    
    if success:
        print("\n🎉 Data 폴더 최종 정리가 완료되었습니다!")
    else:
        print("\n❌ 정리 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 