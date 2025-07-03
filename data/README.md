# Data 폴더 구조 및 관리 가이드

## 📁 개요

`data/` 폴더는 비전 시스템의 **런타임 데이터**를 저장하는 전용 공간입니다. 
이 폴더는 시스템이 실행되는 동안 생성되는 모든 임시 파일, 로그, 결과물을 체계적으로 관리합니다.

## 🎯 폴더의 목적

- **런타임 전용**: 시스템 실행 중에만 사용되는 데이터
- **자동 정리**: 일정 기간 후 자동으로 정리되는 임시 파일들
- **실행 결과**: 처리된 이미지, 인식 결과, 로그 파일들
- **백업 관리**: 중요한 데이터의 백업 및 복구

## 📂 폴더 구조

```
data/
├── domains/                    # 도메인별 데이터
│   └── face_recognition/      # 얼굴인식 도메인 데이터
│       ├── detected_faces/    # 감지된 얼굴 이미지들
│       ├── processed/         # 처리된 데이터
│       ├── raw_input/         # 원본 입력 데이터
│       └── staging/           # 임시 처리 데이터
├── logs/                      # 시스템 로그 파일들
├── runtime/                   # 런타임 전용 데이터
│   ├── logs/                  # 실행 중 로그
│   ├── output/                # 실행 결과물
│   └── temp/                  # 임시 파일들
├── shared/                    # 공유 데이터
│   ├── cache/                 # 캐시 데이터
│   └── models/                # 임시 모델 파일들
└── backups/                   # 백업 데이터
```

## 🔄 데이터 생명주기

### 1. 데이터 생성
- 시스템 실행 시 자동 생성
- 사용자 업로드 또는 카메라 캡처
- 처리 과정에서 생성되는 중간 파일들

### 2. 데이터 처리
- `staging/`: 임시 처리 단계
- `processed/`: 최종 처리 완료
- `detected_faces/`: 감지된 얼굴 저장

### 3. 데이터 정리
- **임시 파일**: 24시간 후 자동 삭제
- **로그 파일**: 30일 후 자동 삭제
- **결과물**: 사용자 설정에 따라 보관

## 📊 폴더별 상세 설명

### domains/face_recognition/
얼굴인식 도메인의 모든 데이터가 저장되는 곳입니다.

#### detected_faces/
- **auto_collected/**: AI가 자동으로 수집한 얼굴 이미지
- **from_captured/**: 카메라로 캡처한 얼굴 이미지
- **from_uploads/**: 사용자가 업로드한 이미지에서 추출한 얼굴
- **from_manual/**: 수동으로 선택한 얼굴 이미지

#### processed/
- **embeddings/**: 얼굴 특징 벡터 (임베딩) 데이터
- **final/**: 최종 처리 완료된 데이터
- **registered/**: 등록된 얼굴 데이터

#### raw_input/
- **captured/**: 카메라로 캡처한 원본 이미지
- **uploads/**: 사용자가 업로드한 원본 이미지

#### staging/
- **grouped/**: 그룹화된 얼굴 이미지들
- **named/**: 이름이 지정된 얼굴 이미지들
- **rejected/**: 품질 검사에서 거부된 이미지들

### logs/
시스템의 모든 로그 파일이 저장됩니다.
- **face_recognition/**: 얼굴인식 관련 로그
- **system/**: 시스템 전체 로그
- **error/**: 오류 로그

### runtime/
실행 중에만 사용되는 임시 데이터들입니다.
- **logs/**: 실행 중 생성되는 로그
- **output/**: 실행 결과물 (캡처된 프레임, 인식 결과)
- **temp/**: 임시 파일들

### shared/
여러 도메인에서 공유하는 데이터들입니다.
- **cache/**: 캐시 데이터 (빠른 접근을 위해)
- **models/**: 임시로 다운로드된 모델 파일들

### backups/
중요한 데이터의 백업 파일들입니다.
- **migration_*/**: 구조 변경 시 백업
- **restructure_*/**: 재구성 시 백업

## ⚠️ 주의사항

### 절대 금지 사항
1. **루트 폴더에 직접 파일 생성 금지**
   - 모든 파일은 적절한 하위 폴더에 저장
   - 임시 파일은 `data/temp/` 또는 `data/runtime/temp/`에 저장

2. **대용량 파일 주의**
   - 이미지 파일은 압축하여 저장
   - 불필요한 파일은 정기적으로 정리

3. **권한 관리**
   - 시스템 실행 중에는 파일 삭제 금지
   - 백업 파일은 수동으로만 삭제

### 권장 사항
1. **정기적인 정리**
   - 주 1회 임시 파일 정리
   - 월 1회 오래된 로그 파일 정리

2. **백업 관리**
   - 중요한 데이터는 `backups/` 폴더에 백업
   - 백업 날짜를 파일명에 포함

3. **모니터링**
   - 디스크 사용량 정기 확인
   - 로그 파일 크기 모니터링

## 🛠️ 유틸리티 스크립트

### 자동 정리 스크립트
```bash
# 임시 파일 정리
python scripts/maintenance/cleanup_temp_files.py

# 로그 파일 정리
python scripts/maintenance/cleanup_logs.py

# 백업 생성
python scripts/maintenance/create_backup.py
```

### 데이터 마이그레이션
```bash
# 구조 변경 시 데이터 마이그레이션
python scripts/migration/migrate_data_structure.py
```

## 📈 모니터링 및 관리

### 디스크 사용량 확인
```bash
# 폴더별 크기 확인
python scripts/utilities/check_disk_usage.py
```

### 데이터 품질 검사
```bash
# 데이터 무결성 검사
python scripts/utilities/validate_data_integrity.py
```

## 🔗 관련 문서

- [데이터 플로우 가이드](../docs/guides/DATA_FLOW_GUIDE.md)
- [배치 처리 가이드](../docs/guides/BATCH_PROCESSING_GUIDE.md)
- [프로젝트 구조 가이드](../docs/guides/PROJECT_ROOT_RULES.md)

---

**마지막 업데이트**: 2025-06-29
**버전**: 1.0
**관리자**: 시스템 관리자
