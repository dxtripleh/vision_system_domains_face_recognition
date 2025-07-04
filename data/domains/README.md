# Domains 폴더 - 도메인별 데이터 관리

## 📋 개요

이 폴더는 비전 시스템의 각 도메인별 런타임 데이터를 저장합니다.
각 도메인은 독립적으로 관리되며, 9단계 파이프라인에 따라 데이터가 체계적으로 저장됩니다.

## 🏗️ 폴더 구조

```
domains/
├── humanoid/                    # 인간형 도메인
│   ├── face_recognition/        # 얼굴인식 데이터
│   │   ├── 1_raw/              # 1단계: 원본 데이터
│   │   ├── 2_extracted/        # 2단계: 추출된 특이점
│   │   ├── 3_clustered/        # 3단계: 그룹핑된 데이터
│   │   ├── 4_labeled/          # 4단계: 라벨링된 데이터
│   │   ├── 5_embeddings/       # 5단계: 임베딩 벡터
│   │   ├── 6_realtime/         # 6단계: 실시간 인식
│   │   ├── 7_database/         # 7단계: 데이터베이스
│   │   ├── 8_monitoring/       # 8단계: 성능 모니터링
│   │   ├── 9_learning/         # 9단계: 지속적 학습
│   │   ├── cache/              # 캐시 데이터
│   │   ├── models/             # 도메인별 모델
│   │   └── traceability/       # 추적성 데이터
│   └── README.md
└── factory/                     # 공장 도메인
    ├── defect_detection/        # 불량 검출 데이터
    │   ├── 6_realtime/         # 6단계: 실시간 검출
    │   ├── 7_database/         # 7단계: 데이터베이스
    │   ├── 8_monitoring/       # 8단계: 성능 모니터링
    │   └── 9_learning/         # 9단계: 지속적 학습
    └── README.md
```

## 🎯 도메인별 특성

### Humanoid 도메인 (인간형)
- **주요 기능**: 얼굴인식, 감정 인식, 자세 추정
- **데이터 특성**: 개인정보 포함, GDPR 준수 필요
- **처리 속도**: 실시간 (30 FPS)
- **정확도 요구**: 95% 이상

### Factory 도메인 (공장)
- **주요 기능**: 불량 검출, 품질 관리, 재고 추적
- **데이터 특성**: 제품 정보, 품질 지표
- **처리 속도**: 실시간 (25 FPS)
- **정확도 요구**: 98% 이상

## 📊 9단계 파이프라인 데이터 흐름

### 1~5단계: 데이터 준비 단계
```
1_raw/ → 2_extracted/ → 3_clustered/ → 4_labeled/ → 5_embeddings/
```

### 6~9단계: 운영 단계
```
6_realtime/ → 7_database/ → 8_monitoring/ → 9_learning/
```

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# Humanoid 도메인
# 패턴: {timestamp}_{person_id}_{feature}.{ext}
# 예시: 20250704_133022_person001_face.jpg

# Factory 도메인  
# 패턴: {timestamp}_{product_id}_{defect_type}.{ext}
# 예시: 20250704_133022_product001_scratch.jpg
```

### 🗂️ 폴더별 접근 권한
- **1~5단계**: 읽기/쓰기 (데이터 준비)
- **6~9단계**: 읽기/쓰기 (운영)
- **cache/**: 읽기/쓰기 (자동 정리)
- **models/**: 읽기 전용 (관리자만 쓰기)

### 🔄 자동 정리 정책
```python
DOMAIN_CLEANUP_POLICIES = {
    '1_raw/': {'max_age_days': 7, 'auto_cleanup': True},
    '2_extracted/': {'max_age_days': 14, 'auto_cleanup': True},
    '3_clustered/': {'max_age_days': 30, 'auto_cleanup': True},
    '4_labeled/': {'max_age_days': 60, 'auto_cleanup': False},
    '5_embeddings/': {'max_age_days': 90, 'auto_cleanup': False},
    '6_realtime/': {'max_age_hours': 24, 'auto_cleanup': True},
    '7_database/': {'max_age_days': 365, 'auto_cleanup': False},
    '8_monitoring/': {'max_age_days': 90, 'auto_cleanup': True},
    '9_learning/': {'max_age_days': 180, 'auto_cleanup': False},
    'cache/': {'max_age_hours': 12, 'auto_cleanup': True}
}
```

## 🔍 데이터 추적성

### 📋 메타데이터 관리
- 모든 데이터 파일은 메타데이터와 함께 저장
- 생성 시간, 수정 시간, 소스 정보 기록
- 데이터 품질 점수 자동 계산

### 🔗 데이터 연결
- 원본 데이터 → 처리된 데이터 → 결과 데이터 연결
- 각 단계별 데이터 간 관계 추적
- 데이터 무결성 검증

## 🚫 절대 금지 사항

### ❌ 도메인 간 데이터 혼재 금지
```python
# 금지 사항
domains/humanoid/face_recognition/1_raw/factory_data.jpg  # ❌ 금지

# 올바른 방법
domains/humanoid/face_recognition/1_raw/humanoid_data.jpg  # ✅ 허용
domains/factory/defect_detection/1_raw/factory_data.jpg    # ✅ 허용
```

### ❌ 하드코딩된 경로 사용 금지
```python
# 금지 사항
path = "C:/Users/user/data/domains/humanoid"  # ❌ 금지

# 올바른 방법
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
domain_path = project_root / "data" / "domains" / "humanoid"  # ✅ 허용
```

## 📈 성능 최적화

### 💾 저장소 최적화
- 압축 알고리즘 사용 (ZIP, LZ4)
- 중복 데이터 제거
- 계층적 저장 구조

### ⚡ 접근 속도 최적화
- 인덱스 파일 생성
- 캐시 시스템 활용
- 병렬 처리 지원

## 🔒 보안 및 개인정보 보호

### 🔐 데이터 암호화
- 민감한 데이터는 암호화 저장
- 접근 권한 관리
- 감사 로그 기록

### 🛡️ GDPR 준수 (Humanoid 도메인)
- 개인정보 자동 익명화
- 데이터 보존 기간 관리
- 사용자 동의 추적

## 📊 모니터링 및 알림

### 📈 저장소 모니터링
- 도메인별 디스크 사용량 모니터링
- 파일 접근 패턴 분석
- 성능 지표 수집

### 🚨 알림 시스템
- 도메인별 저장소 용량 80% 초과 시 알림
- 데이터 손상 감지 시 알림
- 파이프라인 오류 시 알림

## 🔧 유지보수

### 🧹 정기 정리
- 매일: cache/ 폴더 정리
- 매주: 1~3단계 데이터 정리
- 매월: 전체 데이터 정리

### 📋 백업 정책
- 일일: 중요 데이터 백업
- 주간: 전체 도메인 데이터 백업
- 월간: 전체 시스템 백업

## 📚 관련 문서

- [Humanoid 도메인](./humanoid/)
- [Factory 도메인](./factory/)
- [9단계 파이프라인 가이드](../README.md) 