# Humanoid 도메인 - 인간형 데이터 관리

## 📋 개요

이 폴더는 인간형 도메인의 모든 런타임 데이터를 저장합니다.
얼굴인식, 감정 인식, 자세 추정 등 인간과 관련된 비전 처리 데이터가 체계적으로 관리됩니다.

## 🏗️ 폴더 구조

```
humanoid/
├── face_recognition/            # 얼굴인식 데이터
│   ├── 1_raw/                  # 1단계: 원본 데이터
│   │   ├── uploads/            # 업로드된 파일
│   │   ├── captures/           # 카메라 캡처
│   │   └── imports/            # 외부 임포트
│   ├── 2_extracted/            # 2단계: 추출된 특이점
│   │   ├── features/           # 얼굴 특이점
│   │   └── metadata/           # 메타데이터
│   ├── 3_clustered/            # 3단계: 그룹핑된 데이터
│   │   ├── groups/             # 그룹별 데이터
│   │   └── metadata/           # 그룹 메타데이터
│   ├── 4_labeled/              # 4단계: 라벨링된 데이터
│   │   ├── groups/             # 라벨링된 그룹
│   │   └── unknown/            # 미분류 데이터
│   ├── 5_embeddings/           # 5단계: 임베딩 벡터
│   │   ├── index/              # 검색 인덱스
│   │   └── vectors/            # 벡터 데이터
│   ├── 6_realtime/             # 6단계: 실시간 인식
│   │   ├── streams/            # 실시간 스트림
│   │   ├── recognition_logs/   # 인식 로그
│   │   └── performance/        # 성능 데이터
│   ├── 7_database/             # 7단계: 데이터베이스
│   │   ├── vectors/            # 벡터 데이터
│   │   ├── logs/               # 데이터베이스 로그
│   │   └── backups/            # 백업 데이터
│   ├── 8_monitoring/           # 8단계: 성능 모니터링
│   │   ├── metrics/            # 성능 지표
│   │   ├── alerts/             # 알림 데이터
│   │   └── reports/            # 성능 리포트
│   ├── 9_learning/             # 9단계: 지속적 학습
│   │   ├── new_data/           # 새로운 학습 데이터
│   │   ├── model_updates/      # 모델 업데이트
│   │   └── training_logs/      # 학습 로그
│   ├── cache/                  # 캐시 데이터
│   ├── models/                 # 도메인별 모델
│   ├── traceability/           # 추적성 데이터
│   └── pipeline_progress.json  # 파이프라인 진행 상황
└── README.md
```

## 🎯 도메인 특성

### 주요 기능
- **얼굴인식**: 실시간 얼굴 검출 및 인식
- **감정 인식**: 얼굴 표정 기반 감정 분석
- **자세 추정**: 신체 자세 및 동작 인식
- **개인정보 보호**: GDPR 준수 데이터 관리

### 데이터 특성
- **개인정보 포함**: 얼굴 이미지, 개인 식별 정보
- **실시간 처리**: 30 FPS 이상의 실시간 처리
- **고정확도**: 95% 이상의 인식 정확도
- **보안 강화**: 암호화 및 익명화 처리

## 📊 데이터 흐름

### 1~5단계: 데이터 준비
```
1_raw/ → 2_extracted/ → 3_clustered/ → 4_labeled/ → 5_embeddings/
```

### 6~9단계: 운영
```
6_realtime/ → 7_database/ → 8_monitoring/ → 9_learning/
```

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# 기본 패턴: {timestamp}_{person_id}_{feature}.{ext}
# 예시: 20250704_133022_person001_face.jpg

# 단계별 패턴:
# 1단계: {timestamp}_{source}_{index}.{ext}
# 2단계: {raw_id}_f{feature_idx}.{ext}
# 3단계: g{group_id}_{index}.{ext}
# 4단계: {person_id}_{group_id}_{index}.{ext}
# 5단계: {person_id}_emb_{version}.npy
```

### 🗂️ 폴더별 접근 권한
- **1~5단계**: 읽기/쓰기 (데이터 준비)
- **6~9단계**: 읽기/쓰기 (운영)
- **cache/**: 읽기/쓰기 (자동 정리)
- **models/**: 읽기 전용 (관리자만 쓰기)
- **traceability/**: 읽기 전용 (시스템만 쓰기)

### 🔄 자동 정리 정책
```python
HUMANOID_CLEANUP_POLICIES = {
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

## 🔒 보안 및 개인정보 보호

### 🔐 데이터 암호화
- **저장 암호화**: 민감한 데이터는 AES-256 암호화
- **전송 암호화**: TLS 1.3 사용
- **접근 제어**: 역할 기반 접근 제어 (RBAC)

### 🛡️ GDPR 준수
- **개인정보 익명화**: 자동 얼굴 블러 처리
- **동의 관리**: 사용자 동의 추적 및 관리
- **데이터 보존**: 최소 보존 기간 설정
- **삭제 권리**: 사용자 요청 시 데이터 삭제

### 🎭 데이터 익명화
```python
ANONYMIZATION_RULES = {
    'face_blur': True,           # 얼굴 자동 블러
    'metadata_removal': True,    # 개인정보 메타데이터 제거
    'pseudonymization': True,    # 가명화 처리
    'consent_tracking': True     # 동의 추적
}
```

## 📈 성능 지표

### 실시간 성능 목표
- **FPS**: 30 FPS 이상
- **지연시간**: 100ms 이하
- **정확도**: 95% 이상
- **메모리 사용량**: 2GB 이하

### 품질 지표
- **얼굴 검출률**: 98% 이상
- **인식 정확도**: 95% 이상
- **오탐률**: 2% 이하
- **미탐률**: 1% 이하

## 🔍 데이터 추적성

### 📋 메타데이터 관리
- **생성 정보**: 시간, 소스, 처리 단계
- **품질 정보**: 해상도, 밝기, 대비, 노이즈
- **개인정보**: 동의 상태, 익명화 여부
- **처리 정보**: 모델 버전, 파라미터, 결과

### 🔗 데이터 연결
- **원본 연결**: 1단계 → 2단계 → 3단계 연결
- **그룹 연결**: 3단계 → 4단계 → 5단계 연결
- **운영 연결**: 6단계 → 7단계 → 8단계 → 9단계 연결

## 🚫 절대 금지 사항

### ❌ 개인정보 노출 금지
```python
# 금지 사항
# - 원본 얼굴 이미지를 외부로 전송
# - 개인정보가 포함된 로그 파일 생성
# - 암호화되지 않은 개인정보 저장

# 올바른 방법
# - 얼굴 블러 처리 후 전송
# - 개인정보 제거된 로그 생성
# - AES-256 암호화 후 저장
```

### ❌ GDPR 위반 금지
```python
# 금지 사항
# - 사용자 동의 없이 개인정보 수집
# - 보존 기간 초과 데이터 보관
# - 삭제 요청 무시

# 올바른 방법
# - 명시적 동의 후 수집
# - 자동 삭제 시스템 운영
# - 즉시 삭제 처리
```

## 📊 모니터링 및 알림

### 📈 성능 모니터링
- **실시간 성능**: FPS, 지연시간, 정확도
- **시스템 리소스**: CPU, GPU, 메모리 사용량
- **데이터 품질**: 이미지 품질, 라벨 정확도

### 🚨 알림 시스템
- **성능 저하**: FPS 25 이하, 정확도 90% 이하
- **보안 이벤트**: 무단 접근, 데이터 유출 시도
- **GDPR 위반**: 동의 만료, 보존 기간 초과

## 🔧 유지보수

### 🧹 정기 정리
- **매일**: cache/, 6_realtime/ 정리
- **매주**: 1~3단계 데이터 정리
- **매월**: 전체 데이터 품질 검사

### 📋 백업 정책
- **일일**: 중요 데이터 백업 (암호화)
- **주간**: 전체 도메인 데이터 백업
- **월간**: 전체 시스템 백업

## 📚 관련 문서

- [얼굴인식 데이터](./face_recognition/)
- [GDPR 준수 가이드](./face_recognition/GDPR_GUIDE.md)
- [성능 모니터링 가이드](./face_recognition/8_monitoring/README.md) 