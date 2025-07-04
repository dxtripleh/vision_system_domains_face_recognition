# Factory 도메인 - 공장 데이터 관리

## 📋 개요

이 폴더는 공장 도메인의 모든 런타임 데이터를 저장합니다.
불량 검출, 품질 관리, 재고 추적 등 공장 자동화와 관련된 비전 처리 데이터가 체계적으로 관리됩니다.

## 🏗️ 폴더 구조

```
factory/
├── defect_detection/            # 불량 검출 데이터
│   ├── 6_realtime/             # 6단계: 실시간 검출
│   │   ├── streams/            # 실시간 스트림
│   │   ├── detection_logs/     # 검출 로그
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
│   └── README.md
└── README.md
```

## 🎯 도메인 특성

### 주요 기능
- **불량 검출**: 제품 불량 자동 검출 및 분류
- **품질 관리**: 품질 지표 실시간 모니터링
- **재고 추적**: 제품 재고 자동 추적
- **생산 최적화**: 생산성 향상을 위한 데이터 분석

### 데이터 특성
- **제품 정보**: 제품 ID, 배치 번호, 생산 라인
- **실시간 처리**: 25 FPS 이상의 실시간 처리
- **고정확도**: 98% 이상의 검출 정확도
- **품질 중심**: 제품 품질 보증을 위한 데이터

## 📊 데이터 흐름

### 6~9단계: 운영 단계
```
6_realtime/ → 7_database/ → 8_monitoring/ → 9_learning/
```

### 향후 확장 계획
```
1_raw/ → 2_extracted/ → 3_clustered/ → 4_labeled/ → 5_embeddings/
```

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# 기본 패턴: {timestamp}_{product_id}_{defect_type}.{ext}
# 예시: 20250704_133022_product001_scratch.jpg

# 단계별 패턴:
# 6단계: {camera_id}_{timestamp}_{frame_id}.{ext}
# 7단계: {product_id}_defect_{type}_{version}.json
# 8단계: {metric}_{timestamp}_{line_id}.json
# 9단계: {model_name}_v{version}_{date}.{ext}
```

### 🗂️ 폴더별 접근 권한
- **6~9단계**: 읽기/쓰기 (운영)
- **models/**: 읽기 전용 (관리자만 쓰기)
- **backups/**: 읽기 전용 (시스템만 쓰기)

### 🔄 자동 정리 정책
```python
FACTORY_CLEANUP_POLICIES = {
    '6_realtime/': {'max_age_hours': 24, 'auto_cleanup': True},
    '7_database/': {'max_age_days': 365, 'auto_cleanup': False},
    '8_monitoring/': {'max_age_days': 90, 'auto_cleanup': True},
    '9_learning/': {'max_age_days': 180, 'auto_cleanup': False}
}
```

## 🏭 공장 특화 기능

### 📊 품질 관리 시스템
- **불량 유형 분류**: 스크래치, 덴트, 크랙, 변색 등
- **품질 등급**: A급, B급, C급, 불량품
- **통계 분석**: 불량률, 품질 트렌드, 개선 지표

### 🔍 검출 알고리즘
- **다중 스케일 검출**: 다양한 크기의 불량 검출
- **실시간 분류**: 불량 유형 실시간 분류
- **신뢰도 평가**: 검출 결과 신뢰도 점수

### 📈 생산성 분석
- **처리량 모니터링**: 시간당 처리 제품 수
- **효율성 분석**: 라인별 효율성 비교
- **예측 분석**: 불량 발생 예측

## 📈 성능 지표

### 실시간 성능 목표
- **FPS**: 25 FPS 이상
- **지연시간**: 150ms 이하
- **정확도**: 98% 이상
- **메모리 사용량**: 4GB 이하

### 품질 지표
- **불량 검출률**: 99% 이상
- **오탐률**: 1% 이하
- **미탐률**: 0.5% 이하
- **분류 정확도**: 95% 이상

## 🔍 데이터 추적성

### 📋 메타데이터 관리
- **제품 정보**: 제품 ID, 배치 번호, 생산 라인
- **검출 정보**: 검출 시간, 위치, 신뢰도
- **품질 정보**: 품질 등급, 불량 유형, 심각도
- **시스템 정보**: 카메라 ID, 모델 버전, 파라미터

### 🔗 데이터 연결
- **제품 추적**: 원료 → 제품 → 검사 → 출하
- **불량 추적**: 검출 → 분류 → 분석 → 개선
- **라인 추적**: 라인별 성능 비교 및 최적화

## 🚫 절대 금지 사항

### ❌ 품질 데이터 조작 금지
```python
# 금지 사항
# - 검출 결과 인위적 수정
# - 품질 데이터 삭제 또는 변경
# - 검사 기준 임의 변경

# 올바른 방법
# - 원본 데이터 보존
# - 감사 로그 기록
# - 승인된 절차에 따른 수정
```

### ❌ 보안 위반 금지
```python
# 금지 사항
# - 제품 정보 외부 유출
# - 생산 라인 정보 노출
# - 무단 접근 허용

# 올바른 방법
# - 접근 권한 관리
# - 데이터 암호화
# - 감사 로그 기록
```

## 📊 모니터링 및 알림

### 📈 성능 모니터링
- **실시간 성능**: FPS, 지연시간, 정확도
- **시스템 리소스**: CPU, GPU, 메모리 사용량
- **품질 지표**: 불량률, 검출 정확도, 분류 정확도

### 🚨 알림 시스템
- **품질 경고**: 불량률 5% 초과 시 알림
- **성능 저하**: FPS 20 이하, 정확도 95% 이하
- **시스템 오류**: 카메라 오류, 모델 오류, 데이터베이스 오류

### 📊 품질 관리 알림
```python
QUALITY_ALERTS = {
    'defect_rate_high': {'threshold': 0.05, 'severity': 'critical'},
    'detection_accuracy_low': {'threshold': 0.95, 'severity': 'warning'},
    'system_downtime': {'threshold': 300, 'severity': 'critical'},  # 5분
    'data_integrity_error': {'threshold': 0, 'severity': 'critical'}
}
```

## 🔧 유지보수

### 🧹 정기 정리
- **매일**: 6_realtime/ 폴더 정리
- **매주**: 8_monitoring/ 데이터 정리
- **매월**: 전체 시스템 점검

### 📋 백업 정책
- **실시간**: 중요 데이터 실시간 백업
- **일일**: 전체 데이터 백업
- **주간**: 시스템 전체 백업

### 🔧 예방 정비
- **카메라 정비**: 주간 카메라 청소 및 보정
- **모델 업데이트**: 월간 모델 성능 검토
- **시스템 점검**: 분기별 전체 시스템 점검

## 🏭 공장 특화 설정

### 📊 생산 라인 설정
```python
PRODUCTION_LINES = {
    'line_1': {
        'camera_count': 4,
        'products_per_hour': 1000,
        'quality_threshold': 0.98
    },
    'line_2': {
        'camera_count': 6,
        'products_per_hour': 1500,
        'quality_threshold': 0.99
    }
}
```

### 🔍 불량 유형 정의
```python
DEFECT_TYPES = {
    'scratch': {'id': 0, 'severity': 'medium', 'threshold': 0.3},
    'dent': {'id': 1, 'severity': 'high', 'threshold': 0.4},
    'crack': {'id': 2, 'severity': 'critical', 'threshold': 0.5},
    'discoloration': {'id': 3, 'severity': 'low', 'threshold': 0.3}
}
```

## 📚 관련 문서

- [불량 검출 데이터](./defect_detection/)
- [품질 관리 가이드](./defect_detection/QUALITY_GUIDE.md)
- [성능 모니터링 가이드](./defect_detection/8_monitoring/README.md) 