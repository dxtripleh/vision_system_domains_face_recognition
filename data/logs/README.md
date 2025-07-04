# Logs 폴더 - 시스템 로그 관리

## 📋 개요

이 폴더는 비전 시스템의 모든 로그 파일을 저장합니다.
시스템 로그, 애플리케이션 로그, 오류 로그, 성능 로그 등이 체계적으로 관리됩니다.

## 🏗️ 폴더 구조

```
logs/
├── system/                      # 시스템 로그
│   ├── general/                 # 일반 시스템 로그
│   ├── error/                   # 시스템 오류 로그
│   ├── performance/             # 성능 로그
│   └── security/                # 보안 로그
├── applications/                # 애플리케이션 로그
│   ├── face_recognition/       # 얼굴인식 로그
│   │   ├── detection/          # 검출 로그
│   │   ├── recognition/        # 인식 로그
│   │   └── errors/             # 오류 로그
│   ├── defect_detection/       # 불량 검출 로그
│   │   ├── detection/          # 검출 로그
│   │   ├── classification/     # 분류 로그
│   │   └── errors/             # 오류 로그
│   └── common/                 # 공통 애플리케이션 로그
│       ├── pipeline/           # 파이프라인 로그
│       ├── database/           # 데이터베이스 로그
│       └── api/                # API 로그
├── monitoring/                  # 모니터링 로그
│   ├── metrics/                # 성능 지표 로그
│   ├── alerts/                 # 알림 로그
│   └── health/                 # 시스템 상태 로그
└── archive/                    # 아카이브된 로그
    ├── daily/                  # 일일 아카이브
    ├── weekly/                 # 주간 아카이브
    └── monthly/                # 월간 아카이브
```

## 📊 데이터 분류 및 용도

### 🎯 **system/ - 시스템 로그**
- **용도**: 운영체제, 하드웨어, 시스템 서비스 로그
- **보관 기간**: 30일 (자동 로테이션)
- **크기**: 일일 10MB~50MB

### 🎯 **applications/ - 애플리케이션 로그**
- **용도**: 비전 시스템 애플리케이션 로그
- **보관 기간**: 7일 (도메인별)
- **크기**: 일일 50MB~200MB

### 🎯 **monitoring/ - 모니터링 로그**
- **용도**: 성능 모니터링, 알림, 시스템 상태 로그
- **보관 기간**: 90일
- **크기**: 일일 20MB~100MB

### 🎯 **archive/ - 아카이브된 로그**
- **용도**: 오래된 로그 파일 보관
- **보관 기간**: 1년
- **크기**: 압축된 형태로 저장

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# 로그 파일 패턴
# {component}_{date}_{level}.log
# 예시: face_recognition_20250704_info.log

# 압축된 로그 패턴
# {component}_{date}_{level}.log.gz
# 예시: system_20250704_error.log.gz

# 아카이브 패턴
# {component}_{period}_{date}.tar.gz
# 예시: applications_weekly_20250704.tar.gz
```

### 🗂️ 폴더별 접근 권한
- **system/**: 읽기 전용 (관리자만 쓰기)
- **applications/**: 읽기/쓰기 (애플리케이션)
- **monitoring/**: 읽기/쓰기 (모니터링 시스템)
- **archive/**: 읽기 전용 (시스템만 쓰기)

### 🔄 자동 정리 정책
```python
LOGS_CLEANUP_POLICIES = {
    'system/': {
        'max_age_days': 30,
        'max_size_mb': 1000,
        'auto_rotation': True
    },
    'applications/': {
        'max_age_days': 7,
        'max_size_mb': 2000,
        'auto_rotation': True
    },
    'monitoring/': {
        'max_age_days': 90,
        'max_size_mb': 1000,
        'auto_rotation': True
    },
    'archive/': {
        'max_age_days': 365,
        'max_size_mb': 10000,
        'auto_compression': True
    }
}
```

## 🚫 절대 금지 사항

### ❌ 개인정보 포함 로그 생성 금지
```python
# 금지 사항
# - 개인정보가 포함된 로그 생성
# - 암호화되지 않은 민감한 정보 로그
# - 사용자 동의 없이 개인정보 수집

# 올바른 방법
# - 개인정보 제거 후 로그 생성
# - 암호화된 형태로 로그 저장
# - 동의된 정보만 로그에 포함
```

### ❌ 하드코딩된 경로 사용 금지
```python
# 금지 사항
path = "C:/Users/user/data/logs"  # ❌ 금지

# 올바른 방법
from pathlib import Path
project_root = Path(__file__).parent.parent
logs_path = project_root / "data" / "logs"  # ✅ 허용
```

## 📈 성능 최적화

### 💾 저장소 최적화
- **로그 압축**: gzip, bzip2 압축 사용
- **로그 로테이션**: 크기/시간 기반 자동 로테이션
- **중복 제거**: 동일한 로그 메시지 중복 제거

### ⚡ 접근 속도 최적화
- **인덱스 생성**: 로그 검색을 위한 인덱스
- **캐시 시스템**: 자주 접근하는 로그 캐시
- **병렬 처리**: 대용량 로그 병렬 처리

## 🔒 보안 및 개인정보 보호

### 🔐 로그 암호화
- **민감한 로그**: 개인정보 포함 로그 암호화
- **전송 보안**: 로그 전송 시 TLS 사용
- **접근 제어**: 역할 기반 로그 접근 제어

### 🛡️ GDPR 준수
- **개인정보 제거**: 로그에서 개인정보 자동 제거
- **동의 관리**: 로그 수집 동의 추적
- **삭제 권리**: 사용자 요청 시 로그 삭제

## 📊 모니터링 및 알림

### 📈 로그 모니터링
- **로그 크기**: 실시간 로그 크기 모니터링
- **로그 패턴**: 비정상적인 로그 패턴 감지
- **성능 지표**: 로그 처리 속도, 오류율

### 🚨 알림 시스템
- **로그 오버플로우**: 로그 크기 초과 시 알림
- **오류 증가**: 오류 로그 증가 시 알림
- **보안 이벤트**: 보안 관련 로그 발생 시 알림

## 🔧 유지보수

### 🧹 정기 정리
- **매일**: 로그 로테이션 및 압축
- **매주**: 오래된 로그 아카이브
- **매월**: 로그 품질 검사 및 정리

### 📋 백업 정책
- **실시간**: 중요 로그 실시간 백업
- **일일**: 전체 로그 백업
- **주간**: 로그 아카이브 백업

## 📊 로그 레벨 및 형식

### 📋 로그 레벨
```python
LOG_LEVELS = {
    'DEBUG': 10,      # 디버그 정보
    'INFO': 20,       # 일반 정보
    'WARNING': 30,    # 경고
    'ERROR': 40,      # 오류
    'CRITICAL': 50    # 심각한 오류
}
```

### 📝 로그 형식
```python
LOG_FORMAT = {
    'timestamp': '%Y-%m-%d %H:%M:%S',
    'level': 'INFO',
    'component': 'face_recognition',
    'message': 'Face detected with confidence 0.95',
    'metadata': {
        'user_id': 'anonymous',
        'session_id': 'sess_12345',
        'performance_ms': 45.2
    }
}
```

## 🔍 로그 분석 도구

### 📊 로그 분석 스크립트
```python
# 로그 분석 예시
def analyze_logs(log_path: str) -> Dict:
    """로그 파일 분석"""
    analysis = {
        'total_entries': 0,
        'error_count': 0,
        'warning_count': 0,
        'performance_avg_ms': 0.0,
        'most_common_errors': []
    }
    # 로그 분석 로직
    return analysis
```

### 📈 로그 시각화
- **시간별 트렌드**: 오류율, 성능 지표 시각화
- **컴포넌트별 분석**: 도메인별 로그 분석
- **실시간 대시보드**: 실시간 로그 모니터링

## 📚 관련 문서

- [시스템 로그 가이드](./system/README.md)
- [애플리케이션 로그 가이드](./applications/README.md)
- [모니터링 로그 가이드](./monitoring/README.md)
- [로그 분석 가이드](./archive/README.md) 