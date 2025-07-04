# Temp 폴더 - 임시 파일 관리

## 📋 개요

이 폴더는 비전 시스템의 모든 임시 파일을 저장합니다.
처리 중인 데이터, 캐시 파일, 임시 결과물 등이 자동으로 정리되며 관리됩니다.

## 🏗️ 폴더 구조

```
temp/
├── processing/                   # 처리 중인 파일
│   ├── face_recognition/        # 얼굴인식 처리 중
│   │   ├── frames/              # 프레임 데이터
│   │   ├── features/            # 특이점 데이터
│   │   └── results/             # 임시 결과
│   ├── defect_detection/        # 불량 검출 처리 중
│   │   ├── frames/              # 프레임 데이터
│   │   ├── detections/          # 검출 데이터
│   │   └── results/             # 임시 결과
│   └── common/                  # 공통 처리 중
│       ├── uploads/             # 업로드 중인 파일
│       ├── downloads/           # 다운로드 중인 파일
│       └── conversions/         # 변환 중인 파일
├── cache/                       # 캐시 파일
│   ├── models/                  # 모델 캐시
│   ├── images/                  # 이미지 캐시
│   ├── videos/                  # 비디오 캐시
│   └── data/                    # 데이터 캐시
├── sessions/                    # 세션 데이터
│   ├── user_sessions/           # 사용자 세션
│   ├── system_sessions/         # 시스템 세션
│   └── api_sessions/            # API 세션
└── cleanup/                     # 정리 대기 파일
    ├── scheduled/               # 예약 정리
    ├── failed/                  # 정리 실패
    └── manual/                  # 수동 정리
```

## 📊 데이터 분류 및 용도

### 🎯 **processing/ - 처리 중인 파일**
- **용도**: 현재 처리 중인 데이터, 임시 결과물
- **보관 기간**: 처리 완료 후 자동 삭제 (최대 24시간)
- **크기**: 프로젝트별로 다름 (100MB~2GB)

### 🎯 **cache/ - 캐시 파일**
- **용도**: 성능 향상을 위한 캐시 데이터
- **보관 기간**: 12시간 (자동 정리)
- **크기**: 프로젝트별로 다름 (500MB~5GB)

### 🎯 **sessions/ - 세션 데이터**
- **용도**: 사용자 세션, 시스템 세션 데이터
- **보관 기간**: 세션 종료 후 삭제 (최대 8시간)
- **크기**: 프로젝트별로 다름 (50MB~500MB)

### 🎯 **cleanup/ - 정리 대기 파일**
- **용도**: 정리 대기 중인 파일들
- **보관 기간**: 정리 완료 후 삭제
- **크기**: 프로젝트별로 다름 (10MB~100MB)

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# 임시 파일 패턴
# {purpose}_{timestamp}_{random_id}.{ext}
# 예시: processing_20250704_133022_abc123.tmp

# 캐시 파일 패턴
# {component}_{hash}_{version}.{ext}
# 예시: model_face_detection_a1b2c3_v1.0.cache

# 세션 파일 패턴
# {session_type}_{session_id}_{timestamp}.{ext}
# 예시: user_session_sess123_20250704_133022.json
```

### 🗂️ 폴더별 접근 권한
- **processing/**: 읽기/쓰기 (처리 프로세스)
- **cache/**: 읽기/쓰기 (시스템)
- **sessions/**: 읽기/쓰기 (세션 관리자)
- **cleanup/**: 읽기/쓰기 (정리 시스템)

### 🔄 자동 정리 정책
```python
TEMP_CLEANUP_POLICIES = {
    'processing/': {
        'max_age_hours': 24,
        'max_size_mb': 2000,
        'auto_cleanup': True
    },
    'cache/': {
        'max_age_hours': 12,
        'max_size_mb': 5000,
        'auto_cleanup': True
    },
    'sessions/': {
        'max_age_hours': 8,
        'max_size_mb': 500,
        'auto_cleanup': True
    },
    'cleanup/': {
        'max_age_hours': 1,
        'max_size_mb': 100,
        'auto_cleanup': True
    }
}
```

## 🚫 절대 금지 사항

### ❌ 루트 디렉토리에 직접 파일 생성 금지
```python
# 금지 사항
temp/file.txt           # ❌ 금지
temp/result.jpg         # ❌ 금지
temp/debug.log          # ❌ 금지

# 올바른 방법
temp/processing/file.txt    # ✅ 허용
temp/cache/result.jpg       # ✅ 허용
temp/sessions/debug.log     # ✅ 허용
```

### ❌ 영구 저장용 파일 생성 금지
```python
# 금지 사항
# - 중요한 데이터를 temp 폴더에 영구 저장
# - 백업 파일을 temp 폴더에 저장
# - 설정 파일을 temp 폴더에 저장

# 올바른 방법
# - 중요한 데이터는 data/output/ 또는 data/domains/ 사용
# - 백업 파일은 data/backups/ 사용
# - 설정 파일은 config/ 사용
```

### ❌ 하드코딩된 경로 사용 금지
```python
# 금지 사항
path = "C:/Users/user/data/temp"  # ❌ 금지

# 올바른 방법
from pathlib import Path
project_root = Path(__file__).parent.parent
temp_path = project_root / "data" / "temp"  # ✅ 허용
```

## 📈 성능 최적화

### 💾 저장소 최적화
- **압축 알고리즘**: 임시 파일 압축 저장
- **중복 제거**: 동일한 임시 파일 중복 제거
- **계층적 정리**: 중요도별 정리 우선순위

### ⚡ 접근 속도 최적화
- **메모리 캐시**: 자주 접근하는 임시 파일 메모리 캐시
- **SSD 최적화**: SSD 기반 빠른 읽기/쓰기
- **병렬 처리**: 대용량 임시 파일 병렬 처리

## 🔒 보안 및 개인정보 보호

### 🔐 임시 파일 암호화
- **민감한 데이터**: 개인정보 포함 임시 파일 암호화
- **세션 데이터**: 사용자 세션 데이터 암호화
- **접근 제어**: 임시 파일 접근 권한 관리

### 🛡️ GDPR 준수
- **개인정보 제거**: 임시 파일에서 개인정보 자동 제거
- **세션 관리**: 사용자 세션 데이터 보안 관리
- **삭제 보장**: 정리 시 완전한 데이터 삭제

## 📊 모니터링 및 알림

### 📈 임시 파일 모니터링
- **디스크 사용량**: 실시간 임시 파일 크기 모니터링
- **정리 효율성**: 자동 정리 성능 모니터링
- **성능 지표**: 임시 파일 처리 속도

### 🚨 알림 시스템
- **용량 경고**: 임시 파일 크기 80% 초과 시 알림
- **정리 실패**: 자동 정리 실패 시 알림
- **보안 이벤트**: 무단 접근 시도 시 알림

## 🔧 유지보수

### 🧹 정기 정리
- **매시간**: cleanup/ 폴더 정리
- **매일**: processing/, cache/ 폴더 정리
- **매주**: 전체 temp 폴더 점검

### 📋 백업 정책
- **백업 없음**: 임시 파일은 백업하지 않음
- **복구 불가**: 정리된 임시 파일은 복구 불가
- **재생성**: 필요시 임시 파일 재생성

## 🔄 자동 정리 시스템

### ⏰ 정리 스케줄
```python
CLEANUP_SCHEDULE = {
    'cleanup/': '매시간',
    'sessions/': '매 8시간',
    'cache/': '매 12시간',
    'processing/': '매 24시간'
}
```

### 🧹 정리 방법
```python
def cleanup_temp_files():
    """임시 파일 자동 정리"""
    for folder, policy in TEMP_CLEANUP_POLICIES.items():
        cleanup_folder(folder, policy)

def cleanup_folder(folder: str, policy: Dict):
    """폴더별 정리"""
    current_time = time.time()
    max_age_seconds = policy['max_age_hours'] * 3600
    
    for file_path in Path(folder).rglob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()  # 파일 삭제
```

## 📊 임시 파일 통계

### 📈 사용량 통계
- **총 임시 파일 수**: 실시간 모니터링
- **폴더별 사용량**: 각 폴더별 사용량 통계
- **정리 효율성**: 자동 정리 성공률

### 📊 성능 지표
- **처리 속도**: 임시 파일 생성/삭제 속도
- **메모리 사용량**: 임시 파일 관련 메모리 사용량
- **디스크 I/O**: 임시 파일 읽기/쓰기 성능

## 📚 관련 문서

- [처리 중 파일 가이드](./processing/README.md)
- [캐시 관리 가이드](./cache/README.md)
- [세션 관리 가이드](./sessions/README.md)
- [자동 정리 가이드](./cleanup/README.md) 