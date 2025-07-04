# Output 폴더 - 시스템 출력 결과 관리

## 📋 개요

이 폴더는 비전 시스템의 모든 출력 결과를 저장합니다.
처리된 이미지, 비디오, 보고서, 내보내기 파일 등 최종 결과물들이 체계적으로 관리됩니다.

## 🏗️ 폴더 구조

```
output/
├── images/                      # 처리된 이미지
│   ├── face_recognition/       # 얼굴인식 결과
│   │   ├── detected_faces/     # 검출된 얼굴
│   │   ├── recognized_faces/   # 인식된 얼굴
│   │   └── annotated_images/   # 주석이 추가된 이미지
│   ├── defect_detection/       # 불량 검출 결과
│   │   ├── detected_defects/   # 검출된 불량
│   │   ├── classified_defects/ # 분류된 불량
│   │   └── annotated_images/   # 주석이 추가된 이미지
│   └── common/                 # 공통 처리 결과
│       ├── processed/          # 처리된 이미지
│       └── filtered/           # 필터링된 이미지
├── videos/                     # 처리된 비디오
│   ├── face_recognition/       # 얼굴인식 비디오
│   │   ├── real_time_detection/ # 실시간 검출
│   │   └── batch_processing/   # 배치 처리
│   ├── defect_detection/       # 불량 검출 비디오
│   │   ├── real_time_detection/ # 실시간 검출
│   │   └── batch_processing/   # 배치 처리
│   └── common/                 # 공통 비디오
│       ├── processed/          # 처리된 비디오
│       └── compressed/         # 압축된 비디오
├── reports/                    # 보고서
│   ├── performance/            # 성능 보고서
│   ├── statistics/             # 통계 보고서
│   └── error/                  # 오류 보고서
└── exports/                    # 내보내기 파일
    ├── csv/                    # CSV 형식
    ├── json/                   # JSON 형식
    └── xml/                    # XML 형식
```

## 📊 데이터 분류 및 용도

### 🎯 **images/ - 처리된 이미지**
- **용도**: 검출, 인식, 분류 결과 이미지
- **보관 기간**: 사용자 정의 (기본 30일)
- **크기**: 프로젝트별로 다름 (100MB~1GB)

### 🎯 **videos/ - 처리된 비디오**
- **용도**: 실시간 처리, 배치 처리 결과 비디오
- **보관 기간**: 사용자 정의 (기본 7일)
- **크기**: 프로젝트별로 다름 (1GB~10GB)

### 🎯 **reports/ - 보고서**
- **용도**: 성능 분석, 통계, 오류 보고서
- **보관 기간**: 사용자 정의 (기본 90일)
- **크기**: 프로젝트별로 다름 (10MB~100MB)

### 🎯 **exports/ - 내보내기 파일**
- **용도**: 외부 시스템 연동, 데이터 내보내기
- **보관 기간**: 사용자 정의 (기본 30일)
- **크기**: 프로젝트별로 다름 (50MB~500MB)

## 🔧 데이터 관리 규칙

### 📁 파일 네이밍 규칙
```python
# 이미지 파일 패턴
# {domain}_{task}_{timestamp}_{index}.{ext}
# 예시: face_recognition_detection_20250704_133022_001.jpg

# 비디오 파일 패턴
# {domain}_{task}_{timestamp}_{duration}.{ext}
# 예시: defect_detection_realtime_20250704_133022_30s.mp4

# 보고서 파일 패턴
# {report_type}_{date}_{domain}.{ext}
# 예시: performance_20250704_face_recognition.pdf

# 내보내기 파일 패턴
# {data_type}_{date}_{format}.{ext}
# 예시: detection_results_20250704_csv.csv
```

### 🗂️ 폴더별 접근 권한
- **images/**: 읽기/쓰기 (모든 사용자)
- **videos/**: 읽기/쓰기 (모든 사용자)
- **reports/**: 읽기/쓰기 (관리자, 분석가)
- **exports/**: 읽기/쓰기 (모든 사용자)

### 🔄 자동 정리 정책
```python
OUTPUT_CLEANUP_POLICIES = {
    'images/': {
        'max_age_days': 30,
        'max_size_mb': 1000,
        'auto_cleanup': False
    },
    'videos/': {
        'max_age_days': 7,
        'max_size_mb': 5000,
        'auto_cleanup': True
    },
    'reports/': {
        'max_age_days': 90,
        'max_size_mb': 100,
        'auto_cleanup': False
    },
    'exports/': {
        'max_age_days': 30,
        'max_size_mb': 500,
        'auto_cleanup': True
    }
}
```

## 🚫 절대 금지 사항

### ❌ 루트 디렉토리에 직접 파일 생성 금지
```python
# 금지 사항
output/result.jpg          # ❌ 금지
output/temp_file.txt       # ❌ 금지
output/debug.log           # ❌ 금지

# 올바른 방법
output/images/result.jpg   # ✅ 허용
output/reports/temp_file.txt  # ✅ 허용
output/exports/debug.csv   # ✅ 허용
```

### ❌ 하드코딩된 경로 사용 금지
```python
# 금지 사항
path = "C:/Users/user/data/output"  # ❌ 금지

# 올바른 방법
from pathlib import Path
project_root = Path(__file__).parent.parent
output_path = project_root / "data" / "output"  # ✅ 허용
```

## 📈 성능 최적화

### 💾 저장소 최적화
- **압축 알고리즘**: JPEG, PNG, MP4 압축
- **중복 제거**: 동일한 결과물 중복 제거
- **계층적 저장**: 중요도별 저장 구조

### ⚡ 접근 속도 최적화
- **인덱스 파일**: 빠른 검색을 위한 인덱스
- **캐시 시스템**: 자주 접근하는 파일 캐시
- **병렬 처리**: 대용량 파일 병렬 처리

## 🔒 보안 및 개인정보 보호

### 🔐 데이터 암호화
- **민감한 데이터**: 개인정보 포함 파일 암호화
- **전송 보안**: 외부 전송 시 암호화
- **접근 제어**: 역할 기반 접근 제어

### 🛡️ GDPR 준수
- **개인정보 제거**: 얼굴 블러, 개인정보 마스킹
- **동의 관리**: 사용자 동의 추적
- **삭제 권리**: 사용자 요청 시 즉시 삭제

## 📊 모니터링 및 알림

### 📈 저장소 모니터링
- **디스크 사용량**: 실시간 사용량 모니터링
- **파일 접근 패턴**: 사용 패턴 분석
- **성능 지표**: 처리 속도, 품질 지표

### 🚨 알림 시스템
- **용량 경고**: 80% 초과 시 알림
- **품질 저하**: 결과 품질 저하 시 알림
- **시스템 오류**: 처리 오류 시 알림

## 🔧 유지보수

### 🧹 정기 정리
- **매일**: 임시 파일 정리
- **매주**: 오래된 비디오 정리
- **매월**: 전체 데이터 품질 검사

### 📋 백업 정책
- **실시간**: 중요 결과물 실시간 백업
- **일일**: 전체 출력 데이터 백업
- **주간**: 시스템 전체 백업

## 📚 관련 문서

- [이미지 처리 가이드](./images/README.md)
- [비디오 처리 가이드](./videos/README.md)
- [보고서 생성 가이드](./reports/README.md)
- [데이터 내보내기 가이드](./exports/README.md) 