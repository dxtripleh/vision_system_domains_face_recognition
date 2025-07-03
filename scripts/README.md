# Scripts 폴더 가이드

## 📖 개요

이 폴더는 Vision System 프로젝트의 모든 실행 스크립트, 테스트, 개발 도구들을 체계적으로 관리합니다.

## 🚀 빠른 시작

### 시스템 상태 확인
```bash
# 시스템 전체 상태 점검
python scripts/core/test/test_system_health.py

# 기본 시스템 테스트
python scripts/core/test/test_basic_system.py
```

### 얼굴인식 실행
```bash
# 대화형 얼굴인식 데모
python scripts/core/run/run_face_recognition_demo.py --mode interactive

# 실시간 카메라 데모
python scripts/core/run/run_realtime_detection.py --source 0

# 웹 인터페이스 시작
python scripts/interfaces/web/app.py
```

## 📁 폴더 구조

```
scripts/
├── 📋 README.md              # 이 파일 - 전체 가이드
├── 📋 STRUCTURE.md           # 상세 구조 및 규칙
│
├── 🎯 core/                  # 핵심 시스템 스크립트
│   ├── run/                  # 메인 실행 스크립트들
│   ├── test/                 # 시스템 테스트
│   ├── validation/           # 검증 스크립트
│   └── monitoring/           # 모니터링 도구
│
├── 🎯 domains/               # 도메인별 전용 스크립트
│   └── face_recognition/     # 얼굴인식 도메인
│
├── 🛠️ development/          # 개발 도구
│   ├── setup/                # 환경 설정
│   ├── model_management/     # 모델 관리
│   ├── data_processing/      # 데이터 처리
│   └── training/             # 모델 학습
│
├── 🚀 deployment/           # 배포 관련
│   ├── docker/               # Docker 설정
│   ├── cloud/                # 클라우드 배포
│   └── ci_cd/                # CI/CD 파이프라인
│
├── 🖥️ interfaces/           # 사용자 인터페이스
│   ├── web/                  # 웹 인터페이스
│   ├── cli/                  # CLI 도구
│   └── api/                  # API 관련
│
└── 🔧 utilities/            # 유틸리티 도구
    ├── evaluation/           # 성능 평가
    ├── benchmarking/         # 벤치마킹
    └── maintenance/          # 유지보수
```

## 📋 주요 스크립트 목록

### 🎯 핵심 실행 스크립트 (core/run/)
- `run_face_recognition_demo.py` - 얼굴인식 데모 실행
- `run_realtime_detection.py` - 실시간 검출 시스템
- `run_batch_processing.py` - 배치 처리 시스템

### 🧪 테스트 스크립트 (core/test/)
- `test_system_health.py` - 전체 시스템 상태 점검
- `test_basic_system.py` - 기본 시스템 기능 테스트
- `test_performance_benchmark.py` - 성능 벤치마크

### 🔧 개발 도구 (development/)
- `setup/setup_environment.py` - 개발 환경 설정
- `model_management/download_models.py` - 모델 다운로드
- `data_processing/process_dataset.py` - 데이터셋 처리

### 🖥️ 인터페이스 (interfaces/)
- `web/app.py` - 웹 인터페이스 서버
- `cli/face_recognition_cli.py` - CLI 도구
- `api/face_recognition_api.py` - REST API 서버

## 🛠️ 사용법

### 환경 설정
```bash
# 개발 환경 설정
python scripts/development/setup/setup_environment.py

# 모델 다운로드
python scripts/development/model_management/download_models.py --all
```

### 기본 실행
```bash
# 시스템 검증
python scripts/core/validation/validate_system.py

# 얼굴인식 실행
python scripts/core/run/run_face_recognition_demo.py --help
```

### 개발 및 테스트
```bash
# 성능 모니터링
python scripts/core/monitoring/performance_monitor.py

# 정확도 평가
python scripts/utilities/evaluation/evaluate_accuracy.py
```

## 📚 추가 문서

- **[STRUCTURE.md](STRUCTURE.md)**: 상세 폴더 구조 및 규칙
- **각 하위 폴더의 README.md**: 해당 폴더별 상세 가이드

## 🚫 주의사항

1. **scripts 루트에 .py 파일 생성 금지**
   - 모든 Python 스크립트는 적절한 하위 폴더에 배치

2. **명명 규칙 준수**
   - 실행 스크립트: `run_{function}_{domain}.py`
   - 테스트 스크립트: `test_{component}_{type}.py`

3. **의존성 규칙 준수**
   - 도메인 간 직접 의존성 금지
   - 공통 기능은 core 또는 utilities 사용

## 🆘 문제 해결

### 일반적인 문제
1. **Import 오류**: 프로젝트 루트에서 실행하세요
2. **모델 파일 없음**: `download_models.py` 실행
3. **카메라 연결 실패**: USB 카메라 연결 확인

### 로그 확인
```bash
# 시스템 로그
tail -f data/logs/vision_system_*.log

# 에러 로그
tail -f data/logs/error_*.log
```

## 🤝 기여하기

새로운 스크립트를 추가할 때:

1. **적절한 폴더에 배치**
2. **네이밍 규칙 준수**
3. **문서화 완료** (docstring, 사용법)
4. **테스트 코드 작성**
5. **README 업데이트**

자세한 규칙은 [STRUCTURE.md](STRUCTURE.md)를 참조하세요. 