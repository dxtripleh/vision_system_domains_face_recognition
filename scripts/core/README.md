# Core Scripts - 핵심 시스템 스크립트

## 📖 개요

Core 폴더는 Vision System의 핵심 시스템을 실행, 테스트, 검증, 모니터링하는 스크립트들을 포함합니다.

## 📁 폴더 구조

```
core/
├── run/                  # 🚀 주요 실행 스크립트
├── test/                 # 🧪 시스템 테스트
├── validation/           # ✅ 검증 스크립트
└── monitoring/           # 📊 모니터링 도구
```

## 🚀 주요 실행 스크립트 (run/)

### 얼굴인식 데모
```bash
# 대화형 얼굴인식 데모
python scripts/core/run/run_face_recognition.py --mode interactive --source 0

# 배치 처리 모드
python scripts/core/run/run_face_recognition.py --mode batch --input data/input/

# API 서버 모드
python scripts/core/run/run_face_recognition.py --mode api --port 8000
```

### 실시간 처리
```bash
# 실시간 카메라 데모
python scripts/core/run/run_realtime_demo.py --source 0

# 실시간 얼굴인식 시스템
python scripts/core/run/run_realtime_face_recognition.py --camera-id 0
```

## 🧪 시스템 테스트 (test/)

### 기본 시스템 테스트
```bash
# 시스템 상태 점검
python scripts/core/test/test_system_health.py

# 기본 기능 테스트
python scripts/core/test/test_basic_system.py

# 통합 시스템 테스트
python scripts/core/test/test_integrated_system.py

# 완전한 시스템 테스트
python scripts/core/test/test_complete_system.py

# 모델 테스트
python scripts/core/test/test_models.py
```

## ✅ 검증 스크립트 (validation/)

### 개발 규칙 검증
```bash
# 루트 디렉토리 보호 검증
python scripts/core/validation/validate_root_protection.py

# 자동 정리
python scripts/core/validation/validate_root_protection.py --auto-fix

# 실시간 모니터링
python scripts/core/validation/validate_root_protection.py --monitor 60
```

## 📊 모니터링 도구 (monitoring/)

### 성능 모니터링
```bash
# 시스템 성능 모니터링
python scripts/core/monitoring/performance_monitor.py

# 리소스 사용량 추적
python scripts/core/monitoring/performance_monitor.py --track-resources

# 벤치마크 실행
python scripts/core/monitoring/performance_monitor.py --benchmark
```

## 🚫 사용 시 주의사항

1. **프로젝트 루트에서 실행**: 모든 스크립트는 프로젝트 루트 디렉토리에서 실행
2. **카메라 연결**: 실시간 처리 스크립트는 카메라 연결 필요
3. **모델 파일**: 모델 파일이 `models/weights/` 폴더에 있어야 함
4. **의존성**: 필요한 패키지가 설치되어 있어야 함

## 🔧 문제 해결

### 일반적인 문제
- **Import 오류**: 프로젝트 루트에서 실행하세요
- **모델 파일 없음**: `python scripts/development/model_management/download_models.py` 실행
- **카메라 연결 실패**: USB 카메라 연결 및 드라이버 확인

### 로그 확인
```bash
# 시스템 로그
tail -f data/logs/vision_system_*.log

# 에러 로그  
tail -f data/logs/error_*.log
```

## 🤝 기여하기

새로운 핵심 스크립트를 추가할 때:
1. 적절한 하위 폴더에 배치
2. 네이밍 규칙 준수 (`run_`, `test_`, 등)
3. 명령줄 인자 지원 (argparse)
4. 적절한 로깅 및 에러 처리
5. README 업데이트 