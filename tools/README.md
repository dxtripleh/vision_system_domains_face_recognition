# 📁 TOOLS 폴더 - 시스템 도구

## 🎯 **목적**
개발, 배포, 검증, 유지보수에 필요한 시스템 도구들을 관리합니다.
프로젝트의 생산성과 품질을 향상시키기 위한 자동화 도구들이 포함되어 있습니다.

## 📂 **구조**
```
tools/
├── deployment/          # 배포 관련 도구들
├── legacy/             # 레거시 파일들 (삭제 예정)
├── setup/              # 환경 설정 도구들
├── testing/            # 테스트 도구들
└── validation/         # 검증 도구들
```

## 🚀 **주요 기능**

### 1. **환경 설정 (setup/)**
프로젝트 초기 설정 및 의존성 관리 도구들

```bash
# 모델 다운로드
python tools/setup/download_models.py

# 개발 환경 설정
python tools/setup/setup_environment.py
```

### 2. **검증 시스템 (validation/)**
프로젝트 구조, 코드 품질, 규칙 준수 검증 도구들

```bash
# 프로젝트 구조 검증 및 문서 자동 생성
python tools/validation/validate_project_structure.py

# 하드웨어 연결 검증
python tools/validation/validate_hardware_connection.py
```

### 3. **테스트 도구 (testing/)**
시스템 테스트, 성능 벤치마크, 품질 검증 도구들

```bash
# 시스템 전체 테스트
python tools/testing/test_complete_system.py

# 성능 벤치마크
python tools/testing/benchmark_performance.py
```

### 4. **배포 도구 (deployment/)**
운영 환경 배포, 모니터링, 백업 관련 도구들

```bash
# 운영 환경 배포
python tools/deployment/deploy_production.py

# 시스템 모니터링
python tools/deployment/monitor_system.py
```

### 5. **레거시 관리 (legacy/)**
이전 버전 파일들의 임시 보관소 (정리 예정)

## 🔄 **사용 패턴**

### 개발 시작 시
```bash
# 1. 환경 설정
python tools/setup/setup_environment.py

# 2. 모델 다운로드
python tools/setup/download_models.py

# 3. 시스템 검증
python tools/validation/validate_project_structure.py
```

### 개발 중
```bash
# 코드 품질 검증
python tools/validation/validate_code_quality.py

# 테스트 실행
python tools/testing/run_all_tests.py
```

### 배포 전
```bash
# 최종 검증
python tools/validation/validate_production_ready.py

# 배포 실행
python tools/deployment/deploy_production.py
```

## 📝 **개발 가이드라인**

### ✅ **허용되는 것들**
- 개발 생산성 향상 도구
- 자동화 스크립트
- 검증 및 테스트 도구
- 배포 및 유지보수 도구

### ❌ **금지되는 것들**
- 비즈니스 로직 (domains/로 이동)
- 실행 파일 (run_*.py는 도메인 내부로)
- 사용자 대상 기능 (interfaces/로 이동)

## 🔗 **관련 문서**
- [프로젝트 개요](../README.md)
- [구조 문서](STRUCTURE.md)
- [개발 환경 설정 가이드](../docs/guides/DEVELOPMENT_SETUP.md)
- [배포 가이드](../docs/guides/DEPLOYMENT_GUIDE.md)

## 💡 **초보자 팁**

### 1. **프로젝트 시작 시 순서**
```bash
# 첫 번째: 환경 설정
python tools/setup/setup_environment.py

# 두 번째: 필요한 모델 다운로드
python tools/setup/download_models.py

# 세 번째: 시스템 검증
python tools/validation/validate_project_structure.py
```

### 2. **정기적으로 실행할 것들**
- **매일**: `validate_project_structure.py` (구조 검증)
- **커밋 전**: `validate_code_quality.py` (코드 품질)
- **배포 전**: `validate_production_ready.py` (배포 준비)

### 3. **문제 해결 시**
- 환경 문제: `setup/` 폴더의 도구들 확인
- 구조 문제: `validation/` 폴더의 검증 도구 실행
- 성능 문제: `testing/` 폴더의 벤치마크 도구 사용

### 4. **새 도구 추가 시**
- 목적에 맞는 하위 폴더에 배치
- README.md에 사용법 추가
- 필요시 자동화 스크립트에 통합

## ⚠️ **주의사항**
1. **legacy/ 폴더**: 임시 보관소이므로 정기적으로 정리 필요
2. **권한 관리**: 배포 도구들은 적절한 권한 설정 필요
3. **환경 분리**: 개발/테스트/운영 환경을 명확히 구분
4. **백업**: 중요한 도구들은 버전 관리 필수

---
*이 문서는 프로젝트 구조 검증 시스템에 의해 자동 생성되고 수동으로 개선되었습니다.*
