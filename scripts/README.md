# Scripts 폴더

이 폴더는 프로젝트 개발 및 유지보수를 위한 유틸리티 스크립트들을 포함합니다.

## 📁 폴더 구조

```
scripts/
├── __init__.py                 # Python 패키지 초기화
├── README.md                   # 이 파일
├── validate_file_locations.py  # 파일 위치 검증 스크립트
├── setup_file_protection.py    # 파일 보호 시스템 설정
├── watch_files.py              # 파일 감시자 (자동 생성)
├── setup_coding_tools.py       # 코딩 도구 설정
├── validate_rules.py           # 프로젝트 규칙 검증
├── download_models.py          # 모델 다운로드
├── cleanup_temp_files.py       # 임시 파일 정리
└── setup_git_hooks.py          # Git 훅 설정
```

## 🔒 파일 보호 시스템

### 주요 스크립트

#### 1. `validate_file_locations.py`
**목적**: 최상위 루트에 임시 파일이 생성되는 것을 방지

**사용법**:
```bash
# 현재 상태 검증
python scripts/validate_file_locations.py

# 실시간 모니터링
python scripts/validate_file_locations.py --monitor

# Git pre-commit 훅 생성
python scripts/validate_file_locations.py --create-hook

# Git pre-commit 훅 모드
python scripts/validate_file_locations.py --pre-commit
```

**기능**:
- 금지된 파일 패턴 감지 (`.log`, `.tmp`, `.jpg`, `.png` 등)
- 올바른 파일 위치 제안
- 자동 파일 이동 (선택적)
- Git pre-commit 훅 통합

#### 2. `setup_file_protection.py`
**목적**: 파일 보호 시스템 자동 설정

**사용법**:
```bash
python scripts/setup_file_protection.py
```

**설정 내용**:
- Git pre-commit 훅 생성
- VS Code 설정 파일 생성
- pre-commit 설정 파일 생성
- 파일 감시자 스크립트 생성
- 개발 가이드 생성

#### 3. `watch_files.py`
**목적**: 실시간 파일 생성 모니터링

**사용법**:
```bash
python scripts/watch_files.py
```

**기능**:
- 실시간 파일 생성 감지
- 잘못된 위치의 파일 자동 알림
- 자동 파일 이동 (환경변수 설정 시)

## 🛠️ 개발 도구 스크립트

### 코딩 도구 설정
```bash
# 코딩 도구 자동 설치 및 설정
python scripts/setup_coding_tools.py
```

**설치 도구**:
- black (코드 포맷팅)
- isort (import 정렬)
- flake8 (린팅)
- pylint (정적 분석)
- mypy (타입 체킹)

### 프로젝트 규칙 검증
```bash
# 프로젝트 규칙 준수 검증
python scripts/validate_rules.py
```

**검증 항목**:
- 파일 네이밍 규칙
- Import 순서
- Docstring 작성
- Type Hints 사용
- 프로젝트 구조

## 📦 모델 관리 스크립트

### 모델 다운로드
```bash
# 모델 자동 다운로드
python scripts/download_models.py
```

**기능**:
- ONNX 모델 다운로드
- 모델 메타데이터 생성
- 모델 검증

### 임시 파일 정리
```bash
# 임시 파일 자동 정리
python scripts/cleanup_temp_files.py
```

**정리 대상**:
- `data/temp/` 폴더의 오래된 파일
- `data/logs/` 폴더의 오래된 로그
- 임시 백업 파일

## 🔧 Git 훅 설정

### Git 훅 자동 설정
```bash
# Git 훅 설정
python scripts/setup_git_hooks.py
```

**설정 훅**:
- pre-commit: 파일 위치 검증, 코드 포맷팅
- pre-push: 테스트 실행, 보안 검사
- commit-msg: 커밋 메시지 검증

## 📋 스크립트 카테고리

### Setup 스크립트
- `setup_*.py`: 개발 환경 설정
- `install_*.py`: 도구 설치
- `configure_*.py`: 설정 파일 생성

### Validation 스크립트
- `validate_*.py`: 규칙 검증
- `check_*.py`: 상태 확인
- `test_*.py`: 테스트 실행

### Data 스크립트
- `download_*.py`: 데이터 다운로드
- `process_*.py`: 데이터 처리
- `cleanup_*.py`: 데이터 정리

### Model 스크립트
- `download_models.py`: 모델 다운로드
- `convert_models.py`: 모델 변환
- `benchmark_models.py`: 모델 벤치마크

### Utils 스크립트
- `backup_*.py`: 백업 도구
- `migrate_*.py`: 마이그레이션
- `deploy_*.py`: 배포 도구

## 🚀 사용 예시

### 1. 새 개발 환경 설정
```bash
# 1. 파일 보호 시스템 설정
python scripts/setup_file_protection.py

# 2. 코딩 도구 설치
python scripts/setup_coding_tools.py

# 3. Git 훅 설정
python scripts/setup_git_hooks.py

# 4. 모델 다운로드
python scripts/download_models.py
```

### 2. 개발 중 파일 위치 검증
```bash
# 수동 검증
python scripts/validate_file_locations.py

# 실시간 모니터링 (별도 터미널에서)
python scripts/watch_files.py
```

### 3. 정기적인 정리 작업
```bash
# 임시 파일 정리
python scripts/cleanup_temp_files.py

# 프로젝트 규칙 검증
python scripts/validate_rules.py
```

## ⚙️ 환경 변수 설정

### 파일 보호 시스템
```bash
# 자동 파일 이동 활성화
export AUTO_MOVE_FILES=true

# 자동 수정 활성화
export AUTO_FIX_FILE_LOCATION=true
```

### 개발 도구
```bash
# 로그 레벨 설정
export LOG_LEVEL=DEBUG

# 테스트 모드
export TEST_MODE=true
```

## 🔍 문제 해결

### 파일 위치 위반 시
1. **수동 검증**: `python scripts/validate_file_locations.py`
2. **자동 수정**: `export AUTO_MOVE_FILES=true && python scripts/validate_file_locations.py --monitor`
3. **코드 수정**: `common.file_utils` 모듈 사용

### 스크립트 실행 오류 시
1. **권한 확인**: `chmod +x scripts/*.py`
2. **의존성 확인**: `pip install -r requirements.txt`
3. **Python 경로 확인**: `export PYTHONPATH=.`

### Git 훅 오류 시
1. **훅 재설정**: `python scripts/setup_git_hooks.py`
2. **pre-commit 재설치**: `pre-commit install`
3. **수동 실행**: `pre-commit run --all-files`

## 📚 관련 문서

- [파일 위치 규칙 가이드](../docs/file_location_guide.md)
- [개발 환경 설정 가이드](../docs/development_setup.md)
- [Git 훅 사용법](../docs/git_hooks.md)
- [모델 관리 가이드](../docs/model_management.md)

## �� 기여 가이드

### 새 스크립트 추가 시
1. **네이밍 규칙**: `{action}_{target}.py`
2. **문서화**: 스크립트 상단에 docstring 작성
3. **에러 처리**: 적절한 예외 처리 포함
4. **로깅**: `common.logging` 모듈 사용
5. **테스트**: 단위 테스트 작성

### 스크립트 수정 시
1. **기존 기능 유지**: 하위 호환성 보장
2. **문서 업데이트**: README.md 수정
3. **테스트 실행**: 기존 테스트 통과 확인
4. **코드 리뷰**: 다른 개발자 검토

---

**참고**: 모든 스크립트는 프로젝트 루트에서 실행해야 합니다. 