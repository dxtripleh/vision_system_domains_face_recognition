# Scripts 폴더 구조 및 관리 규칙

## 📁 표준 폴더 구조

### 최상위 구조 (도메인별 분리)
```
scripts/
├── README.md                  # 전체 스크립트 가이드
├── STRUCTURE.md              # 이 파일 - 구조 설명
├── 
├── core/                     # 🎯 핵심 시스템 스크립트
│   ├── run/                  # 실행 스크립트
│   ├── test/                 # 시스템 테스트
│   ├── validation/           # 검증 스크립트
│   └── monitoring/           # 모니터링 스크립트
│
├── domains/                  # 🎯 도메인별 스크립트
│   ├── face_recognition/     # 얼굴인식 도메인
│   ├── factory_defect/       # 공장 불량 검출 도메인 (향후)
│   └── powerline_inspection/ # 전선 검사 도메인 (향후)
│
├── development/              # 🛠️ 개발 도구 스크립트
│   ├── setup/                # 환경 설정
│   ├── model_management/     # 모델 관리
│   ├── data_processing/      # 데이터 처리
│   └── training/             # 모델 학습
│
├── deployment/               # 🚀 배포 관련 스크립트
│   ├── docker/               # Docker 설정
│   ├── cloud/                # 클라우드 배포
│   └── ci_cd/                # CI/CD 파이프라인
│
├── interfaces/               # 🖥️ 사용자 인터페이스
│   ├── web/                  # 웹 인터페이스
│   ├── cli/                  # CLI 도구
│   └── api/                  # API 관련
│
└── utilities/                # 🔧 유틸리티 스크립트
    ├── evaluation/           # 성능 평가
    ├── benchmarking/         # 벤치마킹
    └── maintenance/          # 유지보수
```

## 📋 파일 네이밍 규칙

### 실행 스크립트 (core/run/)
- **패턴**: `run_{function}_{domain}.py`
- **예시**: 
  - `run_face_recognition_demo.py`
  - `run_realtime_detection.py`
  - `run_batch_processing.py`

### 테스트 스크립트 (core/test/)
- **패턴**: `test_{component}_{type}.py`
- **예시**: 
  - `test_system_integration.py`
  - `test_performance_benchmark.py`
  - `test_models_accuracy.py`

### 도메인별 스크립트 (domains/{domain}/)
- **패턴**: `{domain}_{function}.py`
- **예시**: 
  - `face_recognition_training.py`
  - `face_recognition_evaluation.py`
  - `factory_defect_analysis.py`

### 개발 도구 (development/)
- **패턴**: `{tool}_{purpose}.py`
- **예시**: 
  - `setup_environment.py`
  - `download_models.py`
  - `process_dataset.py`

## 🚫 금지 사항

### 절대 금지
1. **scripts 루트에 .py 파일 생성 금지**
   - 모든 Python 스크립트는 하위 폴더에 배치
   - README.md, STRUCTURE.md만 루트에 허용

2. **도메인 간 의존성 금지**
   - `domains/face_recognition/` → `domains/factory_defect/` 직접 참조 금지
   - 공통 기능은 `core/` 또는 `utilities/` 사용

3. **중복 기능 스크립트 금지**
   - 같은 기능의 스크립트 여러 개 생성 금지
   - 기존 스크립트 확장 또는 통합 우선

## 📁 폴더별 역할 정의

### core/ - 핵심 시스템
- **목적**: 전체 시스템의 핵심 실행/테스트/모니터링
- **의존성**: common, shared 모듈만 참조 가능
- **예시**: 시스템 상태 체크, 전체 성능 테스트

### domains/{domain}/ - 도메인별
- **목적**: 특정 도메인의 전용 스크립트
- **의존성**: 해당 도메인 + common, shared 참조 가능
- **예시**: 얼굴인식 데모, 불량 검출 학습

### development/ - 개발 도구
- **목적**: 개발 과정에서 사용하는 도구들
- **의존성**: 전체 프로젝트 참조 가능
- **예시**: 환경 설정, 모델 다운로드

### deployment/ - 배포
- **목적**: 프로덕션 배포 관련 스크립트
- **의존성**: 배포 설정만 참조
- **예시**: Docker 빌드, 클라우드 배포

### interfaces/ - 인터페이스
- **목적**: 사용자 인터페이스 제공
- **의존성**: core, domains 참조 가능
- **예시**: 웹 서버, CLI 도구

### utilities/ - 유틸리티
- **목적**: 범용 유틸리티 기능
- **의존성**: common, shared만 참조
- **예시**: 성능 평가, 벤치마킹

## 🔄 마이그레이션 규칙

### 기존 파일 이동 규칙
1. **run_*.py** → `core/run/`
2. **test_*.py** → `core/test/`
3. **{domain}_*.py** → `domains/{domain}/`
4. **setup_*.py** → `development/setup/`
5. **웹 인터페이스** → `interfaces/web/`

### 이동 후 import 경로 수정
- 상대 경로 사용: `from ....common import`
- 절대 경로 지양: 프로젝트 루트 기준 경로 사용

## 📚 문서화 규칙

### 필수 문서 (각 폴더마다)
1. **README.md**: 폴더 내 스크립트 설명
2. **USAGE.md**: 사용법 가이드 (선택적)

### 스크립트 내 필수 요소
1. **파일 헤더**: 목적, 사용법, 예시
2. **명령줄 인자**: argparse 사용 필수
3. **로깅**: 적절한 로그 메시지
4. **에러 처리**: 예외 상황 처리

## 🔒 접근 제어

### 폴더별 접근 권한
- **core/**: 시스템 관리자만 수정
- **domains/**: 해당 도메인 개발자만 수정
- **development/**: 개발팀 전체 수정 가능
- **deployment/**: DevOps 팀만 수정
- **interfaces/**: UI/UX 팀 수정 가능
- **utilities/**: 전체 팀 수정 가능

이 구조를 통해 확장 가능하고 유지보수하기 쉬운 스크립트 관리 시스템을 구축합니다. 